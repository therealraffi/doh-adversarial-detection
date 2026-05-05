# doh-adversarial-detection

Research project studying DNS-over-HTTPS (DoH) traffic as a C2 channel and building ML-based detectors to identify it.

```
doh-adversarial-detection/
├── c2-lab/       # Docker lab for generating labeled C2/benign DoH captures
└── detectors/    # ML pipeline for training and evaluating DoH detectors
```

---

## c2-lab

A Dockerized environment for simulating DNS/DoH-based C2 traffic using [Sliver](https://github.com/BishopFox/sliver). Used to generate labeled packet captures for detector training and evaluation.

### Prerequisites

- Docker + Docker Compose
- `tcpdump` on the host
- `sliver-client` installed locally

### Network Layout

| Container  | IP            | Role                        |
| ---------- | ------------- | --------------------------- |
| `resolver` | 172.20.0.10   | CoreDNS (DoT/DoH forwarder) |
| `attacker` | 172.20.0.20   | Sliver C2 server            |
| `victim`   | 172.20.0.30   | Ubuntu host running implant |

### Setup

```bash
cd c2-lab
docker compose up -d
```

Connect the Sliver client to the attacker container:

```bash
docker exec -it c2-lab-attacker-1 sliver-server
```

If the Sliver server isn't running yet, start it inside the container first, then connect with `sliver-client` from your host.

### Running a Capture Session

1. **Find the project bridge** (run on host):

   ```bash
   docker network inspect c2-lab_c2net --format '{{.Id}}' | cut -c1-12
   # bridge will be br-<id>
   ```

2. **Start recording** on host (replace `br-XXXX` with your bridge):

   ```bash
   tcpdump -i br-XXXX -w captures/<scenario_name>.pcap
   ```

3. **Start the DNS listener** inside Sliver:

   ```
   dns --domains lab.local.
   ```

4. **Force victim DNS to attacker** (if resolver is bypassed):

   ```bash
   docker exec c2-lab-victim-1 bash -c \
     "echo 'nameserver 172.20.0.20' > /etc/resolv.conf && \
      echo '172.20.0.20 lab.local' >> /etc/hosts"
   ```

5. **Run the implant** on the victim:

   ```bash
   docker exec c2-lab-victim-1 /implants/doh_implant
   ```

6. Perform the scenario interactions, then kill the implant and stop `tcpdump`.

### Capture Scenarios

| File                   | Label  | Description                    |
| ---------------------- | ------ | ------------------------------ |
| `01_quiet_beacon.pcap` | C2     | Idle beacon only (~10 min)     |
| `02_discovery.pcap`    | C2     | `ls -R`, `ps`, `whoami`        |
| `03_exfiltration.pcap` | C2     | `download` of a large file     |
| `04_benign_web.pcap`   | Benign | `curl` loop to legitimate URLs |
| `05_lateral_scan.pcap` | C2     | Internal `ping`/`netstat`      |

### Teardown

```bash
docker compose down
```

---

## detectors

ML pipeline that trains and evaluates classifiers (Random Forest, Gradient Boosting, XGBoost, MLP) on flow-level features extracted from DoH traffic.

### Setup

```bash
cd detectors
python -m venv netenv && source netenv/bin/activate
pip install -r requirements.txt   # if present, else install manually
```

### Data

Place raw CSVs under `detectors/data/`. The pipeline expects the combined DoH dataset (e.g. `DoH-combined.zip` extracted in place). See `detectors/data/README.txt` for provenance details.

### Running

```bash
python detector.py
```

Trained model artifacts and evaluation plots are written to `detectors/results/`.
