# End-to-End DoH Command & Control (C2) Detection & Model Optimization

## Project Overview
This project focuses on the detection of malicious Command and Control (C2) malware communications disguised within encrypted DNS-over-HTTPS (DoH) tunnels. Traditional firewalls fail to detect DoH C2 traffic because it is encrypted within standard HTTPS traffic. This project solves this by using **flow-based behavioral analysis** instead of signature-based detection.

The project is divided into three major phases:
1. **Network Simulation & Feature Extraction:** Synthesizing benign and malicious traffic using Python's `scapy`, and translating those raw packets into mathematical behavioral flows using `CICFlowMeter`.
2. **Adversarial Evasion (DoH Deception):** Testing the robustness of the AI by simulating advanced attackers manipulating their statistical footprints to bypass flow-based detection.
3. **Model Optimization & Real-Time Profiling:** Rigorously comparing multiple tree-based ensemble models (Random Forest, Gradient Boosting, XGBoost) and profiling their hardware efficiency (latency and throughput) for live network deployment.

## Project Structure
- `NSP_Master_Notebook.ipynb`
  The core pipeline. Generates simulated PCAPs, extracts features into a labeled CSV, trains the baseline Random Forest model, and performs the adversarial evasion assessment. *(Note: Must be run in a Linux environment like Google Colab due to `tcpdump` and `cicflowmeter` dependencies).*
- `Project_Visualizations.ipynb`
  Reads the generated datasets and creates all the necessary visual matrices for the report, including Confusion Matrices, ROC/AUC Curves, and Feature Importance Graphs.
- `RealTime_Optimization.ipynb`
  The final phase comparing Random Forest, Gradient Boosting, and XGBoost. Includes a real-time streaming simulation to benchmark Average Latency (ms) and Throughput (Flows/Sec) to determine production viability.
- `labeled_doh_dataset.csv` / `raw_features.csv`
  The output statistical flow datasets extracted from the raw packet captures.
- `archive/`
  Contains the authentic university big-data parquet files (`CIRA-CIC-DoHBrw-2020`) used for cross-distribution validation against the simulated data.

## Requirements
To run this project, you will need the following Python libraries:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn scapy pyarrow
```
*Additionally, `cicflowmeter` and `tcpdump` are required for the packet-to-flow extraction phase.*

## Usage Instructions
1. **Data Generation & Baseline:** Upload `NSP_Master_Notebook.ipynb` to Google Colab. Run all cells to simulate the network traffic, extract features into `labeled_doh_dataset.csv`, and execute the evasion attacks.
2. **Visualizations:** Download the generated `.csv` files locally and run `Project_Visualizations.ipynb` to generate the evaluation graphs.
3. **Model Profiling:** Open `RealTime_Optimization.ipynb` and execute the cells to evaluate the latency and throughput differences between the different Gradient Boosting algorithms.

## Key Results & Contributions
Our model optimization and real-time hardware profiling proved that while a standard Random Forest is highly accurate at detecting DoH malware, it is too slow for a live firewall (~5.16ms latency). We empirically justified that **Gradient Boosting** and **XGBoost** are the superior models for live gateways, as they maintain near 100% detection accuracy while operating at ultra-low latencies (< 0.5ms) and handling enterprise-level throughputs of over 7,000 flows per second.
