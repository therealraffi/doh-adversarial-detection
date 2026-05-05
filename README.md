\section{DoHShield: DoH Tunnel Detection and Adversarial Evaluation}

This repository contains a flow-based machine learning framework for detecting DNS-over-HTTPS (DoH) tunnel and C2 traffic, along with adversarial pipelines that test whether those detectors can be evaded.

The project uses the Hokkaido University combined dataset, which includes CIRA-CIC-DoHBrw-2020 and DoH-Tunnel-Traffic-HKD. The detector pipeline trains classifiers on flow-level network features, while the adversarial pipelines generate synthetic DoH C2 traffic, extract the same feature set, and test whether the trained models still detect the traffic.

\subsection{Overview}

This project has two connected parts:

\begin{enumerate}
    \item \textbf{Detector training}
    \begin{itemize}
        \item Trains Random Forest, Gradient Boosting, XGBoost, and optionally a PyTorch MLP.
        \item Uses flow-level features from benign and malicious DoH traffic.
        \item Saves model artifacts, the fitted scaler, feature names, SHAP explanations, and evaluation reports.
    \end{itemize}

    \item \textbf{Adversarial evaluation}
    \begin{itemize}
        \item Generates synthetic DoH C2 traffic using different evasion strategies.
        \item Converts shaped traffic into packet/session-level data using Scapy.
        \item Extracts flow features using CICFlowMeter.
        \item Tests whether the trained detectors classify the generated traffic as malicious or benign.
    \end{itemize}
\end{enumerate}

This means users can now train and test everything from one branch.

\subsection{Project Structure}

\begin{verbatim}
doh-adversarial-detection/
├── detector.py                       # Main detector training and evaluation script
├── real_adversarial_pipeline.py       # White-box adversarial evaluation pipeline
├── blackbox_adversarial_pipeline.py   # Black-box adversarial evaluation pipeline
├── benign_target_sampling.py          # Benign target sampling and correlation validation
├── traffic_shaper.py                  # Core traffic shaping and evasion strategies
├── cira_cic_analyzer.py               # Extracts benign distributions from CIRA-CIC data
├── integrate_detectors.py             # Fast synthetic feature-level attack baseline
├── doh_c2_client.py                   # DoH C2 protocol logic
├── adversarial_loop.py                # Standalone attack-detect-adapt loop
├── main.py                            # Entry point for quick demos
├── requirements.txt                   # Python dependencies
├── data/
│   ├── l2-total-add.csv               # Layer 2: Benign vs. Malicious DoH
│   ├── l3-total-add.csv               # Layer 3: malicious tunnel tool labels
│   ├── l1-total-add.csv               # Layer 1: Non-DoH vs. DoH, not used by detector.py
│   └── README.txt                     # Dataset citation and license
└── results_full/                      # Output directory created after training/runs
    ├── rf.joblib                      # Trained Random Forest model
    ├── gb.joblib                      # Trained Gradient Boosting model
    ├── xgb.joblib                     # Trained XGBoost model
    ├── mlp.pt                         # Optional trained PyTorch MLP
    ├── scaler.joblib                  # Fitted StandardScaler
    ├── feature_names.json             # Ordered detector feature names
    ├── feature_limits.npy             # Feature percentile limits
    ├── pearson_matrix.npy             # Feature correlation matrix
    ├── results.txt                    # Detector evaluation report
    ├── per_tool_results.txt           # Per-tool detection breakdown
    ├── *_confusion_matrix.png         # Confusion matrices
    ├── *_roc_curve.png                # ROC curves
    ├── *_shap_importance.png/json     # SHAP feature importance
    ├── real_adversarial_results.csv   # White-box attack results
    └── blackbox_results.csv           # Black-box attack results
\end{verbatim}

\subsection{Environment Setup}

This project can be run locally or on UVA's HPC cluster.

\subsubsection{Option 1: Local Setup}

\begin{verbatim}
python -m venv netenv
source netenv/bin/activate
pip install -r requirements.txt
\end{verbatim}

On Windows PowerShell:

\begin{verbatim}
python -m venv netenv
.\netenv\Scripts\Activate.ps1
pip install -r requirements.txt
\end{verbatim}

\subsubsection{Option 2: UVA Rivanna/Afton Setup}

Get an interactive GPU node:

\begin{verbatim}
ijob -c 1 -A ml_at_uva -p gpu --gres=gpu:1 --time=0-01:00:00 --mem=64G
\end{verbatim}

Load modules:

\begin{verbatim}
ml cuda miniforge gcc
\end{verbatim}

Activate your environment:

\begin{verbatim}
source netenv/bin/activate
\end{verbatim}

If the environment does not already exist:

\begin{verbatim}
python -m venv netenv
source netenv/bin/activate
pip install -r requirements.txt
\end{verbatim}

\subsection{Dependencies}

The main dependencies are:

\begin{verbatim}
pip install numpy pandas scipy scikit-learn joblib xgboost scapy cicflowmeter matplotlib seaborn shap torch
\end{verbatim}

Notes:

\begin{itemize}
    \item \texttt{xgboost} is required to train and load \texttt{xgb.joblib}.
    \item \texttt{torch} is only needed if running the MLP. Use \texttt{--no\_nn} to skip neural network training.
    \item \texttt{pyarrow} or \texttt{fastparquet} is optional, but needed if using parquet reference files.
    \item \texttt{cicflowmeter} is required for the packet-to-flow feature extraction used by the adversarial pipelines.
\end{itemize}

\subsection{Fix CICFlowMeter v0.5.0 Bugs}

If CICFlowMeter fails during extraction, apply these two patches:

\begin{verbatim}
python3 -c "
path = '/opt/anaconda3/lib/python3.12/site-packages/cicflowmeter/sniffer.py'
txt = open(path).read()
fixed = txt.replace('if fields is not None:', 'if fields is not None and fields is not False and fields != True:')
open(path, 'w').write(fixed)
print('Fixed sniffer.py')
"
\end{verbatim}

\begin{verbatim}
python3 -c "
path = '/opt/anaconda3/lib/python3.12/site-packages/cicflowmeter/flow.py'
txt = open(path).read()
fixed = txt.replace(
    'data = {k: v for k, v in data.items() if k in include_fields}',
    'data = {k: v for k, v in data.items() if (not include_fields or k in include_fields)}'
)
open(path, 'w').write(fixed)
print('Fixed flow.py')
"
\end{verbatim}

Update the path if your Python environment is not located at \texttt{/opt/anaconda3/lib/python3.12/}.

\subsection{Dataset}

Download and unzip the Hokkaido University combined DoH dataset:

\begin{verbatim}
mkdir data
cd data
wget -O DoH-combined.zip "https://eprints.lib.hokudai.ac.jp/dspace/bitstream/2115/88092/1/CIRA-CIC-DoHBrw-2020-and-DoH-Tunnel-Traffic-HKD.zip"
unzip DoH-combined.zip
cd ..
\end{verbatim}

The main files used by this project are:

\begin{verbatim}
data/l2-total-add.csv
data/l3-total-add.csv
\end{verbatim}

\texttt{l2-total-add.csv} is used for binary benign vs. malicious DoH detection.

\texttt{l3-total-add.csv} is used for the per-tool malicious tunnel breakdown.

\subsection{How to Run}

\subsubsection{Step 1: Train the Detectors}

Fast run without the neural network:

\begin{verbatim}
python detector.py \
    --l2 data/l2-total-add.csv \
    --l3 data/l3-total-add.csv \
    --output ./results_full \
    --no_nn
\end{verbatim}

Full run with all four models:

\begin{verbatim}
python detector.py \
    --l2 data/l2-total-add.csv \
    --l3 data/l3-total-add.csv \
    --output ./results_full \
    --seed 42
\end{verbatim}

Development smoke test with a smaller sample:

\begin{verbatim}
python detector.py \
    --l2 data/l2-total-add.csv \
    --l3 data/l3-total-add.csv \
    --output ./results_test \
    --sample 5000 \
    --no_nn
\end{verbatim}

The detector step must be run before the adversarial pipelines because it creates the model files, scaler, and feature names used during attack evaluation.

\subsubsection{Step 2: Run the White-box Adversarial Pipeline}

The white-box pipeline assumes the attacker has access to the trained models, scaler, feature names, and model feedback. This is an upper-bound attack setting.

Default run:

\begin{verbatim}
python real_adversarial_pipeline.py --results ./results_full --flows 20
\end{verbatim}

Full run with correlated benign target sampling, correlation validation, marginal realism filtering, and feature validation:

\begin{verbatim}
python real_adversarial_pipeline.py \
    --results ./results_full \
    --flows 20 \
    --target-sampling correlated \
    --correlation-validation \
    --realism-filter \
    --validate
\end{verbatim}

\subsubsection{Step 3: Run the Black-box Adversarial Pipeline}

The black-box pipeline represents a more realistic attacker. The attacker can observe benign traffic and use public tools, but does not have access to the scaler or model internals.

\begin{verbatim}
python blackbox_adversarial_pipeline.py --results ./results_full --flows 20
\end{verbatim}

\subsubsection{Step 4: Validate Feature Matching}

To print a short CICFlowMeter feature comparison against benign CIRA traffic:

\begin{verbatim}
python real_adversarial_pipeline.py --results ./results_full --flows 3 --validate
\end{verbatim}

\subsubsection{Step 5: Quick Demo Without Trained Models}

\begin{verbatim}
python main.py --demo
python main.py --doh-demo
\end{verbatim}

\subsection{Detector CLI Flags}

\begin{tabular}{|l|l|p{8cm}|}
\hline
\textbf{Flag} & \textbf{Default} & \textbf{Description} \\
\hline
\texttt{--l2} & required & Path to \texttt{l2-total-add.csv}. \\
\hline
\texttt{--l3} & optional & Path to \texttt{l3-total-add.csv} for per-tool breakdown. \\
\hline
\texttt{--output} & \texttt{./results} & Directory to write model artifacts and results. \\
\hline
\texttt{--seed} & \texttt{42} & Random seed for reproducibility. \\
\hline
\texttt{--test\_size} & \texttt{0.2} & Fraction of data held out for testing. \\
\hline
\texttt{--no\_nn} & off & Skip MLP training. \\
\hline
\texttt{--sample} & off & Randomly subsample N rows from l2 for quick testing. \\
\hline
\end{tabular}

\subsection{White-box Pipeline Flags}

\begin{tabular}{|l|p{10cm}|}
\hline
\textbf{Flag} & \textbf{Description} \\
\hline
\texttt{--results} & Directory containing trained model artifacts from \texttt{detector.py}. \\
\hline
\texttt{--flows} & Number of generated flows per evasion strategy. \\
\hline
\texttt{--target-sampling interpolated} & kNN-style blend of nearby benign CIRA rows. \\
\hline
\texttt{--target-sampling correlated} & Multivariate Gaussian sampling with Ledoit-Wolf shrunk covariance. \\
\hline
\texttt{--target-sampling legacy} & Original random wire-stat sampling. \\
\hline
\texttt{--correlation-validation} & Compare real benign CIRA correlation structure against synthetic sampled rows. \\
\hline
\texttt{--correlation-val-n} & Number of synthetic rows used for correlation validation. \\
\hline
\texttt{--correlation-report} & Path to write the correlation validation report. \\
\hline
\texttt{--realism-filter} & Apply marginal realism checks after CICFlowMeter extraction. \\
\hline
\texttt{--realism-pass all|any} & Require all checks or any check to pass. \\
\hline
\texttt{--validate} & Print a CIC column vs. benign comparison for the first extracted flow. \\
\hline
\end{tabular}

\subsection{Architecture}

\subsubsection{Detector Training}

\begin{verbatim}
CIRA/HKD CSVs
      |
Feature cleaning + preprocessing
      |
Train/test split
      |
StandardScaler
      |
RF / GB / XGB / optional MLP
      |
Saved model artifacts in results_full/
\end{verbatim}

\subsubsection{Adversarial Evaluation}

\begin{verbatim}
C2 payload
      |
traffic_shaper.py
      |
Optional benign target sampling
      |  interpolated, correlated, or legacy
      |
Scapy TCP sessions
      |
Temporary PCAP
      |
CICFlowMeter feature extraction
      |
Feature mapping to detector columns
      |
Optional realism filter / validation
      |
White-box: scaler.transform() + model.predict_proba()
Black-box: raw unscaled features + model.predict()
      |
Detected or evaded
\end{verbatim}

\subsection{Evasion Strategies}

\begin{tabular}{|l|p{10cm}|}
\hline
\textbf{Strategy} & \textbf{Description} \\
\hline
\texttt{naive} & Fixed 5 second beacon with no shaping. \\
\hline
\texttt{timing\_only} & Randomized inter-arrival times matching browser-like traffic. \\
\hline
\texttt{size\_mimicry} & Timing plus packet sizes matching benign DoH distributions. \\
\hline
\texttt{cover\_traffic} & Adds decoy queries to real resolvers such as Google or Cloudflare. \\
\hline
\texttt{full\_mimicry} & Combines timing, size, cover traffic, and burst grouping. \\
\hline
\texttt{adaptive} & Full mimicry with a classifier feedback loop. \\
\hline
\end{tabular}

\subsection{Realistic Benign Targeting and Validation}

The white-box pipeline can sample benign CIRA-style targets and use those targets to drive Scapy session generation.

\subsubsection{Target Sampling Modes}

\begin{tabular}{|l|p{10cm}|}
\hline
\textbf{Mode} & \textbf{Behavior} \\
\hline
\texttt{interpolated} & Blends nearby benign rows with noise and quantile clipping. \\
\hline
\texttt{correlated} & Learns a covariance structure from benign rows and samples correlated synthetic rows. \\
\hline
\texttt{legacy} & Uses the older random wire-stat generation method. \\
\hline
\end{tabular}

\subsubsection{Validation Options}

\begin{tabular}{|l|p{10cm}|}
\hline
\textbf{Layer} & \textbf{What it checks} \\
\hline
Tabular correlation validation & Compares correlation matrices from real benign CIRA rows and synthetic sampled rows. \\
\hline
Marginal realism filter & Checks whether mapped CICFlowMeter features fall within benign quantile and z-score ranges. \\
\hline
\texttt{--validate} & Prints a short CICFlowMeter vs. benign feature comparison. \\
\hline
\end{tabular}

Because CICFlowMeter features and CIRA feature columns are not always identical, validation should be interpreted as a realism sanity check rather than exact row reproduction.

\subsection{Results Summary}

\subsubsection{Detector Results}

All four detector models achieve near-perfect offline performance on the original dataset.

\begin{tabular}{|l|l|l|l|}
\hline
\textbf{Model} & \textbf{Accuracy} & \textbf{F1} & \textbf{ROC-AUC} \\
\hline
Random Forest & $\sim$0.999 & $\sim$0.999 & $\sim$1.000 \\
\hline
Gradient Boosting & $\sim$0.999 & $\sim$0.999 & $\sim$1.000 \\
\hline
XGBoost & $\sim$0.999 & $\sim$0.999 & $\sim$1.000 \\
\hline
MLP & $\sim$0.999 & $\sim$0.999 & $\sim$1.000 \\
\hline
\end{tabular}

Per-tool detection is also near-perfect across the six malicious tunnel tools:

\begin{verbatim}
Tool            | RF_F1   | GB_F1   | XGB_F1  | MLP_F1
--------------------------------------------------------
dns2tcp         | 1.0000  | 1.0000  | 1.0000  | 0.9997
dnscat2         | 1.0000  | 1.0000  | 1.0000  | 0.9993
dnstt           | 1.0000  | 1.0000  | 1.0000  | 1.0000
iodine          | 1.0000  | 1.0000  | 0.9999  | 0.9987
tcp-over-dns    | 1.0000  | 1.0000  | 1.0000  | 1.0000
tuns            | 1.0000  | 1.0000  | 1.0000  | 1.0000
\end{verbatim}

\subsubsection{White-box Attack Results}

In the white-box setting, the attacker has access to the scaler, feature names, model internals, and model feedback.

\begin{tabular}{|l|l|l|l|}
\hline
\textbf{Strategy} & \textbf{RF} & \textbf{GB} & \textbf{XGB} \\
\hline
\texttt{naive} & 100\% & 100\% & 100\% \\
\hline
\texttt{timing\_only} & 100\% & 100\% & 100\% \\
\hline
\texttt{size\_mimicry} & 100\% & 100\% & 100\% \\
\hline
\texttt{cover\_traffic} & 100\% & 100\% & 100\% \\
\hline
\texttt{full\_mimicry} & 100\% & 100\% & 100\% \\
\hline
\texttt{adaptive} & 100\% & 100\% & 100\% \\
\hline
\end{tabular}

\subsubsection{Black-box Attack Results}

In the black-box setting, the attacker has observed benign traffic and public tools, but does not have the scaler or model internals.

\begin{tabular}{|l|l|l|l|}
\hline
\textbf{Strategy} & \textbf{RF} & \textbf{GB} & \textbf{XGB} \\
\hline
\texttt{naive} & 0\% & 0\% & 0\% \\
\hline
\texttt{timing\_only} & 0\% & 40\% & 0\% \\
\hline
\texttt{size\_mimicry} & 0\% & 40\% & 0\% \\
\hline
\texttt{cover\_traffic} & 0\% & 25\% & 0\% \\
\hline
\texttt{full\_mimicry} & 0\% & 50\% & 0\% \\
\hline
\texttt{adaptive} & 0\% & 50\% & 0\% \\
\hline
\end{tabular}

\subsection{Key Findings}

\begin{enumerate}
    \item Offline detector accuracy is not the same as deployment robustness.
    \item Attacker knowledge level is the dominant factor in evasion success.
    \item The fitted scaler is a critical defense artifact. With scaler access, evasion becomes much easier.
    \item Random Forest and XGBoost are more robust in the black-box setting.
    \item Gradient Boosting is fastest, but more vulnerable to black-box evasion.
    \item Synthetic feature edits should be validated because changing one feature group in isolation can create unrealistic traffic.
    \item Correlation-aware benign sampling and marginal realism checks help make adversarial evaluation more realistic.
\end{enumerate}

\subsection{Dataset Citation}

The Hokkaido University combined dataset merges:

\begin{enumerate}
    \item \textbf{CIRA-CIC-DoHBrw-2020}, which contains DoH browser traffic captures.
    \item \textbf{DoH-Tunnel-Traffic-HKD}, which contains six malicious DoH tunnel tool implementations.
\end{enumerate}

If you use this dataset, cite both papers:

\begin{quote}
M. MontazeriShatoori et al., ``Detection of DoH Tunnels using Time-series Classification of Encrypted Traffic,'' IEEE CyberSciTech, 2020. https://ieeexplore.ieee.org/document/9251211
\end{quote}

\begin{quote}
R. Mitsuhashi et al., ``Malicious DNS Tunnel Tool Recognition using Persistent DoH Traffic Analysis,'' IEEE TNSM, 2022. https://ieeexplore.ieee.org/document/9924534
\end{quote}

Dataset questions: mitsuhashi@os.ecc.u-tokyo.ac.jp