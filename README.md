# Paper Title:  Discrete Gaussian Integer Aggregation and Trust-Budget Gating for Federated Learning in IoT-Enabled CPS

[![Paper](https://img.shields.io/badge/Paper-IEEE-blue)](https://ieeexplore.ieee.org/document/11337755)

## Overview

PCU-FL is a federated learning pipeline for IoT-enabled cyber-physical systems (CPS) that combines discrete Gaussian integer aggregation with trust budget gating under encoded domain differential privacy (DP). 

The pipeline operates as follows: each client clips its update; the server encodes updates on a grid 1/α, forms an exact integer weighted sum, injects discrete Gaussian noise into the integer domain, and performs a single final decode via (γW_int)^(-1) to release a DP average. Noise is calibrated to the sensitivity of the normalized (averaged) function, explicitly accounting for clipping and rounding.

Key features include:
- Longitudinal trust budget gating mechanism with policy bound update tokens to control per-client participation and cumulative privacy spending across rounds
- FP8 quantization (E4M3/E5M2) for wire compression, evaluated without homomorphic encryption
- Discrete Gaussian DP in the encoded integer domain with trust-budget gating
- Cryptographic components (Ring-LWE encryption and zero-knowledge input validation) specified as blueprints for future work.

Read the full paper: [IEEE Xplore](https://ieeexplore.ieee.org/document/11337755)

## Environment Setup

```bash
# Clone repository
git clone https://github.com/cybw90/PCU-FL.git
cd PCU-FL

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Ubuntu:
source venv/bin/activate

# On macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Dependencies

All dependencies are listed in `requirements.txt`:
- Python 3.9+
- PyTorch 1.13+
- NumPy 1.24+
- Additional packages as specified in requirements.txt

## Core Experiments

Replace `XXX` with the desired number of training rounds.

```bash
# Baseline (No DP)
python main.py --epsilon 1000 --rounds XXX

# Baseline with FP8 (No DP)
python main.py --epsilon 1000 --rounds XXX --fp8 E4M3

# Light privacy
python main.py --epsilon 50 --rounds XXX

# Moderate privacy
python main.py --epsilon 10 --rounds XXX

# Strong privacy
python main.py --epsilon 1 --rounds XXX

# Mia_evaluation
python main.py --epsilon 10000 --rounds XXX --evaluate-mia
```

## Reproducibility

**Dataset:** Edge-IIoT anomaly detection dataset with 2,219 samples and 61 features. Uses the published split without resampling or rebalancing.

**Model:** Three-layer MLP with hidden dimensions [128, 64, 32], ReLU activations, and Xavier initialization.

**Federated Setup:** 
- N=50 simulated clients
- Up to M_t ≤ N clients participate each round
- Local optimization: SGD (learning rate 0.01), batch size 32, one local epoch per round

**Privacy Calibration:**
- Client-level add/remove adjacency
- Discrete Gaussian noise sampled in integer domain
- Truncated at 8σ_int
- Privacy composition with δ=10^-5

**Environment:**
- Fixed seeds: data split (42), client sampling (42+round), Gaussian noise (hash of round and parameter ID)
- Python 3.9, PyTorch 1.13, NumPy 1.24
- CPU-only execution supported at reported scales

**Validation:** Empirical noise norms agree with theoretical values within ±5% across all privacy levels

## Output Files

All outputs are saved to the `logs/` directory:
- Training metrics: `logs/pcufl_run_[timestamp].json`
- Visualization plots: `logs/pcufl_metrics_calibrated.png`

## License

This project is licensed under the MIT License - see the [LICENSE](License) file for details.

## Citation

If you use this work, please cite:

S. H. Shah and M. Borowczak, "Discrete Gaussian Integer Aggregation and Trust-Budget Gating for Federated Learning in IoT-Enabled CPS," *2025 Cyber Awareness and Research Symposium (CARS)*, Grand Forks, ND, USA, 2025, pp. 1-6, doi: [10.1109/CARS67163.2025.11337755](https://doi.org/10.1109/CARS67163.2025.11337755)

<details>
<summary>BibTeX</summary>

@INPROCEEDINGS{11337755,
  author={Shah, Sajjad H. and Borowczak, Mike},
  booktitle={2025 Cyber Awareness and Research Symposium (CARS)}, 
  title={Discrete Gaussian Integer Aggregation and Trust-Budget Gating for Federated Learning in IoT-Enabled CPS}, 
  year={2025},
  volume={},
  number={},
  pages={1-6},
  keywords={Privacy;Differential privacy;Sensitivity;Quantization (signal);Accuracy;Federated learning;Gaussian noise;Robustness;Encryption;Servers;Federated Learning;Secure Aggregation;Differential Privacy;Zero-Knowledge Proof;FP8 Quantization;Byzantine Robustness},
  doi={10.1109/CARS67163.2025.11337755}}

</details>

