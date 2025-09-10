# Paper Title:  Discrete Gaussian Integer Aggregation and Trust-Budget Gating for Federated Learning in IoT-Enabled CPS

## Overview

PCU-FL is an implementation of privacy-preserving federated learning that combines **FP8 quantization** (E4M3/E5M2) with **discrete Gaussian noise** to achieve optimal trade-offs between privacy, communication efficiency, and model utility in cross-device federated learning scenarios.

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
```

## Project Structure

```
PCU-FL/
├── main.py                
├── pcu_algo.py            # Main PCU-FL algorithm
├── server_ops.py          # Server-side operations
├── client_ops.py          # Client-side operations
├── config.py              # Configuration parameters
├── data_loader.py         # Dataset utilities
├── fl_model.py            # Model architectures
├── he_backend.py          # Homomorphic encryption 
├── zk_proofs.py           # Zero-knowledge proofs 
├── noise_commit.py        # Discrete Gaussian noise commitments 
├── proof_utils.py         # Proof utilities 
├── metrics_logger.py      # Metrics and logging
                           # Dataset (user provided)
```

## Reproducibility

**Dataset:** Edge-IIoT anomaly detection dataset with 2,219 samples and 61 features. Uses the published split without resampling or rebalancing.

**Model:** Three-layer MLP with hidden dimensions [128, 64, 32], ReLU activations, and Xavier initialization.

**Federated Setup:** 
- N=50 simulated clients
- Up to M_t ≤ N clients participate each round
- Local optimization: SGD (learning rate 0.01), batch size 32, one local epoch per round

**Round Counts for Operating Points:**
- Non-private: 50 rounds
- ε=50: 30 rounds  
- ε=10: 40 rounds
- ε=1: 100 rounds
- FP8 baseline: 75 rounds

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
