# The Price of Fairness in Active Learning: Fundamental Limits and Optimal Label Acquisition

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository provides the official implementation for the paper:

> **The Price of Fairness in Active Learning: Fundamental Limits and Optimal Label Acquisition**
>
> *KDD 2026*

## Overview

We establish the first information-theoretic lower bounds for fair active learning, proving a structural dichotomy across fairness notions:

| Fairness Notion | Additional Label Cost | Key Insight |
|---|---|---|
| **Demographic Parity (DP)** | 0 (label-free) | Verified from unlabeled data alone |
| **Equal Opportunity (EO)** | Ω(k / (γ² p₊)) | Conditioning on Y=1 creates labeling bottleneck |
| **Equalized Odds (EqOdds)** | Ω(k / (γ² p₊ p₋)) | Additionally requires FPR verification |

We present matching algorithms — **ConstrainedERM-DP** for DP and **FairStratified** for EO/EqOdds — that are tight up to logarithmic factors, and validate the theory on both synthetic and real-world benchmarks.

### Key Results

- **36–39% label savings** over fairness-agnostic baselines on Folktables, COMPAS, and Adult
- **Scaling law verification**: R² = 0.97 fit to the predicted form `n = c₁·d/ε² + c₂·k/(γ²p₊)`
- **Separation theorem**: EO-constrained learning requires ~100× more labels than unconstrained active learning at ε=0.02
- At p₊ = 0.01 (e.g., fraud detection), EO requires **~90× more labels** than DP

## Installation

```bash
# Clone the repository
git clone https://github.com/anonymous-ai-researcher/kdd2026.git
cd kdd2026

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- NumPy 1.25.2
- scikit-learn 1.3.2
- CVXPY 1.4.1
- fairlearn 0.9.0
- folktables 0.0.12
- pandas 2.1.1
- matplotlib 3.8.0

## Project Structure

```
├── README.md
├── requirements.txt
├── LICENSE
├── configs/
│   ├── default.yaml          # Default experiment configuration
│   ├── synthetic.yaml        # Synthetic data experiments (Q1, Q2, Q4)
│   └── benchmark.yaml        # Real-data benchmark experiments (Q3)
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── synthetic.py      # Synthetic Gaussian mixture data generator
│   │   ├── folktables.py     # Folktables (ACSIncome) loader
│   │   ├── compas.py         # COMPAS recidivism loader
│   │   └── adult.py          # UCI Adult Income loader
│   ├── methods/
│   │   ├── __init__.py
│   │   ├── fair_stratified.py      # FairStratified (Algorithm 2, EO/EqOdds)
│   │   ├── constrained_erm_dp.py   # ConstrainedERM-DP (Algorithm 1)
│   │   ├── passive_fair.py         # Passive-Fair baseline
│   │   ├── active_fair.py          # Active-Fair baseline (uncertainty + post-hoc)
│   │   └── fal.py                  # FAL baseline (uncertainty + group-balancing)
│   ├── fairness/
│   │   ├── __init__.py
│   │   ├── metrics.py        # Fairness metrics (DP, EO, EqOdds gaps)
│   │   └── constraints.py    # Constrained ERM via exponentiated gradient
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── evaluation.py     # Evaluation utilities
│   │   └── stopping.py       # Stopping rules (theoretical & calibrated)
│   └── visualization/
│       ├── __init__.py
│       └── plots.py          # Plotting utilities for all figures
├── scripts/
│   ├── run_scaling.py        # Q1: Scaling law verification (Fig. 1)
│   ├── run_separation.py     # Q2: Separation theorem (Fig. 2)
│   ├── run_benchmarks.py     # Q3: Benchmark comparison (Fig. 3, Table 2)
│   ├── run_hierarchy.py      # Q4: Fairness hierarchy (Fig. 4)
│   ├── run_ablation.py       # Ablation studies (Fig. 5, 6)
│   └── run_all.py            # Run all experiments
└── tests/
    ├── test_metrics.py       # Unit tests for fairness metrics
    ├── test_data.py          # Unit tests for data loaders
    └── test_methods.py       # Unit tests for methods
```

## Quick Start

### Run all experiments

```bash
python scripts/run_all.py
```

### Run individual experiments

```bash
# Q1: Scaling law verification (Figure 1)
python scripts/run_scaling.py --config configs/synthetic.yaml

# Q2: Separation theorem (Figure 2)
python scripts/run_separation.py --config configs/synthetic.yaml

# Q3: Benchmark comparison (Figure 3, Table 2)
python scripts/run_benchmarks.py --config configs/benchmark.yaml

# Q4: Fairness hierarchy (Figure 4)
python scripts/run_hierarchy.py --config configs/synthetic.yaml

# Ablation studies (Figures 5, 6)
python scripts/run_ablation.py --config configs/benchmark.yaml
```

### Custom experiment

```python
from src.data.synthetic import SyntheticDataGenerator
from src.methods.fair_stratified import FairStratified
from src.fairness.metrics import compute_eo_gap

# Generate synthetic data
gen = SyntheticDataGenerator(d=10, k=4, p_plus=0.25, n_pool=500_000, seed=42)
X, Y, A = gen.generate()

# Run FairStratified
method = FairStratified(gamma=0.05, delta=0.1, epsilon=0.05, fairness='EO')
h = method.fit(X, Y, A)

# Evaluate
eo_gap = compute_eo_gap(h, X, Y, A)
print(f"EO gap: {eo_gap:.4f}")
```

## Algorithms

### Algorithm 1: ConstrainedERM-DP

For Demographic Parity, fairness verification is label-free:

1. **Phase 1** (Labels for accuracy): Draw `O(d/ε² · log(1/δ))` i.i.d. labeled samples
2. **Phase 2** (Unlabeled for fairness): Compute DP gap from unlabeled data
3. **Solve**: Constrained ERM minimizing error subject to DP ≤ γ/2

### Algorithm 2: FairStratified (EO/EqOdds)

Two-phase design with stratified sampling:

1. **Phase 1** (Stratified fairness samples): For each group, sample until `n_a⁺` positives collected; split into constraint/validation halves
2. **Phase 2** (i.i.d. accuracy training): Draw `O(d/ε² · log(1/δ))` i.i.d. labeled samples
3. **Constrained ERM**: Minimize error subject to fairness constraint on constraint set
4. **Validation**: Check fairness on held-out validation set

## Datasets

| Dataset | n | d | k | p₊ | π_min | Source |
|---|---|---|---|---|---|---|
| Folktables | 195K | 10 | 9 | 0.25 | 0.4% | ACSIncome 2018 |
| COMPAS | 6K | 7 | 6 | 0.45 | 0.8% | ProPublica |
| Adult | 45K | 14 | 5 | 0.24 | 0.8% | UCI |
| Synthetic | 500K | varies | varies | varies | — | Gaussian mixtures |

Real datasets are automatically downloaded on first use.

## Reproducibility

All results are reproducible with seeds `{0, 1, ..., 9}`. Random state controls:

1. 80/20 stratified train/test split
2. Active query sequence
3. Warm-start sample for active baselines
4. Constrained ERM initialization

To reproduce all paper results:

```bash
python scripts/run_all.py --seeds 0 1 2 3 4 5 6 7 8 9
```

## Configuration

Experiments are configured via YAML files in `configs/`. Key parameters:

```yaml
# Fairness parameters
gamma: 0.10          # Fairness tolerance
delta: 0.10          # Failure probability
epsilon: 0.05        # Accuracy tolerance
fairness: "EO"       # DP, EO, or EqOdds

# Stopping rule
use_calibrated: true        # true for real data, false for synthetic
theoretical_constant: 32    # 32/γ² (theoretical) or 2/γ² (calibrated)

# Model
classifier: "logistic"
C: 1.0               # Regularization
solver: "lbfgs"
max_iter: 1000

# Active learning
batch_size: 10
warm_start: 50
```

## Citation

```bibtex
@inproceedings{anonymous2026price,
  title={The Price of Fairness in Active Learning: Fundamental Limits and Optimal Label Acquisition},
  author={Anonymous},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2026}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
