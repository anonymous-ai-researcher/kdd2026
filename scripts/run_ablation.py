#!/usr/bin/env python3
"""
Ablation Studies (Figures 5, 6).

(a) Phase 1 allocation ratio on Folktables
(b) Group imbalance on synthetic data
(c) Class imbalance (1/p+ scaling) on synthetic data

Usage:
    python scripts/run_ablation.py [--config configs/benchmark.yaml]
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.synthetic import SyntheticDataGenerator
from src.methods.fair_stratified import FairStratified
from src.methods.constrained_erm_dp import ConstrainedERM_DP
from src.utils.evaluation import stratified_split
from src.fairness.metrics import compute_eo_gap
from src.visualization.plots import plot_class_imbalance


def run_class_imbalance_sweep(output_dir, n_seeds=10):
    """Figure 6: Class imbalance - EO cost scales as 1/p+."""
    print("\n=== Class Imbalance Sweep ===")

    d, k, gamma = 10, 9, 0.05
    p_plus_values = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

    eo_costs = []
    dp_costs = []

    for p_plus in p_plus_values:
        print(f"  p+ = {p_plus}: ", end='', flush=True)

        eo_budgets = []
        dp_budgets = []

        for seed in range(n_seeds):
            gen = SyntheticDataGenerator(
                d=d, k=k, p_plus=p_plus, n_pool=500_000, seed=seed
            )
            X, Y, A = gen.generate()
            X_tr, Y_tr, A_tr, X_te, Y_te, A_te = \
                stratified_split(X, Y, A, seed=seed)

            # EO
            fs = FairStratified(
                gamma=gamma, delta=0.1, epsilon=0.05,
                fairness='EO', use_calibrated=False
            )
            fs.fit(X_tr, Y_tr, A_tr)
            eo_budgets.append(fs.get_label_budget()['total'])

            # DP
            dp = ConstrainedERM_DP(gamma=gamma, delta=0.1, epsilon=0.05)
            dp.fit(X_tr, Y_tr, A_tr)
            dp_budgets.append(dp.get_label_budget()['total'])

        eo_mean = np.mean(eo_budgets)
        dp_mean = np.mean(dp_budgets)
        eo_costs.append(eo_mean)
        dp_costs.append(dp_mean)
        print(f"EO={eo_mean:.0f}, DP={dp_mean:.0f}")

    # Fit log-log slope for EO
    log_p = np.log(p_plus_values)
    log_eo = np.log(eo_costs)
    slope, intercept = np.polyfit(log_p, log_eo, 1)
    ss_res = np.sum((log_eo - (slope * log_p + intercept))**2)
    ss_tot = np.sum((log_eo - log_eo.mean())**2)
    r_squared = 1 - ss_res / ss_tot

    print(f"\nFitted slope: {slope:.2f} (expected: -1.0)")
    print(f"R^2: {r_squared:.4f}")

    if p_plus_values[0] > 0 and p_plus_values[-1] > 0:
        ratio = eo_costs[0] / dp_costs[0]
        print(f"At p+={p_plus_values[0]}: EO/DP ratio = {ratio:.1f}x")

    results = {
        'p_plus_values': p_plus_values,
        'eo_costs': eo_costs,
        'dp_costs': dp_costs,
        'fitted_slope': slope,
        'r_squared': r_squared,
    }
    plot_class_imbalance(
        results,
        save_path=os.path.join(output_dir, 'fig6_imbalance.pdf')
    )
    print(f"Figure saved to {output_dir}/fig6_imbalance.pdf")


def run_phase1_allocation(output_dir, n_seeds=5):
    """Figure 5a: Phase 1 allocation ratio on synthetic data."""
    print("\n=== Phase 1 Allocation Ratio ===")

    ratios = [0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 2.0]
    d, k, gamma, p_plus = 10, 4, 0.10, 0.25

    for r in ratios:
        accs = []
        eo_gaps = []

        for seed in range(n_seeds):
            gen = SyntheticDataGenerator(
                d=d, k=k, p_plus=p_plus, n_pool=100_000, seed=seed
            )
            X, Y, A = gen.generate()
            X_tr, Y_tr, A_tr, X_te, Y_te, A_te = \
                stratified_split(X, Y, A, seed=seed)

            fs = FairStratified(
                gamma=gamma * r,  # Scale the stopping rule
                delta=0.1, epsilon=0.05, fairness='EO',
                use_calibrated=True
            )
            fs.fit(X_tr, Y_tr, A_tr)
            y_pred = fs.predict(X_te)

            accs.append((y_pred == Y_te).mean())
            eo_gaps.append(compute_eo_gap(y_pred, Y_te, A_te))

        print(f"  r={r:.2f}: acc={np.mean(accs)*100:.1f}%, "
              f"EO={np.mean(eo_gaps):.3f}")


def main():
    parser = argparse.ArgumentParser(description='Ablation Studies')
    parser.add_argument('--config', type=str, default='configs/benchmark.yaml')
    parser.add_argument('--output_dir', type=str, default='results/ablation')
    parser.add_argument('--n_seeds', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Run ablations
    run_phase1_allocation(args.output_dir, n_seeds=min(args.n_seeds, 5))
    run_class_imbalance_sweep(args.output_dir, n_seeds=args.n_seeds)


if __name__ == '__main__':
    main()
