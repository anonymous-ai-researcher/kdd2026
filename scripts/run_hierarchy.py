#!/usr/bin/env python3
"""
Q4: Fairness Hierarchy Verification (Figure 4).

Verifies Corollary 1 (DP < EO < EqOdds label costs) on synthetic data
with Folktables-matched parameters.

Usage:
    python scripts/run_hierarchy.py [--config configs/synthetic.yaml]
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
from src.visualization.plots import plot_hierarchy


def measure_cost(notion, k=9, p_plus=0.25, gamma=0.05, delta=0.1,
                 d=10, n_pool=500_000, n_seeds=10):
    """Measure label cost for a given fairness notion."""
    costs = []

    for seed in range(n_seeds):
        gen = SyntheticDataGenerator(
            d=d, k=k, p_plus=p_plus, n_pool=n_pool, seed=seed
        )
        X, Y, A = gen.generate()
        X_train, Y_train, A_train, X_test, Y_test, A_test = \
            stratified_split(X, Y, A, train_ratio=0.8, seed=seed)

        if notion == 'DP':
            method = ConstrainedERM_DP(
                gamma=gamma, delta=delta, epsilon=0.05
            )
            method.fit(X_train, Y_train, A_train)
            costs.append(method.get_label_budget()['total'])

        elif notion in ('EO', 'EqOdds'):
            method = FairStratified(
                gamma=gamma, delta=delta, epsilon=0.05,
                fairness=notion, use_calibrated=False
            )
            method.fit(X_train, Y_train, A_train)
            costs.append(method.get_label_budget()['total'])

    return np.mean(costs), np.std(costs) / np.sqrt(len(costs))


def main():
    parser = argparse.ArgumentParser(description='Q4: Fairness Hierarchy')
    parser.add_argument('--config', type=str, default='configs/synthetic.yaml')
    parser.add_argument('--output_dir', type=str, default='results/hierarchy')
    parser.add_argument('--n_seeds', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Folktables-matched parameters
    k, p_plus, gamma = 9, 0.25, 0.05

    print(f"Parameters: k={k}, p+={p_plus}, gamma={gamma}")
    print(f"\nTheoretical predictions:")
    print(f"  DP extra cost:     0 (label-free)")
    print(f"  EO extra cost:     k/(gamma^2*p+) = {k/(gamma**2*p_plus):.0f}")
    print(f"  EqOdds extra cost: k/(gamma^2*p+*p-) = "
          f"{k/(gamma**2*p_plus*(1-p_plus)):.0f}")

    notions = ['DP', 'EO', 'EqOdds']
    costs = []
    stderrs = []

    for notion in notions:
        print(f"\nMeasuring {notion}...", end=' ', flush=True)
        mean_cost, se = measure_cost(
            notion, k=k, p_plus=p_plus, gamma=gamma,
            n_seeds=args.n_seeds
        )
        costs.append(mean_cost)
        stderrs.append(se)
        print(f"{mean_cost:.0f} ± {se:.0f}")

    # Compute ratios
    baseline = costs[0]  # DP cost (accuracy only)
    ratios = [c / baseline for c in costs]

    print(f"\nLabel cost ratios (relative to DP baseline):")
    for notion, ratio in zip(notions, ratios):
        print(f"  {notion}: {ratio:.1f}x")

    # Expected ratios with p+ = 0.25:
    # EO/DP ≈ 1 + k/(gamma^2*p+) / (d/eps^2) ≈ 4.6
    # EqOdds/EO ≈ 1 + p+/p- = 1.33
    print(f"\nExpected EqOdds/EO ratio: 1 + p+/p- = {1 + p_plus/(1-p_plus):.2f}")
    if len(costs) >= 3 and costs[1] > 0:
        print(f"Observed EqOdds/EO ratio: {costs[2]/costs[1]:.2f}")

    # Plot
    results = {
        'notions': notions,
        'costs': costs,
        'ratios': ratios,
    }
    plot_hierarchy(results, save_path=os.path.join(args.output_dir,
                                                    'fig4_hierarchy.pdf'))
    print(f"\nFigure saved to {args.output_dir}/fig4_hierarchy.pdf")


if __name__ == '__main__':
    main()
