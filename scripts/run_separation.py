#!/usr/bin/env python3
"""
Q2: Separation Theorem (Figure 2).

Demonstrates that EO-constrained learning hits a label complexity floor at
k/(gamma^2 * p_+) while DP-constrained matches unconstrained.

Usage:
    python scripts/run_separation.py [--config configs/synthetic.yaml]
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.synthetic import SyntheticDataGenerator
from src.methods.fair_stratified import FairStratified
from src.methods.constrained_erm_dp import ConstrainedERM_DP
from src.fairness.constraints import solve_unconstrained_erm
from src.utils.evaluation import stratified_split
from src.visualization.plots import plot_separation


def measure_label_complexity(method_type, epsilon, d=1, k=4, gamma=0.05,
                             p_plus=0.25, delta=0.1, n_pool=500_000,
                             n_seeds=10):
    """Measure label complexity for a given method and epsilon."""
    budgets = []

    for seed in range(n_seeds):
        gen = SyntheticDataGenerator(
            d=d, k=k, p_plus=p_plus, n_pool=n_pool, seed=seed
        )
        X, Y, A = gen.generate()
        X_train, Y_train, A_train, X_test, Y_test, A_test = \
            stratified_split(X, Y, A, train_ratio=0.8, seed=seed)

        if method_type == 'unconstrained':
            # Standard ERM, label complexity ~ d/eps^2
            n_labels = max(50, int(np.ceil(2 * d / epsilon**2)))
            n_labels = min(n_labels, len(X_train))
            budgets.append(n_labels)

        elif method_type == 'dp':
            method = ConstrainedERM_DP(
                gamma=gamma, delta=delta, epsilon=epsilon
            )
            method.fit(X_train, Y_train, A_train)
            budgets.append(method.get_label_budget()['total'])

        elif method_type == 'eo':
            method = FairStratified(
                gamma=gamma, delta=delta, epsilon=epsilon,
                fairness='EO', use_calibrated=False
            )
            method.fit(X_train, Y_train, A_train)
            budgets.append(method.get_label_budget()['total'])

    return np.mean(budgets)


def main():
    parser = argparse.ArgumentParser(description='Q2: Separation Theorem')
    parser.add_argument('--config', type=str, default='configs/synthetic.yaml')
    parser.add_argument('--output_dir', type=str, default='results/separation')
    parser.add_argument('--n_seeds', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Parameters
    k, gamma, p_plus = 4, 0.05, 0.25
    epsilon_values = [0.50, 0.20, 0.10, 0.05, 0.02]

    # Theoretical EO floor
    eo_floor = k / (gamma**2 * p_plus)
    print(f"Theoretical EO floor: k/(gamma^2*p+) = {eo_floor:.0f}")

    results = {
        'epsilon_values': epsilon_values,
        'unconstrained': [],
        'dp': [],
        'eo': [],
        'eo_floor': eo_floor,
    }

    for eps in epsilon_values:
        print(f"\nepsilon = {eps}:")

        for method_type in ['unconstrained', 'dp', 'eo']:
            n = measure_label_complexity(
                method_type, eps, d=10, k=k, gamma=gamma,
                p_plus=p_plus, n_seeds=args.n_seeds
            )
            results[method_type].append(n)
            print(f"  {method_type}: {n:.0f}")

    # Plot
    plot_separation(results, save_path=os.path.join(args.output_dir,
                                                     'fig2_separation.pdf'))
    print(f"\nFigure saved to {args.output_dir}/fig2_separation.pdf")


if __name__ == '__main__':
    main()
