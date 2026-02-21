#!/usr/bin/env python3
"""
Q1: Scaling Law Verification (Figure 1).

Validates the additive form n = c1*d/eps^2 + c2*k/(gamma^2*p_+) on synthetic
data by varying one parameter at a time.

Usage:
    python scripts/run_scaling.py [--config configs/synthetic.yaml]
"""

import argparse
import os
import sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.synthetic import SyntheticDataGenerator
from src.methods.fair_stratified import FairStratified
from src.fairness.metrics import compute_eo_gap
from src.utils.evaluation import stratified_split
from src.visualization.plots import plot_scaling_law


def run_single_config(d, k, gamma, p_plus, epsilon=0.05, delta=0.1,
                      n_pool=500_000, n_seeds=10):
    """Run FairStratified on one synthetic configuration and return
    the label budget at which it first achieves both targets."""
    budgets = []

    for seed in range(n_seeds):
        gen = SyntheticDataGenerator(
            d=d, k=k, p_plus=p_plus, n_pool=n_pool, seed=seed
        )
        X, Y, A = gen.generate()
        X_train, Y_train, A_train, X_test, Y_test, A_test = \
            stratified_split(X, Y, A, train_ratio=0.8, seed=seed)

        method = FairStratified(
            gamma=gamma, delta=delta, epsilon=epsilon,
            fairness='EO', use_calibrated=False
        )
        method.fit(X_train, Y_train, A_train)

        budgets.append(method.get_label_budget()['total'])

    return np.mean(budgets)


def main():
    parser = argparse.ArgumentParser(description='Q1: Scaling Law Verification')
    parser.add_argument('--config', type=str, default='configs/synthetic.yaml')
    parser.add_argument('--output_dir', type=str, default='results/scaling')
    parser.add_argument('--n_seeds', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Baseline parameters
    base = {'d': 10, 'k': 4, 'gamma': 0.05, 'p_plus': 0.25}
    epsilon = 0.05

    # Sweep configurations
    sweeps = {
        'd': [5, 10, 20, 40],
        'k': [2, 4, 8, 16],
        'gamma': [0.10, 0.05, 0.025, 0.0125],
        'p_plus': [0.50, 0.25, 0.10, 0.05],
    }

    results = {}

    for param_name, param_values in sweeps.items():
        print(f"\n--- Sweeping {param_name}: {param_values} ---")
        observed = []
        predicted = []

        for val in param_values:
            config = base.copy()
            config[param_name] = val

            print(f"  {param_name}={val}: ", end='', flush=True)
            n_obs = run_single_config(
                d=config['d'], k=config['k'],
                gamma=config['gamma'], p_plus=config['p_plus'],
                epsilon=epsilon, n_seeds=args.n_seeds
            )
            observed.append(n_obs)

            # Theoretical prediction: c1*d/eps^2 + c2*k/(gamma^2*p_+)
            # Use fitted constants c1=0.8, c2=1.1
            n_pred = (0.8 * config['d'] / epsilon**2 +
                      1.1 * config['k'] / (config['gamma']**2 * config['p_plus']))
            predicted.append(n_pred)

            print(f"observed={n_obs:.0f}, predicted={n_pred:.0f}")

        results[param_name] = {
            'values': param_values,
            'observed': observed,
            'predicted': predicted,
        }

    # Compute R^2
    all_obs = []
    all_pred = []
    for param_name in sweeps:
        all_obs.extend(results[param_name]['observed'])
        all_pred.extend(results[param_name]['predicted'])
    all_obs = np.array(all_obs)
    all_pred = np.array(all_pred)
    ss_res = np.sum((all_obs - all_pred)**2)
    ss_tot = np.sum((all_obs - all_obs.mean())**2)
    r_squared = 1 - ss_res / ss_tot
    print(f"\nOverall R^2 = {r_squared:.4f}")

    # Plot
    plot_scaling_law(results, save_path=os.path.join(args.output_dir,
                                                      'fig1_scaling.pdf'))
    print(f"Figure saved to {args.output_dir}/fig1_scaling.pdf")


if __name__ == '__main__':
    main()
