#!/usr/bin/env python3
"""
Q3: Benchmark Comparison (Figure 3, Table 2).

Compares FairStratified against baselines on Folktables, COMPAS, and Adult
under EO constraint with gamma=0.10.

Usage:
    python scripts/run_benchmarks.py [--config configs/benchmark.yaml]
"""

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.folktables import load_folktables
from src.data.compas import load_compas
from src.data.adult import load_adult
from src.methods.fair_stratified import FairStratified
from src.methods.passive_fair import PassiveFair
from src.methods.active_fair import ActiveFair
from src.methods.fal import FAL
from src.fairness.metrics import compute_eo_gap, compute_dp_gap
from src.fairness.constraints import solve_unconstrained_erm
from src.utils.evaluation import stratified_split, evaluate_model


DATASET_CONFIGS = {
    'Folktables': {'budget': 45000, 'tau_acc': 0.784},
    'COMPAS': {'budget': 5000, 'tau_acc': 0.637},
    'Adult': {'budget': 23000, 'tau_acc': 0.809},
}


def load_dataset(name):
    """Load and return dataset."""
    if name == 'Folktables':
        return load_folktables()
    elif name == 'COMPAS':
        return load_compas()
    elif name == 'Adult':
        return load_adult()
    else:
        raise ValueError(f"Unknown dataset: {name}")


def run_single_seed(dataset_name, seed, gamma=0.10, delta=0.10):
    """Run all methods on one dataset with one seed."""
    X, Y, A, group_names = load_dataset(dataset_name)
    k = int(A.max()) + 1
    config = DATASET_CONFIGS[dataset_name]
    budget = config['budget']

    X_train, Y_train, A_train, X_test, Y_test, A_test = \
        stratified_split(X, Y, A, train_ratio=0.8, seed=seed)

    results = {}

    # 1. Unconstrained Active (accuracy ceiling)
    model_unc = solve_unconstrained_erm(X_train[:budget], Y_train[:budget])
    y_pred = model_unc.predict(X_test)
    results['Unconstrained'] = {
        'accuracy': (y_pred == Y_test).mean(),
        'eo_gap': compute_eo_gap(y_pred, Y_test, A_test, k),
    }

    # 2. Passive-Fair
    passive = PassiveFair(gamma=gamma, notion='EO')
    n_passive = min(budget, len(X_train))
    passive.fit(X_train[:n_passive], Y_train[:n_passive],
                A_train[:n_passive], seed=seed)
    y_pred = passive.predict(X_test, A_test)
    results['Passive-Fair'] = {
        'accuracy': (y_pred == Y_test).mean(),
        'eo_gap': compute_eo_gap(y_pred, Y_test, A_test, k),
    }

    # 3. Active-Fair
    active = ActiveFair(gamma=gamma, notion='EO')
    model_af, _ = active.fit(X_train, Y_train, A_train, budget, seed=seed)
    y_pred = model_af.predict(X_test, A_test)
    results['Active-Fair'] = {
        'accuracy': (y_pred == Y_test).mean(),
        'eo_gap': compute_eo_gap(y_pred, Y_test, A_test, k),
    }

    # 4. FAL
    fal = FAL(gamma=gamma, notion='EO')
    model_fal, _ = fal.fit(X_train, Y_train, A_train, budget, seed=seed)
    y_pred = model_fal.predict(X_test, A_test)
    results['FAL'] = {
        'accuracy': (y_pred == Y_test).mean(),
        'eo_gap': compute_eo_gap(y_pred, Y_test, A_test, k),
    }

    # 5. FairStratified (ours)
    fs = FairStratified(gamma=gamma, delta=delta, fairness='EO',
                        use_calibrated=True)
    fs.fit(X_train, Y_train, A_train, budget=budget)
    y_pred = fs.predict(X_test)
    results['FairStratified'] = {
        'accuracy': (y_pred == Y_test).mean(),
        'eo_gap': compute_eo_gap(y_pred, Y_test, A_test, k),
        'budget_breakdown': fs.get_label_budget(),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='Q3: Benchmark Comparison')
    parser.add_argument('--config', type=str, default='configs/benchmark.yaml')
    parser.add_argument('--output_dir', type=str, default='results/benchmarks')
    parser.add_argument('--datasets', nargs='+',
                        default=['Folktables', 'COMPAS', 'Adult'])
    parser.add_argument('--seeds', nargs='+', type=int,
                        default=list(range(10)))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for dname in args.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dname}")
        print(f"{'='*60}")

        all_results = {method: {'accuracy': [], 'eo_gap': []}
                       for method in ['Unconstrained', 'Passive-Fair',
                                      'Active-Fair', 'FAL', 'FairStratified']}

        for seed in args.seeds:
            print(f"\n  Seed {seed}...", end=' ', flush=True)
            t0 = time.perf_counter()

            try:
                seed_results = run_single_seed(dname, seed)
                for method, metrics in seed_results.items():
                    all_results[method]['accuracy'].append(metrics['accuracy'])
                    all_results[method]['eo_gap'].append(metrics['eo_gap'])
                elapsed = time.perf_counter() - t0
                print(f"done ({elapsed:.1f}s)")
            except Exception as e:
                print(f"FAILED: {e}")
                continue

        # Print summary table
        print(f"\n--- {dname} Results (mean ± stderr, {len(args.seeds)} seeds) ---")
        print(f"{'Method':<20} {'Acc(%)':<15} {'EO Gap':<12}")
        print("-" * 50)
        for method in ['Unconstrained', 'Passive-Fair', 'Active-Fair',
                       'FAL', 'FairStratified']:
            accs = all_results[method]['accuracy']
            gaps = all_results[method]['eo_gap']
            if len(accs) > 0:
                acc_mean = np.mean(accs) * 100
                acc_se = np.std(accs) / np.sqrt(len(accs)) * 100
                gap_mean = np.mean(gaps)
                gap_se = np.std(gaps) / np.sqrt(len(gaps))
                marker = '†' if gap_mean > 0.10 else ' '
                print(f"{method:<20} {acc_mean:.1f}±{acc_se:.1f}      "
                      f".{int(gap_mean*1000):03d}{marker}")


if __name__ == '__main__':
    main()
