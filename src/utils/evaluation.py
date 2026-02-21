"""
Evaluation utilities for fair active learning experiments.
"""

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from ..fairness.metrics import compute_fairness_gap


def stratified_split(X, Y, A, train_ratio=0.8, seed=0):
    """Perform stratified train/test split preserving (A, Y) proportions.

    Parameters
    ----------
    X : ndarray, shape (n, d)
    Y : ndarray, shape (n,)
    A : ndarray, shape (n,)
    train_ratio : float
    seed : int

    Returns
    -------
    X_train, Y_train, A_train, X_test, Y_test, A_test
    """
    # Create composite stratification key
    strat_key = A * 2 + Y  # unique per (group, label) pair
    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=train_ratio, random_state=seed
    )
    train_idx, test_idx = next(splitter.split(X, strat_key))

    return (
        X[train_idx], Y[train_idx], A[train_idx],
        X[test_idx], Y[test_idx], A[test_idx]
    )


def evaluate_model(model, X_test, Y_test, A_test, notion='EO', k=None):
    """Evaluate a model on test data.

    Returns
    -------
    dict with keys: 'accuracy', 'error', 'fairness_gap', 'group_rates'
    """
    y_pred = model.predict(X_test)
    acc = (y_pred == Y_test).mean()
    fairness_gap = compute_fairness_gap(y_pred, Y_test, A_test, notion, k)

    return {
        'accuracy': float(acc),
        'error': float(1 - acc),
        'fairness_gap': float(fairness_gap),
    }


def compute_label_savings(n_method, n_passive):
    """Compute label savings relative to Passive-Fair.

    Saved = 1 - n_method / n_passive
    """
    if n_passive <= 0 or n_method <= 0:
        return 0.0
    return max(0.0, 1.0 - n_method / n_passive)


def find_convergence_budget(accuracies, fairness_gaps, budgets,
                            tau_acc, gamma):
    """Find the smallest budget at which method reaches target.

    Parameters
    ----------
    accuracies : list of float
        Accuracy at each budget step.
    fairness_gaps : list of float
        Fairness gap at each budget step.
    budgets : list of int
        Label budget at each step.
    tau_acc : float
        Target accuracy threshold.
    gamma : float
        Fairness tolerance.

    Returns
    -------
    n_converge : int or None
        Smallest budget achieving both targets, or None.
    """
    for acc, fg, b in zip(accuracies, fairness_gaps, budgets):
        if acc >= tau_acc and fg <= gamma:
            return b
    return None


def run_seeds(run_fn, seeds, **kwargs):
    """Run an experiment across multiple seeds and aggregate results.

    Parameters
    ----------
    run_fn : callable
        Function that takes seed as first argument and returns a dict.
    seeds : list of int
    **kwargs : additional arguments passed to run_fn.

    Returns
    -------
    dict with mean and stderr for each metric.
    """
    all_results = []
    for seed in seeds:
        result = run_fn(seed, **kwargs)
        all_results.append(result)

    # Aggregate
    keys = all_results[0].keys()
    aggregated = {}
    for key in keys:
        values = [r[key] for r in all_results if r[key] is not None]
        if len(values) > 0 and isinstance(values[0], (int, float)):
            arr = np.array(values, dtype=float)
            aggregated[f'{key}_mean'] = float(arr.mean())
            aggregated[f'{key}_stderr'] = float(arr.std() / np.sqrt(len(arr)))
        else:
            aggregated[key] = values

    return aggregated
