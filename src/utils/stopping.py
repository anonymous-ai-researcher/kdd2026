"""
Stopping rules for Phase 1 of FairStratified.

Theoretical rule: n_a^+ = ceil((32/gamma^2) * ln(8k/delta))
Calibrated rule:  n_a^+ = ceil((2/gamma^2) * ln(2k/delta))
"""

import numpy as np


def theoretical_stopping_rule(gamma, k, delta):
    """Compute per-group positive target using theoretical constants.

    n_a^+ = ceil((32/gamma^2) * ln(8k/delta))

    The constant 32 arises from:
      (i)   factor of 2 in Hoeffding's two-sided bound
      (ii)  factor of 1/8^2 = 1/64 from targeting precision gamma/8
      (iii) ln(8k/delta) from union bound over k groups with failure budget delta/4

    Parameters
    ----------
    gamma : float
        Fairness tolerance.
    k : int
        Number of protected groups.
    delta : float
        Failure probability.

    Returns
    -------
    n_a_plus : int
        Required number of positive samples per group.
    """
    n_a_plus = int(np.ceil((32.0 / gamma**2) * np.log(8 * k / delta)))
    return n_a_plus


def calibrated_stopping_rule(gamma, k, delta):
    """Compute per-group positive target using calibrated constants.

    n_a^+ = ceil((2/gamma^2) * ln(2k/delta))

    Corresponds to Hoeffding bound targeting precision gamma directly:
        Pr[|TPR_hat_a - TPR_a| > gamma] <= 2*exp(-2*n_a^+*gamma^2) = delta/k

    Parameters
    ----------
    gamma : float
        Fairness tolerance.
    k : int
        Number of protected groups.
    delta : float
        Failure probability.

    Returns
    -------
    n_a_plus : int
        Required number of positive samples per group.
    """
    n_a_plus = int(np.ceil((2.0 / gamma**2) * np.log(2 * k / delta)))
    return n_a_plus


def compute_phase1_budget(gamma, k, delta, p_a, use_calibrated=True,
                          fairness='EO'):
    """Compute expected Phase 1 label budget.

    Parameters
    ----------
    gamma : float
        Fairness tolerance.
    k : int
        Number of groups.
    delta : float
        Failure probability.
    p_a : ndarray or float
        Per-group positive rate(s). If scalar, assumes equal across groups.
    use_calibrated : bool
        Use calibrated (True) or theoretical (False) stopping rule.
    fairness : str
        'EO' or 'EqOdds'.

    Returns
    -------
    total_budget : float
        Expected total Phase 1 label queries.
    per_group_positives : int
        Target positives per group (n_a^+).
    """
    if use_calibrated:
        n_a_plus = calibrated_stopping_rule(gamma, k, delta)
    else:
        n_a_plus = theoretical_stopping_rule(gamma, k, delta)

    if np.isscalar(p_a):
        p_a = np.full(k, p_a)

    # Expected labels per group: 2*n_a^+/p_a (factor 2 for sample splitting)
    if fairness == 'EO':
        total_budget = sum(2 * n_a_plus / p_a[a] for a in range(k))
    elif fairness == 'EqOdds':
        total_budget = sum(
            2 * n_a_plus / p_a[a] + 2 * n_a_plus / (1 - p_a[a])
            for a in range(k)
        )
    else:
        raise ValueError(f"Phase 1 not needed for {fairness}")

    return float(total_budget), n_a_plus
