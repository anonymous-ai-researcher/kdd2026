"""
Fairness metrics: DP gap, EO gap, EqOdds gap.

All metrics return the maximum absolute difference across group pairs.
"""

import numpy as np


def compute_group_rates(y_pred, y_true, A, k=None):
    """Compute per-group positive prediction rates, TPRs, and FPRs.

    Parameters
    ----------
    y_pred : ndarray, shape (n,)
        Predicted labels {0, 1}.
    y_true : ndarray, shape (n,)
        True labels {0, 1}.
    A : ndarray, shape (n,)
        Group membership {0, ..., k-1}.
    k : int or None
        Number of groups (inferred from A if None).

    Returns
    -------
    dict with keys:
        'ppr': ndarray, per-group positive prediction rates
        'tpr': ndarray, per-group true positive rates
        'fpr': ndarray, per-group false positive rates
        'group_sizes': ndarray, per-group sample counts
    """
    if k is None:
        k = int(A.max()) + 1

    ppr = np.full(k, np.nan)
    tpr = np.full(k, np.nan)
    fpr = np.full(k, np.nan)
    group_sizes = np.zeros(k, dtype=int)

    for a in range(k):
        mask_a = (A == a)
        group_sizes[a] = mask_a.sum()
        if group_sizes[a] == 0:
            continue

        y_pred_a = y_pred[mask_a]
        y_true_a = y_true[mask_a]

        # Positive prediction rate
        ppr[a] = y_pred_a.mean()

        # True positive rate
        pos_mask = (y_true_a == 1)
        if pos_mask.sum() > 0:
            tpr[a] = y_pred_a[pos_mask].mean()

        # False positive rate
        neg_mask = (y_true_a == 0)
        if neg_mask.sum() > 0:
            fpr[a] = y_pred_a[neg_mask].mean()

    return {'ppr': ppr, 'tpr': tpr, 'fpr': fpr, 'group_sizes': group_sizes}


def compute_dp_gap(y_pred, A, k=None):
    """Compute Demographic Parity gap: max |r_a - r_b| over all group pairs.

    DP can be computed without labels (y_true not needed).
    """
    if k is None:
        k = int(A.max()) + 1

    rates = []
    for a in range(k):
        mask = (A == a)
        if mask.sum() > 0:
            rates.append(y_pred[mask].mean())

    if len(rates) < 2:
        return 0.0
    return float(max(rates) - min(rates))


def compute_eo_gap(y_pred, y_true, A, k=None):
    """Compute Equal Opportunity gap: max |TPR_a - TPR_b| over all group pairs."""
    rates = compute_group_rates(y_pred, y_true, A, k)
    tprs = rates['tpr']
    valid = ~np.isnan(tprs)
    if valid.sum() < 2:
        return 0.0
    return float(np.nanmax(tprs) - np.nanmin(tprs))


def compute_eqodds_gap(y_pred, y_true, A, k=None):
    """Compute Equalized Odds gap: max(EO gap, max |FPR_a - FPR_b|)."""
    rates = compute_group_rates(y_pred, y_true, A, k)

    tprs = rates['tpr']
    fprs = rates['fpr']

    eo_gap = 0.0
    valid_tpr = ~np.isnan(tprs)
    if valid_tpr.sum() >= 2:
        eo_gap = float(np.nanmax(tprs) - np.nanmin(tprs))

    fpr_gap = 0.0
    valid_fpr = ~np.isnan(fprs)
    if valid_fpr.sum() >= 2:
        fpr_gap = float(np.nanmax(fprs) - np.nanmin(fprs))

    return max(eo_gap, fpr_gap)


def compute_fairness_gap(y_pred, y_true, A, notion='EO', k=None):
    """Compute fairness gap for the specified notion.

    Parameters
    ----------
    notion : str
        One of 'DP', 'EO', 'EqOdds'.
    """
    if notion == 'DP':
        return compute_dp_gap(y_pred, A, k)
    elif notion == 'EO':
        return compute_eo_gap(y_pred, y_true, A, k)
    elif notion == 'EqOdds':
        return compute_eqodds_gap(y_pred, y_true, A, k)
    else:
        raise ValueError(f"Unknown fairness notion: {notion}")


def compute_empirical_tpr(y_pred, y_true, A, k=None):
    """Compute per-group empirical TPR estimates."""
    rates = compute_group_rates(y_pred, y_true, A, k)
    return rates['tpr']


def compute_empirical_fpr(y_pred, y_true, A, k=None):
    """Compute per-group empirical FPR estimates."""
    rates = compute_group_rates(y_pred, y_true, A, k)
    return rates['fpr']
