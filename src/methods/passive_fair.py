"""
Passive-Fair baseline.

Draws labeled samples uniformly at random (standard passive learning).
Fairness is enforced via post-hoc ROC threshold adjustment (Hardt et al. 2016):
per-group thresholds are chosen to equalize TPR (EO) or TPR+FPR (EqOdds).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..fairness.metrics import compute_group_rates


class PassiveFair:
    """Passive learning with post-hoc fairness adjustment.

    Parameters
    ----------
    gamma : float
        Fairness tolerance.
    notion : str
        'EO' or 'EqOdds'.
    C : float
        Logistic regression regularization.
    max_iter : int
        Maximum solver iterations.
    calibration_ratio : float
        Fraction of labeled budget reserved for threshold calibration.
    n_threshold_candidates : int
        Number of threshold candidates per group.
    """

    def __init__(self, gamma=0.10, notion='EO', C=1.0, max_iter=1000,
                 calibration_ratio=0.2, n_threshold_candidates=200):
        self.gamma = gamma
        self.notion = notion
        self.C = C
        self.max_iter = max_iter
        self.calibration_ratio = calibration_ratio
        self.n_threshold_candidates = n_threshold_candidates

        self.model_ = None
        self.thresholds_ = None

    def fit(self, X_labeled, Y_labeled, A_labeled, seed=0):
        """Fit passive learner with post-hoc threshold adjustment.

        Parameters
        ----------
        X_labeled : ndarray, shape (n, d)
        Y_labeled : ndarray, shape (n,)
        A_labeled : ndarray, shape (n,)
        seed : int

        Returns
        -------
        self
        """
        rng = np.random.RandomState(seed)
        n = len(X_labeled)
        k = int(A_labeled.max()) + 1

        # Split into train and calibration
        n_cal = max(50, int(n * self.calibration_ratio))
        n_train = n - n_cal
        perm = rng.permutation(n)
        train_idx = perm[:n_train]
        cal_idx = perm[n_train:]

        X_train, Y_train = X_labeled[train_idx], Y_labeled[train_idx]
        X_cal, Y_cal, A_cal = (
            X_labeled[cal_idx], Y_labeled[cal_idx], A_labeled[cal_idx]
        )

        # Train base classifier
        self.model_ = LogisticRegression(
            C=self.C, solver='lbfgs', max_iter=self.max_iter, tol=1e-4
        )
        self.model_.fit(X_train, Y_train)

        # Get probability predictions on calibration set
        probs = self.model_.predict_proba(X_cal)[:, 1]

        # Find per-group thresholds to equalize TPR
        self.thresholds_ = self._find_thresholds(probs, Y_cal, A_cal, k)

        return self

    def _find_thresholds(self, probs, Y, A, k):
        """Find per-group thresholds to minimize fairness gap."""
        candidates = np.linspace(0.01, 0.99, self.n_threshold_candidates)
        best_thresholds = np.full(k, 0.5)
        best_gap = float('inf')

        # Grid search over threshold per group
        # For efficiency, fix other groups and optimize one at a time
        for iteration in range(3):  # few rounds of coordinate descent
            for a in range(k):
                mask_a = (A == a)
                if mask_a.sum() == 0:
                    continue

                best_t = best_thresholds[a]
                for t in candidates:
                    thresholds_try = best_thresholds.copy()
                    thresholds_try[a] = t

                    # Compute predictions with these thresholds
                    y_pred = np.zeros(len(probs), dtype=int)
                    for g in range(k):
                        mask_g = (A == g)
                        y_pred[mask_g] = (
                            probs[mask_g] >= thresholds_try[g]
                        ).astype(int)

                    # Compute fairness gap
                    rates = compute_group_rates(y_pred, Y, A, k)
                    if self.notion == 'EO':
                        tprs = rates['tpr']
                        valid = ~np.isnan(tprs)
                        if valid.sum() < 2:
                            continue
                        gap = np.nanmax(tprs) - np.nanmin(tprs)
                    elif self.notion == 'EqOdds':
                        tprs = rates['tpr']
                        fprs = rates['fpr']
                        tpr_gap = np.nanmax(tprs) - np.nanmin(tprs)
                        fpr_gap = np.nanmax(fprs) - np.nanmin(fprs)
                        gap = max(tpr_gap, fpr_gap)
                    else:
                        gap = 0.0

                    if gap < best_gap:
                        best_gap = gap
                        best_t = t

                best_thresholds[a] = best_t

        return best_thresholds

    def predict(self, X, A=None):
        """Predict with per-group thresholds."""
        if self.model_ is None:
            raise RuntimeError("Model not fitted.")

        probs = self.model_.predict_proba(X)[:, 1]

        if A is not None and self.thresholds_ is not None:
            y_pred = np.zeros(len(X), dtype=int)
            k = len(self.thresholds_)
            for a in range(k):
                mask = (A == a)
                y_pred[mask] = (probs[mask] >= self.thresholds_[a]).astype(int)
            return y_pred
        else:
            return (probs >= 0.5).astype(int)
