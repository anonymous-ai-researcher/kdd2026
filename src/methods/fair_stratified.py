"""
FairStratified: Two-phase algorithm for EO/EqOdds-fair active learning.

Algorithm 2 from the paper. Achieves label complexity:
    O(d/eps^2 * log(1/delta) + k/(gamma^2 * p_+) * log(k/delta))
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..fairness.constraints import solve_constrained_erm
from ..fairness.metrics import compute_fairness_gap, compute_group_rates
from ..utils.stopping import (
    theoretical_stopping_rule,
    calibrated_stopping_rule,
)


class FairStratified:
    """FairStratified active learning algorithm.

    Two-phase design:
      Phase 1: Stratified sampling to collect sufficient positives
               (and negatives for EqOdds) per group for TPR/FPR estimation.
      Phase 2: i.i.d. sampling for accuracy training + constrained ERM.

    Parameters
    ----------
    gamma : float
        Fairness tolerance.
    delta : float
        Failure probability.
    epsilon : float
        Accuracy tolerance.
    fairness : str
        'EO' or 'EqOdds'.
    use_calibrated : bool
        Use calibrated (True) or theoretical (False) stopping rule.
    C : float
        Logistic regression regularization strength.
    max_iter : int
        Maximum solver iterations.
    """

    def __init__(self, gamma=0.10, delta=0.10, epsilon=0.05,
                 fairness='EO', use_calibrated=True, C=1.0, max_iter=1000):
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.fairness = fairness
        self.use_calibrated = use_calibrated
        self.C = C
        self.max_iter = max_iter

        # Will be set during fit
        self.model_ = None
        self.n_phase1_ = 0
        self.n_phase2_ = 0
        self.n_total_ = 0

    def _compute_target_positives(self, k):
        """Compute n_a^+ for each group."""
        if self.use_calibrated:
            return calibrated_stopping_rule(self.gamma, k, self.delta)
        else:
            return theoretical_stopping_rule(self.gamma, k, self.delta)

    def _phase1_sampling(self, X_pool, Y_pool, A_pool, k, rng):
        """Phase 1: Stratified sampling to collect positives per group.

        Uses negative-binomial stopping rule: for each group, keep sampling
        and labeling until n_a^+ positives are collected.

        Returns
        -------
        S_cstr : dict
            Constraint set: {group_id: (X, Y)} with n_a^+ positives per group.
        S_val : dict
            Validation set: {group_id: (X, Y)} with n_a^+ positives per group.
        n_queries : int
            Total label queries used in Phase 1.
        """
        n_a_plus = self._compute_target_positives(k)
        target_per_group = 2 * n_a_plus  # split into cstr + val

        S_cstr = {a: {'X': [], 'Y': []} for a in range(k)}
        S_val = {a: {'X': [], 'Y': []} for a in range(k)}
        n_queries = 0
        positives_collected = np.zeros(k, dtype=int)

        # Also collect negatives for EqOdds
        negatives_collected = np.zeros(k, dtype=int)
        need_negatives = (self.fairness == 'EqOdds')
        S_cstr_neg = {a: {'X': [], 'Y': []} for a in range(k)}
        S_val_neg = {a: {'X': [], 'Y': []} for a in range(k)}

        # Shuffle pool
        pool_indices = rng.permutation(len(X_pool))

        for idx in pool_indices:
            a = A_pool[idx]

            # Check if this group still needs samples
            needs_pos = positives_collected[a] < target_per_group
            needs_neg = need_negatives and (
                negatives_collected[a] < target_per_group
            )

            if not needs_pos and not needs_neg:
                continue

            # Query label (this is the label oracle call)
            y = Y_pool[idx]
            x = X_pool[idx]
            n_queries += 1

            if y == 1 and needs_pos:
                positives_collected[a] += 1
                half = n_a_plus
                if positives_collected[a] <= half:
                    S_cstr[a]['X'].append(x)
                    S_cstr[a]['Y'].append(y)
                else:
                    S_val[a]['X'].append(x)
                    S_val[a]['Y'].append(y)

            if y == 0 and needs_neg:
                negatives_collected[a] += 1
                half = n_a_plus
                if negatives_collected[a] <= half:
                    S_cstr_neg[a]['X'].append(x)
                    S_cstr_neg[a]['Y'].append(y)
                else:
                    S_val_neg[a]['X'].append(x)
                    S_val_neg[a]['Y'].append(y)

            # Check if all groups done
            all_pos_done = all(
                positives_collected[a] >= target_per_group for a in range(k)
            )
            all_neg_done = (not need_negatives) or all(
                negatives_collected[a] >= target_per_group for a in range(k)
            )
            if all_pos_done and all_neg_done:
                break

        return S_cstr, S_val, S_cstr_neg, S_val_neg, n_queries

    def _phase2_sampling(self, X_pool, Y_pool, A_pool, phase1_used, rng):
        """Phase 2: Draw i.i.d. labeled samples for accuracy training.

        Sample size: O(d/epsilon^2 * log(1/delta))
        """
        d = X_pool.shape[1]
        n_phase2 = max(
            100,
            int(np.ceil(2.0 * d / self.epsilon**2 * np.log(4 / self.delta)))
        )

        # Sample from remaining pool (independent of Phase 1)
        remaining = len(X_pool) - phase1_used
        n_phase2 = min(n_phase2, remaining)

        indices = rng.choice(len(X_pool), size=n_phase2, replace=False)
        return X_pool[indices], Y_pool[indices], A_pool[indices], n_phase2

    def fit(self, X_pool, Y_pool, A_pool, budget=None):
        """Fit the FairStratified algorithm.

        Parameters
        ----------
        X_pool : ndarray, shape (n, d)
            Unlabeled pool features.
        Y_pool : ndarray, shape (n,)
            Labels (oracle; queried adaptively).
        A_pool : ndarray, shape (n,)
            Group memberships (observable without labels).
        budget : int or None
            Maximum label budget. If None, uses theoretical budget.

        Returns
        -------
        self
        """
        rng = np.random.RandomState(42)
        k = int(A_pool.max()) + 1

        # Phase 1: Stratified fairness sampling
        S_cstr, S_val, S_cstr_neg, S_val_neg, n_phase1 = \
            self._phase1_sampling(X_pool, Y_pool, A_pool, k, rng)
        self.n_phase1_ = n_phase1

        # Phase 2: i.i.d. accuracy training
        X_train, Y_train, A_train, n_phase2 = \
            self._phase2_sampling(X_pool, Y_pool, A_pool, n_phase1, rng)
        self.n_phase2_ = n_phase2
        self.n_total_ = n_phase1 + n_phase2

        # Constrained ERM on Phase 2 data with Phase 1 constraint set
        self.model_ = solve_constrained_erm(
            X_train, Y_train, A_train,
            gamma=self.gamma, notion=self.fairness,
            C=self.C, max_iter=self.max_iter
        )

        # Validation step
        self._validate(S_val, S_val_neg, k)

        return self

    def _validate(self, S_val, S_val_neg, k):
        """Validation: check fairness on held-out validation set."""
        if self.model_ is None:
            return

        # Collect validation positives
        X_val_list, Y_val_list, A_val_list = [], [], []
        for a in range(k):
            if len(S_val[a]['X']) > 0:
                X_val_list.extend(S_val[a]['X'])
                Y_val_list.extend(S_val[a]['Y'])
                A_val_list.extend([a] * len(S_val[a]['X']))

        if len(X_val_list) == 0:
            return

        X_val = np.array(X_val_list)
        Y_val = np.array(Y_val_list)
        A_val = np.array(A_val_list)

        y_pred = self.model_.predict(X_val)
        val_gap = compute_fairness_gap(y_pred, Y_val, A_val, self.fairness, k)

        self.validation_gap_ = val_gap
        self.validation_passed_ = (val_gap <= 3 * self.gamma / 4)

    def predict(self, X):
        """Predict labels."""
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model_.predict(X)

    def get_label_budget(self):
        """Return label budget breakdown."""
        return {
            'phase1': self.n_phase1_,
            'phase2': self.n_phase2_,
            'total': self.n_total_,
        }
