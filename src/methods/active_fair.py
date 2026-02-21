"""
Active-Fair baseline.

Uncertainty sampling (max entropy) with post-hoc ROC threshold adjustment.
Selects instances with highest predictive entropy, retrains after each batch.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from .passive_fair import PassiveFair


class ActiveFair:
    """Active learning with uncertainty sampling + post-hoc fairness.

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
    batch_size : int
        Number of instances to query per batch.
    warm_start : int
        Number of initial random samples.
    candidate_subsample : int
        Max candidates to evaluate per batch (for efficiency).
    """

    def __init__(self, gamma=0.10, notion='EO', C=1.0, max_iter=1000,
                 batch_size=10, warm_start=50, candidate_subsample=10000):
        self.gamma = gamma
        self.notion = notion
        self.C = C
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.warm_start = warm_start
        self.candidate_subsample = candidate_subsample

    def fit(self, X_pool, Y_pool, A_pool, budget, seed=0):
        """Run active learning with uncertainty sampling.

        Parameters
        ----------
        X_pool : ndarray, shape (n, d)
        Y_pool : ndarray, shape (n,)
        A_pool : ndarray, shape (n,)
        budget : int
            Total label budget.
        seed : int

        Returns
        -------
        model : PassiveFair instance (with post-hoc threshold adjustment)
        history : dict with learning curves
        """
        rng = np.random.RandomState(seed)
        n = len(X_pool)

        # Track labeled/unlabeled
        labeled_mask = np.zeros(n, dtype=bool)
        labeled_indices = []

        # Warm start: random samples
        warm_idx = rng.choice(n, size=min(self.warm_start, budget), replace=False)
        labeled_mask[warm_idx] = True
        labeled_indices.extend(warm_idx.tolist())

        history = {'budget': [], 'accuracy': [], 'fairness_gap': []}

        # Active learning loop
        while len(labeled_indices) < budget:
            X_L = X_pool[labeled_mask]
            Y_L = Y_pool[labeled_mask]
            A_L = A_pool[labeled_mask]

            # Train current model
            model = LogisticRegression(
                C=self.C, solver='lbfgs', max_iter=self.max_iter, tol=1e-4
            )
            if len(np.unique(Y_L)) < 2:
                # Need both classes; add more random samples
                unlabeled = np.where(~labeled_mask)[0]
                if len(unlabeled) == 0:
                    break
                extra = rng.choice(unlabeled, size=min(10, len(unlabeled)),
                                   replace=False)
                labeled_mask[extra] = True
                labeled_indices.extend(extra.tolist())
                continue

            model.fit(X_L, Y_L)

            # Select candidates from unlabeled pool
            unlabeled = np.where(~labeled_mask)[0]
            if len(unlabeled) == 0:
                break

            n_candidates = min(self.candidate_subsample, len(unlabeled))
            candidate_idx = rng.choice(unlabeled, size=n_candidates,
                                       replace=False)

            # Compute entropy for candidates
            probs = model.predict_proba(X_pool[candidate_idx])[:, 1]
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            entropy = -(probs * np.log(probs) +
                       (1 - probs) * np.log(1 - probs))

            # Select top batch_size by entropy
            n_select = min(self.batch_size, budget - len(labeled_indices))
            top_idx = np.argsort(-entropy)[:n_select]
            selected = candidate_idx[top_idx]

            labeled_mask[selected] = True
            labeled_indices.extend(selected.tolist())

        # Final model with post-hoc threshold adjustment
        X_L = X_pool[labeled_mask]
        Y_L = Y_pool[labeled_mask]
        A_L = A_pool[labeled_mask]

        final_model = PassiveFair(
            gamma=self.gamma, notion=self.notion,
            C=self.C, max_iter=self.max_iter
        )
        final_model.fit(X_L, Y_L, A_L, seed=seed)

        return final_model, labeled_indices
