"""
FAL (Fair Active Learning) baseline.

Combines uncertainty sampling with a group-balancing term:
    alpha_FAL(x, a) = H(p_hat(y|x)) + lambda / (n_a + 1)

Following Anahideh et al. (2022).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from .passive_fair import PassiveFair


class FAL:
    """Fair Active Learning with uncertainty + group-balancing.

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
        Queries per batch.
    warm_start : int
        Initial random samples.
    lam : float
        Group-balancing coefficient (lambda).
    candidate_subsample : int
        Max candidates per batch.
    """

    def __init__(self, gamma=0.10, notion='EO', C=1.0, max_iter=1000,
                 batch_size=10, warm_start=50, lam=0.5,
                 candidate_subsample=10000):
        self.gamma = gamma
        self.notion = notion
        self.C = C
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.warm_start = warm_start
        self.lam = lam
        self.candidate_subsample = candidate_subsample

    def fit(self, X_pool, Y_pool, A_pool, budget, seed=0):
        """Run FAL active learning.

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
        model : PassiveFair instance
        labeled_indices : list of int
        """
        rng = np.random.RandomState(seed)
        n = len(X_pool)
        k = int(A_pool.max()) + 1

        labeled_mask = np.zeros(n, dtype=bool)
        labeled_indices = []
        group_counts = np.zeros(k, dtype=int)

        # Warm start
        warm_idx = rng.choice(n, size=min(self.warm_start, budget),
                              replace=False)
        labeled_mask[warm_idx] = True
        labeled_indices.extend(warm_idx.tolist())
        for idx in warm_idx:
            group_counts[A_pool[idx]] += 1

        while len(labeled_indices) < budget:
            X_L = X_pool[labeled_mask]
            Y_L = Y_pool[labeled_mask]

            model = LogisticRegression(
                C=self.C, solver='lbfgs', max_iter=self.max_iter, tol=1e-4
            )
            if len(np.unique(Y_L)) < 2:
                unlabeled = np.where(~labeled_mask)[0]
                if len(unlabeled) == 0:
                    break
                extra = rng.choice(unlabeled, size=min(10, len(unlabeled)),
                                   replace=False)
                labeled_mask[extra] = True
                labeled_indices.extend(extra.tolist())
                for idx in extra:
                    group_counts[A_pool[idx]] += 1
                continue

            model.fit(X_L, Y_L)

            unlabeled = np.where(~labeled_mask)[0]
            if len(unlabeled) == 0:
                break

            n_candidates = min(self.candidate_subsample, len(unlabeled))
            candidate_idx = rng.choice(unlabeled, size=n_candidates,
                                       replace=False)

            # Compute acquisition scores
            probs = model.predict_proba(X_pool[candidate_idx])[:, 1]
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            entropy = -(probs * np.log(probs) +
                       (1 - probs) * np.log(1 - probs))

            # Group-balancing term
            groups = A_pool[candidate_idx]
            balance_term = np.array([
                self.lam / (group_counts[a] + 1) for a in groups
            ])

            scores = entropy + balance_term

            # Select top batch
            n_select = min(self.batch_size, budget - len(labeled_indices))
            top_idx = np.argsort(-scores)[:n_select]
            selected = candidate_idx[top_idx]

            labeled_mask[selected] = True
            labeled_indices.extend(selected.tolist())
            for idx in selected:
                group_counts[A_pool[idx]] += 1

        # Final model with post-hoc adjustment
        X_L = X_pool[labeled_mask]
        Y_L = Y_pool[labeled_mask]
        A_L = A_pool[labeled_mask]

        final_model = PassiveFair(
            gamma=self.gamma, notion=self.notion,
            C=self.C, max_iter=self.max_iter
        )
        final_model.fit(X_L, Y_L, A_L, seed=seed)

        return final_model, labeled_indices
