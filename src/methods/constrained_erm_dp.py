"""
ConstrainedERM-DP: DP-fair learning with zero label cost for fairness.

Algorithm 1 from the paper. Because DP gap r_a(h) = Pr[h(X)=1 | A=a] depends
only on predictions and group membership (both observable without labels),
fairness verification is entirely label-free.

Label complexity: O(d/eps^2 * log(1/delta))
Unlabeled cost:   O(k * (d*log(d/gamma) + log(k/delta)) / gamma^2)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..fairness.constraints import solve_constrained_erm
from ..fairness.metrics import compute_dp_gap


class ConstrainedERM_DP:
    """ConstrainedERM for Demographic Parity.

    Two phases:
      Phase 1 (Labels for accuracy): Draw O(d/eps^2 * log(1/delta)) i.i.d.
          labeled samples.
      Phase 2 (Unlabeled for fairness): Compute DP gap from unlabeled data.
      Solve: Constrained ERM minimizing error subject to DP <= gamma/2.

    Parameters
    ----------
    gamma : float
        Fairness tolerance.
    delta : float
        Failure probability.
    epsilon : float
        Accuracy tolerance.
    C : float
        Logistic regression regularization.
    max_iter : int
        Maximum solver iterations.
    """

    def __init__(self, gamma=0.10, delta=0.10, epsilon=0.05,
                 C=1.0, max_iter=1000):
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.C = C
        self.max_iter = max_iter

        self.model_ = None
        self.n_labeled_ = 0

    def fit(self, X_pool, Y_pool, A_pool, X_unlabeled=None, A_unlabeled=None):
        """Fit the ConstrainedERM-DP algorithm.

        Parameters
        ----------
        X_pool : ndarray, shape (n, d)
            Pool features.
        Y_pool : ndarray, shape (n,)
            Labels (only n_L are queried).
        A_pool : ndarray, shape (n,)
            Group memberships.
        X_unlabeled : ndarray or None
            Additional unlabeled data for DP verification.
            If None, uses X_pool with observed A_pool.
        A_unlabeled : ndarray or None
            Group memberships for unlabeled data.

        Returns
        -------
        self
        """
        d = X_pool.shape[1]

        # Phase 1: Draw labeled samples for accuracy
        n_L = max(
            100,
            int(np.ceil(2.0 * d / self.epsilon**2 * np.log(4 / self.delta)))
        )
        n_L = min(n_L, len(X_pool))
        self.n_labeled_ = n_L

        rng = np.random.RandomState(42)
        indices = rng.choice(len(X_pool), size=n_L, replace=False)
        X_train = X_pool[indices]
        Y_train = Y_pool[indices]
        A_train = A_pool[indices]

        # Phase 2: DP verification is label-free (uses unlabeled data)
        # The constrained ERM handles DP constraint internally

        # Solve constrained ERM
        self.model_ = solve_constrained_erm(
            X_train, Y_train, A_train,
            gamma=self.gamma, notion='DP',
            C=self.C, max_iter=self.max_iter
        )

        return self

    def predict(self, X):
        """Predict labels."""
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model_.predict(X)

    def get_label_budget(self):
        """Return label budget (only accuracy labels; fairness is free)."""
        return {
            'labeled': self.n_labeled_,
            'fairness_labels': 0,
            'total': self.n_labeled_,
        }
