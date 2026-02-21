"""
Synthetic data generator using Gaussian mixture model.

Generates data with configurable d, k, gamma, p_plus and a linear decision
boundary ensuring realizability. Group means are equally spaced on the unit sphere.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit


class SyntheticDataGenerator:
    """Generate synthetic data from a Gaussian mixture model.

    Parameters
    ----------
    d : int
        Feature dimensionality.
    k : int
        Number of protected groups.
    p_plus : float
        Target marginal positive rate Pr[Y=1].
    n_pool : int
        Pool size (number of instances to generate).
    sigma : float
        Standard deviation for Gaussian clusters.
    sigma_eta : float
        Label noise std (0 = realizable setting).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, d=10, k=4, p_plus=0.25, n_pool=500_000,
                 sigma=0.5, sigma_eta=0.0, seed=None):
        self.d = d
        self.k = k
        self.p_plus = p_plus
        self.n_pool = n_pool
        self.sigma = sigma
        self.sigma_eta = sigma_eta
        self.seed = seed

        # Bayes-optimal direction
        self.w = np.zeros(d)
        self.w[0] = 1.0

        # Group means equally spaced on unit sphere (first 2 dims)
        self.group_means = np.zeros((k, d))
        for a in range(k):
            angle = 2 * np.pi * a / k
            self.group_means[a, 0] = np.cos(angle)
            if d >= 2:
                self.group_means[a, 1] = np.sin(angle)

    def _calibrate_intercept(self, rng, n_calib=100_000):
        """Find intercept b such that Pr[Y=1] ≈ p_plus via bisection."""
        # Generate calibration samples
        A_cal = rng.choice(self.k, size=n_calib)
        X_cal = np.array([
            rng.normal(self.group_means[a], self.sigma, size=self.d)
            for a in A_cal
        ])
        logits_no_bias = X_cal @ self.w

        def marginal_pplus(b):
            probs = expit(logits_no_bias + b)
            return probs.mean() - self.p_plus

        # Search for the right intercept
        try:
            b_opt = brentq(marginal_pplus, -10.0, 10.0, xtol=1e-6)
        except ValueError:
            b_opt = 0.0
        return b_opt

    def generate(self):
        """Generate the full dataset.

        Returns
        -------
        X : ndarray, shape (n_pool, d)
            Feature matrix.
        Y : ndarray, shape (n_pool,)
            Binary labels {0, 1}.
        A : ndarray, shape (n_pool,)
            Group membership {0, 1, ..., k-1}.
        """
        rng = np.random.RandomState(self.seed)

        # Calibrate intercept
        b = self._calibrate_intercept(rng)

        # Generate group assignments (uniform)
        A = rng.choice(self.k, size=self.n_pool)

        # Generate features
        X = np.array([
            rng.normal(self.group_means[a], self.sigma, size=self.d)
            for a in A
        ])

        # Generate labels
        logits = X @ self.w + b
        if self.sigma_eta > 0:
            logits += rng.normal(0, self.sigma_eta, size=self.n_pool)
        probs = expit(logits)
        Y = (rng.uniform(size=self.n_pool) < probs).astype(int)

        return X, Y, A

    def get_params(self):
        """Return dictionary of generator parameters."""
        return {
            'd': self.d, 'k': self.k, 'p_plus': self.p_plus,
            'n_pool': self.n_pool, 'sigma': self.sigma,
            'sigma_eta': self.sigma_eta, 'seed': self.seed
        }
