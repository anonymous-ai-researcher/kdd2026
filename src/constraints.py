"""
Constrained ERM via exponentiated gradient reduction.

Uses fairlearn's ExponentiatedGradient meta-algorithm for solving:
    min_{h: F(h) <= gamma/2} err(h)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


def solve_constrained_erm(X_train, Y_train, A_train, gamma, notion='EO',
                          C=1.0, max_iter=1000):
    """Solve constrained empirical risk minimization.

    Parameters
    ----------
    X_train : ndarray, shape (n, d)
        Training features.
    Y_train : ndarray, shape (n,)
        Training labels.
    A_train : ndarray, shape (n,)
        Training group memberships.
    gamma : float
        Fairness tolerance (constraint uses gamma/2).
    notion : str
        Fairness notion: 'DP', 'EO', or 'EqOdds'.
    C : float
        Logistic regression regularization.
    max_iter : int
        Maximum iterations for the solver.

    Returns
    -------
    model : fitted model with .predict(X) method.
    """
    try:
        from fairlearn.reductions import (
            ExponentiatedGradient,
            DemographicParity,
            EqualizedOdds,
            TruePositiveRateParity,
        )
    except ImportError:
        raise ImportError(
            "Please install fairlearn: pip install fairlearn==0.9.0"
        )

    base_estimator = LogisticRegression(
        C=C, solver='lbfgs', max_iter=max_iter, tol=1e-4
    )

    # Select constraint
    constraint_tolerance = gamma / 2.0
    if notion == 'DP':
        constraint = DemographicParity(difference_bound=constraint_tolerance)
    elif notion == 'EO':
        constraint = TruePositiveRateParity(
            difference_bound=constraint_tolerance
        )
    elif notion == 'EqOdds':
        constraint = EqualizedOdds(difference_bound=constraint_tolerance)
    else:
        raise ValueError(f"Unknown fairness notion: {notion}")

    mitigator = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=constraint,
        max_iter=50,
        eps=1e-4,
    )

    mitigator.fit(X_train, Y_train, sensitive_features=A_train)
    return mitigator


def solve_unconstrained_erm(X_train, Y_train, C=1.0, max_iter=1000):
    """Solve standard (unconstrained) ERM."""
    model = LogisticRegression(
        C=C, solver='lbfgs', max_iter=max_iter, tol=1e-4
    )
    model.fit(X_train, Y_train)
    return model
