"""Unit tests for learning methods."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.synthetic import SyntheticDataGenerator
from src.methods.fair_stratified import FairStratified
from src.methods.passive_fair import PassiveFair
from src.utils.stopping import (
    theoretical_stopping_rule,
    calibrated_stopping_rule,
    compute_phase1_budget,
)


class TestStoppingRules:
    def test_theoretical_positive(self):
        n = theoretical_stopping_rule(gamma=0.05, k=4, delta=0.1)
        assert n > 0

    def test_calibrated_smaller(self):
        n_theo = theoretical_stopping_rule(gamma=0.05, k=4, delta=0.1)
        n_cal = calibrated_stopping_rule(gamma=0.05, k=4, delta=0.1)
        assert n_cal < n_theo, "Calibrated should be smaller than theoretical"

    def test_gamma_scaling(self):
        n1 = theoretical_stopping_rule(gamma=0.05, k=4, delta=0.1)
        n2 = theoretical_stopping_rule(gamma=0.10, k=4, delta=0.1)
        # Halving gamma should ~quadruple the budget
        assert n1 > 3 * n2, f"Expected ~4x scaling, got {n1/n2:.1f}x"

    def test_phase1_budget_eo(self):
        budget, n_plus = compute_phase1_budget(
            gamma=0.10, k=4, delta=0.1, p_a=0.25,
            use_calibrated=True, fairness='EO'
        )
        assert budget > 0
        assert n_plus > 0

    def test_phase1_budget_eqodds_larger(self):
        budget_eo, _ = compute_phase1_budget(
            gamma=0.10, k=4, delta=0.1, p_a=0.25,
            use_calibrated=True, fairness='EO'
        )
        budget_eq, _ = compute_phase1_budget(
            gamma=0.10, k=4, delta=0.1, p_a=0.25,
            use_calibrated=True, fairness='EqOdds'
        )
        assert budget_eq > budget_eo, "EqOdds should cost more than EO"


class TestFairStratified:
    @pytest.fixture
    def synthetic_data(self):
        gen = SyntheticDataGenerator(d=5, k=3, p_plus=0.3, n_pool=5000,
                                     seed=42)
        return gen.generate()

    def test_fit_predict(self, synthetic_data):
        X, Y, A = synthetic_data
        fs = FairStratified(gamma=0.15, delta=0.1, epsilon=0.1,
                            fairness='EO', use_calibrated=True)
        fs.fit(X, Y, A)
        y_pred = fs.predict(X[:100])
        assert len(y_pred) == 100
        assert set(np.unique(y_pred)).issubset({0, 1})

    def test_budget_breakdown(self, synthetic_data):
        X, Y, A = synthetic_data
        fs = FairStratified(gamma=0.15, delta=0.1, epsilon=0.1,
                            fairness='EO', use_calibrated=True)
        fs.fit(X, Y, A)
        budget = fs.get_label_budget()
        assert budget['phase1'] > 0
        assert budget['phase2'] > 0
        assert budget['total'] == budget['phase1'] + budget['phase2']


class TestPassiveFair:
    def test_fit_predict(self):
        rng = np.random.RandomState(42)
        n = 500
        X = rng.randn(n, 5)
        Y = (X[:, 0] > 0).astype(int)
        A = rng.choice(3, size=n)

        pf = PassiveFair(gamma=0.10, notion='EO')
        pf.fit(X, Y, A, seed=42)
        y_pred = pf.predict(X, A)
        assert len(y_pred) == n
        assert set(np.unique(y_pred)).issubset({0, 1})


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
