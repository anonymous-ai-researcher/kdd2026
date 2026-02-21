"""Unit tests for fairness metrics."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fairness.metrics import (
    compute_dp_gap,
    compute_eo_gap,
    compute_eqodds_gap,
    compute_group_rates,
)


class TestDPGap:
    def test_perfect_parity(self):
        """DP gap should be 0 when all groups have same prediction rate."""
        y_pred = np.array([1, 0, 1, 0, 1, 0])
        A = np.array([0, 0, 1, 1, 2, 2])
        assert compute_dp_gap(y_pred, A) == 0.0

    def test_maximal_gap(self):
        """DP gap should be 1 when one group all-positive, another all-negative."""
        y_pred = np.array([1, 1, 0, 0])
        A = np.array([0, 0, 1, 1])
        assert compute_dp_gap(y_pred, A) == 1.0

    def test_no_labels_needed(self):
        """DP should not depend on true labels."""
        y_pred = np.array([1, 0, 1, 1])
        A = np.array([0, 0, 1, 1])
        # Should work without y_true
        gap = compute_dp_gap(y_pred, A)
        assert isinstance(gap, float)


class TestEOGap:
    def test_equal_tpr(self):
        """EO gap should be 0 when all groups have same TPR."""
        y_pred = np.array([1, 0, 1, 0])
        y_true = np.array([1, 0, 1, 0])
        A = np.array([0, 0, 1, 1])
        assert compute_eo_gap(y_pred, y_true, A) == 0.0

    def test_unequal_tpr(self):
        """EO gap should reflect TPR difference."""
        # Group 0: TPR = 1/1 = 1.0
        # Group 1: TPR = 0/1 = 0.0
        y_pred = np.array([1, 0, 0, 0])
        y_true = np.array([1, 0, 1, 0])
        A = np.array([0, 0, 1, 1])
        assert compute_eo_gap(y_pred, y_true, A) == 1.0

    def test_no_positives(self):
        """EO gap should be 0 when no group has positives."""
        y_pred = np.array([1, 0, 1, 0])
        y_true = np.array([0, 0, 0, 0])
        A = np.array([0, 0, 1, 1])
        assert compute_eo_gap(y_pred, y_true, A) == 0.0


class TestEqOddsGap:
    def test_combines_tpr_and_fpr(self):
        """EqOdds should return max of TPR gap and FPR gap."""
        # Group 0: TPR=1, FPR=0
        # Group 1: TPR=0, FPR=1
        y_pred = np.array([1, 0, 0, 1])
        y_true = np.array([1, 0, 1, 0])
        A = np.array([0, 0, 1, 1])
        gap = compute_eqodds_gap(y_pred, y_true, A)
        assert gap == 1.0


class TestGroupRates:
    def test_basic(self):
        """Test that group rates are computed correctly."""
        y_pred = np.array([1, 1, 0, 0, 1, 0])
        y_true = np.array([1, 0, 1, 0, 1, 0])
        A = np.array([0, 0, 0, 1, 1, 1])

        rates = compute_group_rates(y_pred, y_true, A, k=2)

        # Group 0: ppr=2/3, tpr=1/1=1.0, fpr=1/2=0.5
        assert abs(rates['ppr'][0] - 2/3) < 1e-10
        assert abs(rates['tpr'][0] - 1.0) < 1e-10
        assert abs(rates['fpr'][0] - 0.5) < 1e-10

        # Group 1: ppr=1/3, tpr=1/1=1.0, fpr=0/2=0.0
        assert abs(rates['ppr'][1] - 1/3) < 1e-10
        assert abs(rates['tpr'][1] - 1.0) < 1e-10
        assert abs(rates['fpr'][1] - 0.0) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
