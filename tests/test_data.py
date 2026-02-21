"""Unit tests for data generators."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.synthetic import SyntheticDataGenerator


class TestSyntheticGenerator:
    def test_shape(self):
        gen = SyntheticDataGenerator(d=5, k=3, n_pool=1000, seed=42)
        X, Y, A = gen.generate()
        assert X.shape == (1000, 5)
        assert Y.shape == (1000,)
        assert A.shape == (1000,)

    def test_binary_labels(self):
        gen = SyntheticDataGenerator(d=5, k=3, n_pool=1000, seed=42)
        X, Y, A = gen.generate()
        assert set(np.unique(Y)).issubset({0, 1})

    def test_group_range(self):
        gen = SyntheticDataGenerator(d=5, k=4, n_pool=1000, seed=42)
        X, Y, A = gen.generate()
        assert A.min() >= 0
        assert A.max() <= 3

    def test_approximate_base_rate(self):
        gen = SyntheticDataGenerator(
            d=5, k=4, p_plus=0.3, n_pool=50000, seed=42
        )
        X, Y, A = gen.generate()
        actual_p_plus = Y.mean()
        assert abs(actual_p_plus - 0.3) < 0.05, \
            f"Expected p+ ≈ 0.3, got {actual_p_plus:.3f}"

    def test_reproducibility(self):
        gen1 = SyntheticDataGenerator(d=5, k=3, n_pool=100, seed=42)
        X1, Y1, A1 = gen1.generate()
        gen2 = SyntheticDataGenerator(d=5, k=3, n_pool=100, seed=42)
        X2, Y2, A2 = gen2.generate()
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(Y1, Y2)
        np.testing.assert_array_equal(A1, A2)

    def test_uniform_groups(self):
        gen = SyntheticDataGenerator(d=5, k=4, n_pool=10000, seed=42)
        X, Y, A = gen.generate()
        for a in range(4):
            frac = (A == a).mean()
            assert abs(frac - 0.25) < 0.05, \
                f"Group {a} fraction = {frac:.3f}, expected ~0.25"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
