from app.auto_ab import ABTest
import unittest
import pytest
import scipy


class TestABTestParams(unittest.TestCase):
    def test_valid_alpha_alternative(self):
        ABTest(alpha=0.05, alternative='less')

    def test_invalid_alpha(self):
        with pytest.raises(Exception) as e:
            assert ABTest(alpha=1.05, alternative='less')
        assert str(e.value) == "Significance level must be inside interval [0, 1]. Your input: 1.05."

    def test_invalid_alternative(self):
        with pytest.raises(Exception) as e:
            assert ABTest(alpha=0.05, alternative='one-sided')
        assert str(e.value) == "Alternative must be either 'less', 'greater', or 'two-sided'. Your input: 'one-sided'."

    def test_power_range(self):
        pass

    def test_split_rates(self):
        pass