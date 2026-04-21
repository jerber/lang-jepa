from src.common.stats import spearman_rho


def test_spearman_perfect_positive():
    assert spearman_rho([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) > 0.999


def test_spearman_perfect_negative():
    assert spearman_rho([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) < -0.999


def test_spearman_is_rank_invariant():
    # Any monotonic transformation leaves Spearman ρ unchanged.
    a = [1, 2, 3, 4, 5]
    b = [1.0, 4.0, 9.0, 16.0, 25.0]
    assert spearman_rho(a, b) > 0.999


def test_spearman_degenerate_inputs():
    # Zero variance on one side should return 0.0, not NaN.
    assert spearman_rho([1, 1, 1, 1], [1, 2, 3, 4]) == 0.0
