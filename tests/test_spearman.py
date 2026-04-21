import sys

# Import the internal Spearman implementation used by eval_representations.
sys.path.insert(0, "scripts")
from eval_representations import _spearman  # noqa: E402


def test_spearman_perfect_positive():
    assert _spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) > 0.999


def test_spearman_perfect_negative():
    assert _spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) < -0.999


def test_spearman_is_rank_invariant():
    # Monotonic transformation shouldn't change Spearman.
    a = [1, 2, 3, 4, 5]
    b = [1.0, 4.0, 9.0, 16.0, 25.0]  # b = a^2
    assert _spearman(a, b) > 0.999


def test_spearman_degenerate_inputs():
    # All-equal input has zero variance; _spearman should return 0 not NaN.
    assert _spearman([1, 1, 1, 1], [1, 2, 3, 4]) == 0.0
