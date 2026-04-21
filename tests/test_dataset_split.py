from src.common.datasets.sentences import is_val_doc


def test_is_val_deterministic():
    # Same text, same fraction → same assignment, always.
    text = "Hello world. This is a test document."
    assert is_val_doc(text, 0.1) == is_val_doc(text, 0.1) == is_val_doc(text, 0.1)


def test_is_val_ratio_approximates_fraction():
    # Over many docs, the empirical val rate should roughly match val_fraction.
    docs = [f"Sample document number {i}, with variety." for i in range(2000)]
    n_val = sum(is_val_doc(d, 0.1) for d in docs)
    assert 0.07 * len(docs) < n_val < 0.13 * len(docs)


def test_is_val_zero_means_never():
    docs = [f"Sample doc {i}" for i in range(200)]
    assert not any(is_val_doc(d, 0.0) for d in docs)


def test_is_val_one_captures_all_docs():
    # All bucket IDs are in [0, 1000), so val_fraction=1.0 → every doc is val.
    docs = [f"Sample doc {i}" for i in range(200)]
    assert all(is_val_doc(d, 1.0) for d in docs)
