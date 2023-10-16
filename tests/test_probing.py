import numpy as np
import torch as t
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression

from circrl.probing import linear_probe, linear_probes, linear_probes_over_dim


def check_probe_result(result):
    assert isinstance(result, dict)
    assert "train_score" in result
    assert "test_score" in result
    assert "x" in result
    assert "x_train" in result
    assert "y_train" in result
    assert "x_test" in result
    assert "y_test" in result
    assert "model" in result
    assert result["train_score"] >= 0.0 and result["train_score"] <= 1.0
    assert result["test_score"] >= 0.0 and result["test_score"] <= 1.0
    assert result["x"].shape == (100, 10)
    assert result["x_train"].shape[0] == 80
    assert result["x_test"].shape[0] == 20
    assert result["y_train"].shape[0] == 80
    assert result["y_test"].shape[0] == 20
    assert isinstance(result["model"], type(LogisticRegression()))


def test_linear_probe():
    # Test with numpy arrays and Pytorch tensors
    rng = np.random.default_rng(42)
    x = rng.random((100, 10))
    y = rng.integers(0, 2, size=(100,))
    result = linear_probe(x, y)
    check_probe_result(result)
    # Test with PyTorch tensors
    t.manual_seed(42)
    x = t.rand(100, 10)
    y = t.randint(0, 2, size=(100,))
    result = linear_probe(x, y)
    check_probe_result(result)


def test_linear_probe_classification():
    # Generate random classification data
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    # Test logistic regression
    result = linear_probe(X, y, model_type="classifier", random_state=42)
    assert result["train_score"] > 0.8
    assert result["test_score"] > 0.8
    assert result["conf_matrix"].shape == (2, 2)
    assert result["report"] is not None


def test_linear_probe_regression():
    # Generate random regression data
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    # Test ridge regression
    result = linear_probe(X, y, model_type="ridge", random_state=42)
    assert result["train_score"] > 0.8
    assert result["test_score"] > 0.8


def test_linear_probes():
    # Generate multiple random classification data sets
    X1, y1 = make_classification(n_samples=100, n_features=10, random_state=42)
    X2, y2 = make_classification(n_samples=100, n_features=10, random_state=43)
    X3, y3 = make_classification(n_samples=100, n_features=10, random_state=44)
    # Test multiple linear probes
    results_df = linear_probes([(X1, y1), (X2, y2), (X3, y3)], random_state=42)
    print(results_df)
    assert len(results_df) == 3
    assert all(results_df["train_score"] > 0.8)
    assert all(results_df["test_score"] > 0.8)


def test_linear_probes_over_dim():
    # Generate a random classification data set
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    # Concatenate X 3 times over a new dimension between the existing
    # batch and feature dimensions, so that X.shape == (100, 3, 10),
    # and add a small random variation to each copy of X
    rng = np.random.default_rng(42)
    X = np.concatenate(
        [X[:, None, :] + rng.random((100, 1, 10)) * 0.01 for _ in range(3)],
        axis=1,
    )
    # Test linear probes over each channel
    results_df = linear_probes_over_dim(X, y, dim=1, random_state=42)
    assert len(results_df) == 3
    assert all(results_df["train_score"] > 0.8)
    assert all(results_df["test_score"] > 0.8)
