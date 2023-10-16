"""Functions for training sparse and full linear probes."""

from typing import List, Union, Tuple, Dict, Any, Optional

import torch as t
import numpy as np
import pandas as pd
from einops import rearrange
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.feature_selection import f_classif, f_regression
from sklearn import metrics
from tqdm.auto import tqdm


def f_classif_fixed(
    x: np.ndarray, y: np.ndarray, eps: float = 1e-6, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Handle columns with zero variance, hackily"""
    # TODO: only with columns that actually have zero variance?
    x_old = x[0, :] * 1.0
    x[0, :] += eps
    ret = f_classif(x, y, **kwargs)
    x[0, :] = x_old
    return ret


def f_regression_fixed(
    x: np.ndarray, y: np.ndarray, eps: float = 1e-6, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Handle columns with zero variance, hackily"""
    x_old = x[0, :] * 1.0
    x[0, :] += eps
    ret = f_regression(x, y, **kwargs)
    x[0, :] = x_old
    return ret


def get_sort_inds_and_ranks(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Utility function to get the sort indices and ranks of a 1D array"""
    sort_inds = x.argsort()
    ranks = np.empty_like(sort_inds)
    ranks[sort_inds] = np.arange(len(x))
    return sort_inds, ranks


def linear_probe(
    x: Union[np.ndarray, t.Tensor],
    y: Union[np.ndarray, t.Tensor],
    model_type: str = "classifier",
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    x_rearrange: Optional[str] = "b ... -> b (...)",
    y_rearrange: Optional[str] = None,
    sparse_num: Optional[int] = None,
    sparse_frac: Optional[float] = None,
    sparse_method: Optional[str] = None,
    **regression_kwargs,
) -> Dict[str, Any]:
    """Perform a linear probe (classification or ridge regression) on a provided
    X, y dataset.  Performs train/test split based on provided test_size, and
    passes any additional keyword arguments to the constructor of the regression
    object.  Input arrays can be optionally rearranged to handle extra
    dimensions in various ways, i.e. by folding them into the final
    batch dimension, or the final feature dimension.

    Returns a dict of scores, the trained model, and other relevant
    results."""
    # Check arguments as needed
    assert model_type in ["classifier", "ridge"], "Invalid model type"
    assert (
        sparse_num is None or sparse_frac is None
    ), "Cannot specify both sparse_num and sparse_frac"
    assert sparse_method in [None, "f_test"], "Invalid sparse method"
    assert (
        sparse_method is None
        or sparse_num is not None
        or sparse_frac is not None
    ), "Must specify sparse_num or sparse_frac if sparse_method is not None"
    # Convert to numpy arrays if necessary
    if isinstance(x, t.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, t.Tensor):
        y = y.cpu().numpy()
    # Rearrange if necessary
    if x_rearrange is not None:
        x = rearrange(x, x_rearrange)
    if y_rearrange is not None:
        y = rearrange(y, y_rearrange)
    # Split into train and test set
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )  # type: ignore
    # Get the sparse indices if requested
    if sparse_frac is not None:
        sparse_num = int(sparse_frac * x.shape[1])
    if sparse_method is not None:
        if sparse_method == "f_test":
            if model_type == "classifier":
                f_test_train, _ = f_classif_fixed(x_train, y_train)
            elif model_type == "ridge":
                f_test_train, _ = f_regression_fixed(x_train, y_train)
            else:
                raise ValueError("Invalid model type")
            sort_inds_train = f_test_train.argsort()
        else:
            raise ValueError("Invalid sparse method")
        top_k_inds_train = sort_inds_train[::-1][:sparse_num]
        # Update all x data to only use sparse features
        x = x[:, top_k_inds_train]
        x_train = x_train[:, top_k_inds_train]
        x_test = x_test[:, top_k_inds_train]
    # Create an appropriate classifier
    if model_type == "classifier":
        mdl = LogisticRegression(
            random_state=random_state, **regression_kwargs
        )
    elif model_type == "ridge":
        mdl = Ridge(random_state=random_state, **regression_kwargs)
    else:
        raise ValueError("Invalid model type")
    # Train!
    mdl.fit(x_train, y_train)
    y_pred = mdl.predict(x_test)
    # Return the results
    result = {
        "train_score": mdl.score(x_train, y_train),
        "test_score": mdl.score(x_test, y_test),
        "x": x,
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "model": mdl,
    }
    if sparse_method is not None:
        result["sparse_inds"] = top_k_inds_train  # type: ignore
    if model_type == "classifier":
        result["conf_matrix"] = metrics.confusion_matrix(y_test, y_pred)
        result["report"] = metrics.classification_report(y_test, y_pred)
    return result


def linear_probes(
    xys: List[Union[Tuple[np.ndarray, np.ndarray], Tuple[t.Tensor, t.Tensor]]],
    model_type: str = "classifier",
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    show_progress: bool = True,
    **regression_kwargs,
) -> pd.DataFrame:
    """Perform multiple linear probes on a list of X-Y data pairs,
    returning a DataFrame of results. Train/test split random state is
    kept the same for each probe."""
    if random_state is None:
        random_state = np.random.randint(0, 2**32 - 1)
    results = []
    for x, y in tqdm(xys, disable=not show_progress):
        result = linear_probe(
            x,
            y,
            model_type=model_type,
            test_size=test_size,
            random_state=random_state,
            **regression_kwargs,
        )
        results.append(result)
    results_df = pd.DataFrame(results)
    return results_df


def linear_probes_over_dim(
    x: Union[np.ndarray, t.Tensor],
    y: Union[np.ndarray, t.Tensor],
    dim: int = 1,
    model_type: str = "classifier",
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    **regression_kwargs,
) -> pd.DataFrame:
    """Convenience function to perform multiple linear probes, one for
    each slice of the provided X array along the specified dimension.
    For example, this could be used to probe individually on each
    channel of a multi-channel convolutional layer activation."""
    xys = []
    for ii in range(x.shape[dim]):
        xys.append(
            (
                np.take(
                    x if isinstance(x, np.ndarray) else x.cpu().numpy(),
                    ii,
                    axis=dim,
                ),
                y,
            )
        )
    return linear_probes(
        xys,
        model_type=model_type,
        test_size=test_size,
        random_state=random_state,
        **regression_kwargs,
    )
