from dataclasses import dataclass
from typing import List, Union, Tuple, Dict, Any, Optional

import torch as t
import numpy as np
import pandas as pd
from einops import rearrange
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, f_regression
from sklearn import metrics
from tqdm.auto import tqdm


def f_classif_fixed(
    X: np.array, y: np.array, eps: float = 1e-6, **kwargs
) -> Tuple[np.array, np.array]:
    """Handle columns with zero variance, hackily"""
    # TODO: only with columns that actually have zero variance?
    X_old = X[0, :] * 1.0
    X[0, :] += eps
    ret = f_classif(X, y, **kwargs)
    X[0, :] = X_old
    return ret


def f_regression_fixed(
    X: np.array, y: np.array, eps: float = 1e-6, **kwargs
) -> Tuple[np.array, np.array]:
    """Handle columns with zero variance, hackily"""
    X_old = X[0, :] * 1.0
    X[0, :] += eps
    ret = f_regression(X, y, **kwargs)
    X[0, :] = X_old
    return ret


def get_sort_inds_and_ranks(x: np.ndarray) -> Tuple[np.array, np.array]:
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
    random_state: Optional[Union[int, np.random.RandomState]] = None,
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
        or sparse_frac is not None,
        "Must specify sparse_num or sparse_frac if sparse_method is not None",
    )
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
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    # Get the sparse indices if requested
    if sparse_frac is not None:
        sparse_num = int(sparse_frac * x.shape[1])
    if sparse_method == "f_test":
        if model_type == "classifier":
            f_test_train, _ = f_classif_fixed(x_train, y_train)
        elif model_type == "ridge":
            f_test_train, _ = f_regression_fixed(x_train, y_train)
        sort_inds_train = f_test_train.argsort()
    if sparse_method is not None:
        top_K_inds_train = sort_inds_train[-sparse_num:]
        # Update all x data to only use sparse features
        x = x[:, top_K_inds_train]
        x_train = x_train[:, top_K_inds_train]
        x_test = x_test[:, top_K_inds_train]
    # Create an appropriate classifier
    if model_type == "classifier":
        mdl = LogisticRegression(
            random_state=random_state, **regression_kwargs
        )
    elif model_type == "ridge":
        mdl = Ridge(random_state=random_state, **regression_kwargs)
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
        result["sparse_inds"] = top_K_inds_train
    if model_type == "classifier":
        result["conf_matrix"] = metrics.confusion_matrix(y_test, y_pred)
        result["report"] = metrics.classification_report(y_test, y_pred)
    return result


def linear_probes(
    xys: List[Union[Tuple[np.ndarray, np.ndarray], Tuple[t.Tensor, t.Tensor]]],
    model_type: str = "classifier",
    test_size: float = 0.2,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    show_progress: bool = True,
    **regression_kwargs,
) -> pd.DataFrame:
    """Perform multiple linear probes on a list of X-Y data pairs,
    returning a DataFrame of results. Train/test split random state is
    kept the same for each probe."""
    if random_state is None:
        random_state = np.random.RandomState()
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
    random_state: Optional[Union[int, np.random.RandomState]] = None,
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
                    x if isinstance(x, np.array) else x.cpu().numpy(),
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


# TODO: move sparse probing feature into linear probe function as an
# optional argument
# def sparse_linear_probe(
#     hook,
#     value_labels,
#     target,
#     model_type="classifier",
#     index_fracs=np.linspace(0, 1.0, 11, endpoint=True),
#     index_nums=None,
#     rank_method="f_test",
#     test_size=0.2,
#     do_scale_activations=False,
#     target_labels=None,
#     random_state=None,
#     **regression_kwargs,
# ):
#     """Run a probe to train linear classifiers (possibly multi-class) on
#     datasets of activations from a provided list of value labels, given a set
#     of target classfier outputs.  These classifiers can optionally be sparse,
#     in which case the activations will be ranked using the provided rank_method.
#     Either a list of fractions or nums can be provided to determine the top-K
#     activations to pull from the activations after they have been ranked.
#     The same training subset is used to rank activations and to train
#     all the classifers for a given value label.

#     Activation values can be optionally scaled using a StandardScaler before
#     training.  Default is yes.

#     Target can by an xarray or numpy array

#     Hook values must have been created such that their first dimension is
#     the dimension over the dataset batch.

#     Returns an array of trained classifier objects, with dims
#     (value_label, num_acts)"""
#     # TODO: check argument shapes, etc.

#     # Make sure target is a numpy array
#     if isinstance(target, xr.DataArray):
#         target = target.values

#     # Create target class labels if not provided
#     y = target
#     y_unique = np.unique(y)
#     num_target_labels = len(y_unique)
#     is_multiclass = num_target_labels > 2
#     if target_labels is None:
#         target_labels = y_unique

#     # Try all values if none provided
#     if value_labels is None:
#         value_labels = hook.values_by_label.keys()

#     # Sizes of results structure
#     num_values = len(value_labels)
#     num_index_nums = (
#         len(index_nums) if index_nums is not None else len(index_fracs)
#     )
#     results_shape = (num_values, num_index_nums)
#     results_dims = ["value_label", "index_num_step"]

#     # Structure to hold the results
#     probe_results = xr.Dataset(
#         dict(
#             model=xr.DataArray(
#                 data=np.zeros(results_shape, dtype=LogisticRegression),
#                 dims=results_dims,
#             ),
#             score=xr.DataArray(
#                 data=np.zeros(results_shape), dims=results_dims
#             ),
#             y_pred_all=xr.DataArray(
#                 data=np.zeros(results_shape + y.shape),
#                 dims=results_dims
#                 + ["batch"]
#                 + [f"yd{ii}" for ii in range(len(y.shape) - 1)],
#             ),
#         )
#     ).assign_coords(
#         dict(
#             value_label=value_labels, index_num_step=np.arange(num_index_nums)
#         )
#     )

#     if model_type == "classifier":
#         probe_results["conf_matrix"] = xr.DataArray(
#             data=np.zeros(
#                 results_shape + (num_target_labels, num_target_labels),
#                 dtype=int,
#             ),
#             dims=results_dims + ["test_label", "pred_label"],
#         )
#         probe_results["report"] = xr.DataArray(
#             data=np.zeros(results_shape, dtype=str), dims=results_dims
#         )
#         probe_results.assign_coords(
#             dict(test_label=target_labels, pred_label=target_labels)
#         )

#     # Iterate over value labels
#     for label in tqdm(value_labels):
#         value = hook.get_value_by_label(label)
#         try:
#             value = value.values  # Handle xarray values
#         except:
#             pass

#         # Create training data, reshaping activation value to 2D
#         X = rearrange(value, "b ... -> b (...)")
#         D_act = X.shape[1]

#         # Scale, if requested
#         if do_scale_activations:
#             scaler = StandardScaler()
#             X_scl = scaler.fit_transform(X)
#         else:
#             scaler = None
#             X_scl = X

#         # Split into train and test set
#         X_train, X_test, y_train, y_test = train_test_split(
#             X_scl, y, test_size=test_size, random_state=random_state
#         )

#         # Get activation ranking using provided method
#         if rank_method == "f_test":
#             if model_type == "classifier":
#                 f_test_train, _ = f_classif_fixed(X_train, y_train)
#             elif model_type == "ridge":
#                 f_test_train, _ = f_regression_fixed(X_train, y_train)
#             sort_inds_train, ranks_train = get_sort_inds_and_ranks(
#                 f_test_train
#             )

#         # Get the number of activations to use at each step, if not provided
#         if index_nums is None:
#             index_nums = np.array(index_fracs * D_act).astype(int)

#         # Iterate over activation nums, training with each set of activations
#         for kk, K in enumerate(index_nums):
#             # Get the indices of the activations to use
#             top_K_inds_train = sort_inds_train[-K:]

#             # Get the top-K data
#             X_top_train = X_train[:, top_K_inds_train]
#             X_top_test = X_test[:, top_K_inds_train]

#             # Create an appropriate classifier
#             if model_type == "classifier":
#                 if is_multiclass:
#                     # Multi-class regression
#                     mdl = LogisticRegression(
#                         multi_class="ovr",
#                         solver="liblinear",
#                         random_state=random_state,
#                         **regression_kwargs,
#                     )
#                 else:
#                     mdl = LogisticRegression(
#                         random_state=random_state, **regression_kwargs
#                     )
#             elif model_type == "ridge":
#                 mdl = Ridge(random_state=random_state, **regression_kwargs)

#             # Train!
#             mdl.fit(X_top_train, y_train)

#             # Test
#             # (Some useful reference code here: https://www.kaggle.com/code/satishgunjal/multiclass-logistic-regression-using-sklearn/notebook)
#             y_pred = mdl.predict(X_top_test)
#             score = mdl.score(X_top_test, y_test)
#             y_pred_all = mdl.predict(X_scl[:, top_K_inds_train])

#             # Collate all the results into an object
#             # relevant_indices.loc[dict(value=value_label, count_step=final_count_step)].values[()] = relinds_this
#             probe_results["model"].loc[
#                 dict(value_label=label, index_num_step=kk)
#             ].values[()] = mdl
#             probe_results["score"].loc[
#                 dict(value_label=label, index_num_step=kk)
#             ].values[()] = score
#             probe_results["y_pred_all"].loc[
#                 dict(value_label=label, index_num_step=kk)
#             ].values[()] = y_pred_all

#             if model_type == "classifier":
#                 # if is_multiclass:
#                 #     conf_matrix = metrics.multilabel_confusion_matrix(y_test, y_pred,
#                 #         labels=target_labels)
#                 # else:
#                 #     conf_matrix = metrics.confusion_matrix(y_test, y_pred, labels=target_labels)
#                 conf_matrix = metrics.confusion_matrix(
#                     y_test, y_pred, labels=target_labels
#                 )
#                 report = metrics.classification_report(y_test, y_pred)

#                 probe_results["conf_matrix"].loc[
#                     dict(value_label=label, index_num_step=kk)
#                 ].values[:, :] = conf_matrix
#                 probe_results["report"].loc[
#                     dict(value_label=label, index_num_step=kk)
#                 ].values[()] = report

#     # Return the results
#     return probe_results, scaler
