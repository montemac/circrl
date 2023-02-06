from dataclasses import dataclass

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

def f_classif_fixed(X, y, **kwargs):
    '''Handle columns with zero variance, hackily'''
    X_fixed = X
    X_fixed[0,:] += 1e-6
    return f_classif(X_fixed, y, **kwargs)

def f_regression_fixed(X, y, **kwargs):
    '''Handle columns with zero variance, hackily'''
    X_fixed = X
    X_fixed[0,:] += 1e-6
    return f_regression(X_fixed, y, **kwargs)

def get_sort_inds_and_ranks(x):
    sort_inds = x.argsort()
    ranks = np.empty_like(sort_inds)
    ranks[sort_inds] = np.arange(len(x))
    return sort_inds, ranks

def linear_probe(X_full, y, model_type='classifier', test_size=0.2, 
        **regression_kwargs):
    '''Perform a linear probe (classification or ridge regression) on a provided
    X, y dataset.  Performs train/test split based on provided test_size, and
    passes any additional keyword arguments to the constructor of the regression
    object.  Returns a dict of scores, the trained model, and other relevant 
    results objects.'''
    X = rearrange(X_full, 'b ... -> b (...)')
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
            test_size=0.2, random_state=42)

    if model_type == 'classifier':
        mdl = LogisticRegression(**regression_kwargs)
    elif model_type == 'ridge':
        mdl = Ridge(**regression_kwargs)
    mdl.fit(X_train, y_train)
    y_pred = mdl.predict(X_test)

    result = {
        'train_score': mdl.score(X_train, y_train),
        'test_score':  mdl.score(X_test, y_test),
        'X': X,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'model': mdl
    }

    if model_type == 'classifier':
        result['conf_matrix'] = metrics.confusion_matrix(y_test, y_pred)
        result['report'] = metrics.classification_report(y_test, y_pred)

    return result

def linear_probe_multi_channels(hook, value_label, y, ch_inds, **regression_kwargs):
    '''Probe with a set of channels from a specific value taken from the
    provided hook object.  Wrapper around linear_probe.'''
    value = hook.get_value_by_label(value_label)
    return linear_probe(value[:,ch_inds,:,:].values, y, **regression_kwargs)

def linear_probe_single_channels(hook, value_labels, y, **regression_kwargs):
    '''Probe with all individual channels in all specified values taken from the
    provided hook object.  Wrapper around linear_probe.'''
    results = []
    for value_label in tqdm(value_labels):

        value = hook.get_value_by_label(value_label)

        for ch_ind in tqdm(range(value.shape[1])):
            ch = value[:,ch_ind,:,:]
            results_this = linear_probe(ch.values, y, **regression_kwargs)
            results_this.update({
                'value_label': value_label,
                'channel': ch_ind,})
            results.append(results_this)
            
    results_df = pd.DataFrame(results).set_index(['value_label', 'channel']).sort_values(
        'test_score', ascending=False)
    return results_df

def sparse_linear_probe(hook, value_labels, target, 
        model_type = 'classifier',
        index_fracs = np.linspace(0, 1.0, 11, endpoint=True),
        index_nums = None,
        rank_method = 'f_test',
        test_size = 0.2,
        do_scale_activations = False,
        target_labels = None,
        **regression_kwargs):
    '''Run a probe to train linear classifiers (possibly multi-class) on
    datasets of activations from a provided list of value labels, given a set
    of target classfier outputs.  These classifiers can optionally be sparse,
    in which case the activations will be ranked using the provided rank_method.
    Either a list of fractions or nums can be provided to determine the top-K
    activations to pull from the activations after they have been ranked.
    The same training subset is used to rank activations and to train 
    all the classifers for a given value label.

    Activation values can be optionally scaled using a StandardScaler before
    training.  Default is yes.

    Target can by an xarray or numpy array

    Hook values must have been created such that their first dimension is 
    the dimension over the dataset batch.
    
    Returns an array of trained classifier objects, with dims 
    (value_label, num_acts)'''
    # TODO: check argument shapes, etc.

    # Make sure target is a numpy array
    if isinstance(target, xr.DataArray):
        target = target.values
    
    # Create target class labels if not provided
    y = target
    y_unique = np.unique(y)
    num_target_labels = len(y_unique)
    is_multiclass = num_target_labels > 2
    if target_labels is None:
        target_labels = y_unique
    
    # Try all values if none provided
    if value_labels is None:
        value_labels = hook.values_by_label.keys()

    # Sizes of results structure
    num_values = len(value_labels)
    num_index_nums = len(index_nums) if index_nums is not None \
        else len(index_fracs)
    results_shape = (num_values, num_index_nums)
    results_dims = ['value_label', 'index_num_step']

    # Structure to hold the results
    probe_results = xr.Dataset(dict(
        model = xr.DataArray(
            data=np.zeros(results_shape, dtype=LogisticRegression),
            dims=results_dims),
        score = xr.DataArray(
            data=np.zeros(results_shape), 
            dims=results_dims))).assign_coords(dict(
                value_label = value_labels,
                index_num_step = np.arange(num_index_nums)))
        
    if model_type == 'classifier':
        probe_results['conf_matrix'] = xr.DataArray(
            data=np.zeros(results_shape+(num_target_labels, num_target_labels), 
                dtype=int),
            dims=results_dims+['test_label', 'pred_label'])
        probe_results['report'] = xr.DataArray(data=np.zeros(results_shape, dtype=str), 
            dims=results_dims)
        probe_results.assign_coords(dict(
            test_label = target_labels,
            pred_label = target_labels))
            
    # Iterate over value labels
    for label in tqdm(value_labels):
        value = hook.get_value_by_label(label)

        # Create training data, reshaping activation value to 2D
        X = rearrange(value.values, 'b ... -> b (...)')
        D_act = X.shape[1]

        # Scale, if requested
        if do_scale_activations:
            scaler = StandardScaler()
            X_scl = scaler.fit_transform(X)
        else:
            scaler = None
            X_scl = X

        # Split into train and test set
        X_train, X_test, y_train, y_test = train_test_split(X_scl, y, 
            test_size=test_size)

        # Get activation ranking using provided method
        if rank_method == 'f_test':
            if model_type == 'classifier':
                f_test_train, _ = f_classif_fixed(X_train, y_train)
            elif model_type == 'ridge':
                f_test_train, _ = f_regression_fixed(X_train, y_train)
            sort_inds_train, ranks_train = get_sort_inds_and_ranks(f_test_train)


        # Get the number of activations to use at each step, if not provided
        if index_nums is None:
            index_nums = np.array(index_fracs * D_act).astype(int)
        
        # Iterate over activation nums, training with each set of activations
        for kk, K in enumerate(index_nums):
            # Get the indices of the activations to use
            top_K_inds_train = sort_inds_train[-K:]
            
            # Get the top-K data
            X_top_train = X_train[:,top_K_inds_train]
            X_top_test = X_test[:,top_K_inds_train]

            # Create an appropriate classifier
            if model_type == 'classifier':
                if is_multiclass:
                    # Multi-class regression
                    mdl = LogisticRegression(multi_class='ovr', 
                        solver='liblinear', **regression_kwargs)
                else:
                    mdl = LogisticRegression(**regression_kwargs)
            elif model_type == 'ridge':
                mdl = Ridge(**regression_kwargs)

            # Train!
            mdl.fit(X_top_train, y_train)

            # Test
            # (Some useful reference code here: https://www.kaggle.com/code/satishgunjal/multiclass-logistic-regression-using-sklearn/notebook)
            y_pred = mdl.predict(X_top_test)
            score = mdl.score(X_top_test, y_test)
            
            # Collate all the results into an object
            #relevant_indices.loc[dict(value=value_label, count_step=final_count_step)].values[()] = relinds_this
            probe_results['model'].loc[
                dict(value_label=label, index_num_step=kk)].values[()] = mdl
            probe_results['score'].loc[
                dict(value_label=label, index_num_step=kk)].values[()] = score
            
            if model_type == 'classifier':
                # if is_multiclass:
                #     conf_matrix = metrics.multilabel_confusion_matrix(y_test, y_pred, 
                #         labels=target_labels)
                # else:
                #     conf_matrix = metrics.confusion_matrix(y_test, y_pred, labels=target_labels)
                conf_matrix = metrics.confusion_matrix(y_test, y_pred, labels=target_labels)
                report = metrics.classification_report(y_test, y_pred)

                probe_results['conf_matrix'].loc[
                    dict(value_label=label, index_num_step=kk)].values[:,:] = conf_matrix
                probe_results['report'].loc[
                    dict(value_label=label, index_num_step=kk)].values[()] = report

    # Return the results
    return probe_results, scaler