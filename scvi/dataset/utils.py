import copy
import numpy as np
import scipy.sparse as sp_sparse
import logging

from typing import Dict, Tuple, Union
from scvi.dataset.constants import (
    X_KEY,
    BATCH_KEY,
    LOCAL_L_MEAN_KEY,
    LOCAL_L_VAR_KEY,
    LABELS_KEY,
)

logger = logging.getLogger(__name__)


def register_anndata(adata, data_registry_dict: Dict[str, Tuple[str, str]]):
    """Registers the AnnData object by adding data_registry_dict to adata.uns
    
    Format is: {<scvi_key>: (<anndata dataframe>, <dataframe key> )}
    Example: 
    {"batch" :("obs", "batch_idx")}
    {"X": (None, "X")}
 
    Parameters
    ----------
    adata
        anndata object
    data_registry_dict
        dictionary mapping keys used by scvi models to their respective location in adata.

    """
    for df, df_key in data_registry_dict.values():
        if df is not None:
            assert df_key in getattr(
                adata, df
            ), "anndata.{} has no attribute '{}'".format(df, df_key)
        else:
            assert (
                hasattr(adata, df_key) == True
            ), "anndata has no attribute '{}'".format(df_key)
    adata.uns["scvi_data_registry"] = copy.copy(data_registry_dict)


def get_from_registry(adata, key: str):
    """Returns an the object in Anndata associated the key in adata.uns['scvi_data_registry']
    
    Parameters
    ----------
    adata
        anndata object
    key
        key of object to get from adata.uns['scvi_data_registry']
    
    Returns
    -------
    
    """
    assert "scvi_data_registry" in adata.uns.keys(), "AnnData was never registered"
    data_loc = adata.uns["scvi_data_registry"][key]
    df, df_key = data_loc[0], data_loc[1]
    if df is not None:
        return getattr(adata, df)[df_key]
    else:
        return getattr(adata, df_key)


def compute_library_size(
    data: Union[sp_sparse.csr_matrix, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    sum_counts = data.sum(axis=1)
    masked_log_sum = np.ma.log(sum_counts)
    if np.ma.is_masked(masked_log_sum):
        logger.warning(
            "This dataset has some empty cells, this might fail scVI inference."
            "Data should be filtered with `my_dataset.filter_cells_by_count()"
        )
    log_counts = masked_log_sum.filled(0)
    local_mean = (np.mean(log_counts).reshape(-1, 1)).astype(np.float32)
    local_var = (np.var(log_counts).reshape(-1, 1)).astype(np.float32)
    return local_mean, local_var


def compute_library_size_batch(
    adata,
    batch_key: str,
    local_l_mean_key: str = None,
    local_l_var_key: str = None,
    X_layers_key=None,
    copy: bool = False,
):
    """Computes the library size  

    Parameters
    ----------
    adata 
        anndata object containing counts
    batch_key
        key in obs for batch information
    local_l_mean_key
        key in obs to save the local log mean
    local_l_var_key
        key in obs to save the local log variance 
    X_layers_key
        if not None, will use this in adata.layers[] for X
    copy
        if True, returns a copy of the adata

    Returns
    -------
    type
        anndata.AnnData if copy was True, else None

    """
    assert batch_key in adata.obs_keys(), "batch_key not valid key in obs dataframe"
    local_means = np.zeros((adata.shape[0], 1))
    local_vars = np.zeros((adata.shape[0], 1))
    batch_indices = adata.obs[batch_key]
    for i_batch in np.unique(batch_indices):
        idx_batch = np.squeeze(batch_indices == i_batch)
        if X_layers_key is not None:
            assert (
                X_layers_key in adata.layers.keys()
            ), "X_layers_key not a valid key for adata.layers"
            data = adata[idx_batch].layers[X_layers_key]
        else:
            data = adata[idx_batch].X
        (local_means[idx_batch], local_vars[idx_batch],) = compute_library_size(data)
    if local_l_mean_key is None:
        local_l_mean_key = "_scvi_local_l_mean"
    if local_l_var_key is None:
        local_l_var_key = "_scvi_local_l_var"

    if copy:
        copy = adata.copy()
        copy.obs[local_l_mean_key] = local_means
        copy.obs[local_l_var_key] = local_vars
        return copy
    else:
        adata.obs[local_l_mean_key] = local_means
        adata.obs[local_l_var_key] = local_vars


def check_nonnegative_integers(X: Union[np.ndarray, sp_sparse.csr_matrix]):
    """Checks values of X to ensure it is count data
    """

    data = X if type(X) is np.ndarray else X.data
    # Check no negatives
    if np.any(data < 0):
        return False
    # Check all are integers
    elif np.any(~np.equal(np.mod(data, 1), 0)):
        return False
    else:
        return True


def setup_anndata(
    adata,
    batch_key: str = None,
    labels_key: str = None,
    X_layers_key: str = None,
    copy: bool = False,
):
    """Sets up anndata object for scVI models. This method will compute the log mean and log variance per batch. 
    A mapping will be created between in

    Parameters
    ----------
    adata
        anndata object containing raw counts
    batch_key
        key in adata.obs for batch information. Will automatically be converted into integer categories
    labels_key
        key in adata.obs for label information. Will automatically be converted into integer categories
    X_layers_key
        if not None, uses this as the key in adata.layers for raw count
    copy
        if True, a copy of anndata is returned

    Returns
    -------
    """

    # add logging for everything
    if copy:
        adata = adata.copy()

    ###checking layers
    if X_layers_key is None:
        check_nonnegative_integers(adata.X)
    else:
        assert (
            X_layers_key in adata.layers.keys()
        ), "{} is not a valid key in adata.layers".format(X_layers_key)
        check_nonnegative_integers(adata.layers[X_layers_key])

    ###checking batch
    if batch_key is None:
        batch_key = "_scvi_batch"
        adata.obs[batch_key] = np.zeros(adata.shape[0])
    else:
        assert (
            batch_key in adata.obs.keys()
        ), "{} is not a valid key in adata.obs".format(batch_key)
    # check the datatype of batches. if theyre not integers, make them ints
    user_batch_dtype = adata.obs[batch_key].dtype
    if np.issubdtype(user_batch_dtype, np.integer) is False:
        adata.obs["_scvi_batch"] = adata.obs[batch_key].astype("category").cat.codes
        batch_key = "_scvi_batch"

    if labels_key is None:
        labels_key = "_scvi_labels"
        adata.obs[labels_key] = np.zeros(adata.shape[0])
    else:
        assert (
            labels_key in adata.obs.keys()
        ), "{} is not a valid key in adata.obs".format(labels_key)
    # check the datatype of labels. if theyre not integers, make them ints
    user_labels_dtype = adata.obs[labels_key].dtype
    if np.issubdtype(user_labels_dtype, np.integer) is False:
        adata.obs["_scvi_labels"] = adata.obs[labels_key].astype("category").cat.codes
        labels_key = "_scvi_labels"

    # computes the library size per batch
    local_l_mean_key = "_scvi_local_l_mean"
    local_l_var_key = "_scvi_local_l_var"

    compute_library_size_batch(
        adata,
        batch_key=batch_key,
        local_l_mean_key=local_l_mean_key,
        local_l_var_key=local_l_var_key,
        X_layers_key=X_layers_key,
    )

    if X_layers_key is None:
        X_loc = None
        X_key = "X"
    else:
        X_loc = "layers"
        X_key = X_layers_key

    data_registry = {
        X_KEY: (X_loc, X_key),
        BATCH_KEY: ("obs", batch_key),
        LOCAL_L_MEAN_KEY: ("obs", local_l_mean_key),
        LOCAL_L_VAR_KEY: ("obs", local_l_var_key),
        LABELS_KEY: ("obs", labels_key),
    }

    register_anndata(adata, data_registry_dict=data_registry)

    n_batch = len(np.unique(adata.obs[batch_key]))
    n_cells = adata.shape[0]
    n_genes = adata.shape[1]
    summary_stats = {"n_batch": n_batch, "n_cells": n_cells, "n_genes": n_genes}
    adata.uns["scvi_summary_stats"] = summary_stats
    if copy:
        return adata

