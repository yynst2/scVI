import scipy.sparse as sp_sparse
import pandas as pd
import anndata
import pdb
import logging
import numpy as np
import os
import urllib.request

from typing import Union, Tuple

logger = logging.getLogger(__name__)

from scvi.dataset._constants import (
    _X_KEY,
    _BATCH_KEY,
    _LOCAL_L_MEAN_KEY,
    _LOCAL_L_VAR_KEY,
    _LABELS_KEY,
    _PROTEIN_EXP_KEY,
)


def _download(url: str, save_path: str, filename: str):
    """Writes data from url to file.
    """
    if os.path.exists(os.path.join(save_path, filename)):
        logger.info("File %s already downloaded" % (os.path.join(save_path, filename)))
        return
    req = urllib.request.Request(url, headers={"User-Agent": "Magic Browser"})
    r = urllib.request.urlopen(req)
    logger.info("Downloading file at %s" % os.path.join(save_path, filename))

    def read_iter(file, block_size=1000):
        """Given a file 'file', returns an iterator that returns bytes of
        size 'blocksize' from the file, using read().
        """
        while True:
            block = file.read(block_size)
            if not block:
                break
            yield block

    # Create the path to save the data
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, filename), "wb") as f:
        for data in read_iter(r):
            f.write(data)


def _unpack_tensors(tensors):
    x = tensors[_X_KEY]
    local_l_mean = tensors[_LOCAL_L_MEAN_KEY]
    local_l_var = tensors[_LOCAL_L_VAR_KEY]
    batch_index = tensors[_BATCH_KEY]
    y = tensors[_LABELS_KEY]
    return x, local_l_mean, local_l_var, batch_index, y


def _compute_library_size(
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


def _compute_library_size_batch(
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
        (local_means[idx_batch], local_vars[idx_batch]) = _compute_library_size(data)
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


def _check_nonnegative_integers(
    X: Union[pd.DataFrame, np.ndarray, sp_sparse.csr_matrix]
):
    """Checks values of X to ensure it is count data
    """

    if type(X) is np.ndarray:
        data = X
    elif issubclass(type(X), sp_sparse.spmatrix):
        data = X.data
    elif type(X) is pd.DataFrame:
        data = X.to_numpy()
    else:
        raise TypeError("X type not understood")
    # Check no negatives
    if np.any(data < 0):
        return False
    # Check all are integers
    elif np.any(~np.equal(np.mod(data, 1), 0)):
        return False
    else:
        return True


def _get_batch_mask_protein_data(
    adata: anndata.AnnData, protein_expression_obsm_key: str, batch_key: str
):
    """Returns a list with length number of batches where each entry is a mask over present
    cell measurement columns

    Parameters
    ----------
    attribute_name
        cell_measurement attribute name

    Returns
    -------
    type
        List of ``np.ndarray`` containing, for each batch, a mask of which columns were
        actually measured in that batch. This is useful when taking the union of a cell measurement
        over datasets.

    """
    pro_exp = adata.obsm[protein_expression_obsm_key]
    pro_exp = pro_exp.to_numpy() if type(pro_exp) is pd.DataFrame else pro_exp
    batches = adata.obs[batch_key].values
    batch_mask = []
    for b in np.unique(batches):
        b_inds = np.where(batches.ravel() == b)[0]
        batch_sum = pro_exp[b_inds, :].sum(axis=0)
        all_zero = batch_sum == 0
        batch_mask.append(~all_zero)

    return batch_mask
