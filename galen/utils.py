import copy
import numpy as np
from typing import Dict, Tuple
from scvi.dataset.dataset import compute_library_size

# old register_anndata option
# def register_anndata(adata, batch_loc, local_mean_loc, local_var_loc, labels_loc):
#     registry_dict = {}
#     registry_dict['X'] = (None, 'X')
#     registry_dict['batch_indices'] = batch_loc
#     registry_dict['local_l_mean'] = local_mean_loc
#     registry_dict['local_l_var'] = local_var_loc
#     registry_dict['labels'] = labels_loc
#     adata.uns["scvi_data_registry"] = registry_dict


def register_anndata(adata, data_registry_dict: Dict[str, Tuple[str, str]]):
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
    data_loc = adata.uns["scvi_data_registry"][key]
    df, df_key = data_loc[0], data_loc[1]
    if df is not None:
        return getattr(adata, df)[df_key]
    else:
        return getattr(adata, df_key)


def compute_library_size_batch(
    adata,
    batch_key: str,
    local_l_mean_key: str = None,
    local_l_var_key: str = None,
    in_place: bool = True,
):
    assert batch_key in adata.obs_keys(), "batch_key not valid key in obs dataframe"
    local_means = np.zeros((adata.shape[0], 1))
    local_vars = np.zeros((adata.shape[0], 1))
    batch_indices = adata.obs[batch_key]
    for i_batch in range(len(np.unique(batch_indices))):
        idx_batch = np.squeeze(batch_indices == i_batch)
        (local_means[idx_batch], local_vars[idx_batch],) = compute_library_size(
            adata[idx_batch].X
        )
    if local_l_mean_key is None:
        local_l_mean_key = "_scvi_local_l_mean"
    if local_l_var_key is None:
        local_l_var_key = "_scvi_local_l_var"

    if in_place:
        adata.obs[local_l_mean_key] = local_means
        adata.obs[local_l_var_key] = local_vars
    else:
        copy = adata.copy()
        copy.obs[local_l_mean_key] = local_means
        copy.obs[local_l_var_key] = local_vars
        return copy
