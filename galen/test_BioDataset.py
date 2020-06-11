import numpy as np
import anndata
import torch

from BioDataset import BioDataset
from utils import register_anndata, get_from_registry, compute_library_size_batch


def test_BioDataset():
    # TODO change this path
    file_path = "/Users/galen/scVI/tests/data/pbmc_10k_protein_v3.h5ad"
    adata = anndata.read_h5ad(file_path)

    batch_idx = np.zeros(adata.shape[0])
    batch_idx[:100] += 1
    adata.obs["batch"] = batch_idx

    labels = np.arange(adata.shape[0])
    adata.obs["labels"] = labels

    compute_library_size_batch(
        adata, "batch", local_l_mean_key="mean", local_l_var_key="var", in_place=True
    )

    data_registry = {
        "X": (None, "X"),
        "batch": ("obs", "batch"),
        "local_l_mean": ("obs", "mean"),
        "local_l_var": ("obs", "var"),
        "labels_loc": ("obs", "labels"),
    }

    register_anndata(adata, data_registry)
    test_BioDataset_getitem(adata)


def test_BioDataset_getitem(adata):
    # check that we can successfully pass in a list of tensors to get
    tensors_to_get = ["batch", "local_l_var"]
    bd = BioDataset(adata, getitem_tensors=tensors_to_get)
    np.testing.assert_array_equal(tensors_to_get, list(bd[1].keys()))

    # check that we can successfully pass in a dict of tensors and their associated types
    bd = BioDataset(adata, getitem_tensors={"X": np.int, "local_l_var": np.float64})
    assert bd[1]["X"].dtype == torch.int64
    assert bd[1]["local_l_var"].dtype == torch.float64

    # check that by default we get all the registered tensors
    bd = BioDataset(adata)
    all_registered_tensors = list(adata.uns["scvi_data_registry"].keys())
    np.testing.assert_array_equal(all_registered_tensors, list(bd[1].keys()))
    assert bd[1]["X"].shape[0] == bd.nb_genes


test_BioDataset()
