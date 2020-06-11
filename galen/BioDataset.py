import torch
import anndata
import copy
import numpy as np
import scipy as sp
from torch.utils.data import Dataset
from typing import Union, List, Dict, Tuple
from utils import register_anndata, get_from_registry


class BioDataset(Dataset):
    # should we have new name for getitem tensors?
    def __init__(
        self,
        adata: anndata.AnnData,
        getitem_tensors: Union[List[str], Dict[str, type]] = None,
    ):
        assert (
            "scvi_data_registry" in adata.uns_keys()
        ), "Please register your anndata first"

        self.adata = adata
        self.check_X()
        self.attributes_and_types = None
        self.setup_getitem(getitem_tensors)

    def get_registered_keys(self,):
        return self.adata.uns["scvi_data_registry"].keys()

    # TODO: i dont like this name, should probably change
    def setup_getitem(self, getitem_tensors: Union[List[str], Dict[str, type]] = None):
        registered_keys = self.get_registered_keys()

        if isinstance(getitem_tensors, List):
            keys = getitem_tensors
            keys_to_type = {key: np.float32 for key in keys}
        elif isinstance(getitem_tensors, Dict):
            keys = getitem_tensors.keys()
            keys_to_type = getitem_tensors
        elif getitem_tensors is None:
            keys = registered_keys
            keys_to_type = {key: np.float32 for key in keys}
        else:
            raise ValueError(
                "getitem_tensors invalid type. Expected: List[str] or Dict[str, type] or None"
            )

        for key in keys:
            assert (
                key in registered_keys
            ), "{} not in anndata.uns['scvi_data_registry']".format(key)

        self.attributes_and_types = keys_to_type

    def check_X(self,):
        # check if observations are unnormalized using first 10
        # code from: https://github.com/theislab/dca/blob/89eee4ed01dd969b3d46e0c815382806fbfc2526/dca/io.py#L63-L69
        if len(self.adata) > 10:
            X_subset = self.adata.X[:10]
        else:
            X_subset = self.adata.X
        norm_error = (
            "Make sure that the dataset (adata.X) contains unnormalized count data."
        )
        if sp.sparse.issparse(X_subset):
            assert (X_subset.astype(int) != X_subset).nnz == 0, norm_error
        else:
            assert np.all(X_subset.astype(int) == X_subset), norm_error

    def __len__(self):
        return self.adata.shape[0]

    # NOTE: the way this getitem is setup is that we can only get items from obs with an idx
    # meaning we're replicating data and is not the most memory efficient way of doing it
    # also the keys are hard coded. do we want to change it at all?
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # trying to preslice, but then X is 2d of [1, n_genes]
        # data = self.adata[idx]
        # data_numpy = {
        #     key: get_from_registry(data, key).astype(dtype)
        #     if isinstance(get_from_registry(data, key), np.ndarray)
        #     else np.array(get_from_registry(data, key)).astype(dtype)
        #     for key, dtype in self.attributes_and_types.items()
        # }

        data_numpy = {
            key: get_from_registry(self.adata, key)[idx].astype(dtype)
            if isinstance(get_from_registry(self.adata, key)[idx], np.ndarray)
            else np.array(get_from_registry(self.adata, key)[idx]).astype(dtype)
            for key, dtype in self.attributes_and_types.items()
        }
        data_torch = {k: torch.from_numpy(d) for k, d in data_numpy.items()}
        return data_torch

    @property
    def nb_cells(self) -> int:
        """Returns the number of cells in the dataset
        """
        return self.adata.shape[0]

    @property
    def nb_genes(self) -> int:
        """Returns the number of genes in the dataset
        """
        return self.adata.shape[1]

    def to_anndata(self,) -> anndata.AnnData:
        return self.adata.copy()
