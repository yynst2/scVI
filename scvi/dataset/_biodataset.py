import torch
import scipy
import pandas as pd
import anndata
import numpy as np
import logging

from torch.utils.data import Dataset
from typing import Union, List, Dict, Tuple
from scvi.dataset._anndata import get_from_registry
from scvi.dataset._utils import _check_nonnegative_integers

from scvi.dataset._constants import _X_KEY, _LABELS_KEY

logger = logging.getLogger(__name__)


class BioDataset(Dataset):
    def __init__(
        self,
        adata: anndata.AnnData,
        getitem_tensors: Union[List[str], Dict[str, type]] = None,
    ):
        assert (
            "scvi_data_registry" in adata.uns_keys()
        ), "Please register your anndata first."

        stats = adata.uns["scvi_summary_stats"]
        error_msg = "Number of {} in anndata different from when setup_anndata was run. Please rerun setup_anndata."
        assert adata.shape[0] == stats["n_cells"], error_msg.format("cells")
        assert adata.shape[1] == stats["n_genes"], error_msg.format("gene")
        assert (
            len(np.unique(get_from_registry(adata, _LABELS_KEY))) == stats["n_labels"]
        ), error_msg.format("labels")
        if "protein_expression" in adata.uns["scvi_data_registry"].keys():
            assert (
                stats["n_proteins"]
                == get_from_registry(adata, "protein_expression").shape[1]
            ), error_msg.format("proteins")

        is_nonneg_int = _check_nonnegative_integers(get_from_registry(adata, _X_KEY))
        if not is_nonneg_int:
            logger.warning(
                "Make sure the registered X field in anndata contains unnormalized count data."
            )

        self.adata = adata
        self.attributes_and_types = None
        self.setup_getitem(getitem_tensors)
        self.n_batches = len(
            np.unique(get_from_registry(self.adata, key="batch_indices"))
        )
        self.gene_names = self.adata.var_names
        self.norm_X = None

    def get_registered_keys(self,):
        """Returns the keys of the mappings in scvi data registry
        """
        return self.adata.uns["scvi_data_registry"].keys()

    def setup_getitem(self, getitem_tensors: Union[List[str], Dict[str, type]] = None):
        """Sets up the getitem function

        By default, getitem will return every single item registered in the scvi data registry
        and also set their type to np.float32.

        If you want to specify which specific tensors to return you can pass in a List of keys from the scvi data registry
        If you want to speficy specific tensors to return as well as their associated types, then you can pass in a dictionary with their type.

        Paramaters
        ----------
        getitem_tensors:
            Either a list of keys in the scvi data registry to return when getitem is called
            or

        Example
        -------
        bd = BioDataset(adata)

        #following will only return the X and batch_indices both by defualt as np.float32
        bd.setup_getitem(getitem_tensors  = ['X,'batch_indices'])

        #This will return X as an integer and batch_indices as np.float32
        bd.setup_getitem(getitem_tensors  = {'X':np.int64, 'batch_indices':np.float32])
        """
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

    def __len__(self):
        return self.adata.shape[0]

    def normalize(self):
        # TODO change to add a layer in anndata and update registry, store as sparse?
        X = get_from_registry(self.adata, _X_KEY)
        scaling_factor = X.mean(axis=1)
        self.norm_X = X / scaling_factor.reshape(len(scaling_factor), 1)

    def raw_counts_properties(
        self, idx1: Union[List[int], np.ndarray], idx2: Union[List[int], np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Computes and returns some statistics on the raw counts of two sub-populations.

        Parameters
        ----------
        idx1
            subset of indices describing the first population.
        idx2
            subset of indices describing the second population.
        Returns
        -------
        type
            Tuple of ``np.ndarray`` containing, by pair (one for each sub-population),
            mean expression per gene, proportion of non-zero expression per gene, mean of normalized expression.

        """
        X = get_from_registry(self.adata, _X_KEY)
        mean1 = (X[idx1]).mean(axis=0)
        mean2 = (X[idx2]).mean(axis=0)
        nonz1 = (X[idx1] != 0).mean(axis=0)
        nonz2 = (X[idx2] != 0).mean(axis=0)
        if self.norm_X is None:
            self.normalize()
        norm_mean1 = self.norm_X[idx1, :].mean(axis=0)
        norm_mean2 = self.norm_X[idx2, :].mean(axis=0)
        return (
            np.squeeze(np.asarray(mean1)),
            np.squeeze(np.asarray(mean2)),
            np.squeeze(np.asarray(nonz1)),
            np.squeeze(np.asarray(nonz2)),
            np.squeeze(np.asarray(norm_mean1)),
            np.squeeze(np.asarray(norm_mean2)),
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data_numpy = {
            key: get_from_registry(self.adata, key)
            for key, _ in self.attributes_and_types.items()
        }
        data_numpy = {
            key: data_numpy[key][idx].astype(dtype)
            if isinstance(data_numpy[key], np.ndarray)
            else data_numpy[key].iloc[idx, :].to_numpy().astype(dtype)
            if isinstance(data_numpy[key], pd.DataFrame)
            else data_numpy[key][idx].toarray().astype(dtype)
            for key, dtype in self.attributes_and_types.items()
        }
        return data_numpy

    @property
    def n_cells(self) -> int:
        """Returns the number of cells in the dataset
        """
        n_cells = self.adata.uns["scvi_summary_stats"]["n_cells"]
        assert n_cells == self.adata.shape[0]
        return n_cells

    @property
    def n_genes(self) -> int:
        """Returns the number of genes in the dataset
        """
        n_genes = self.adata.uns["scvi_summary_stats"]["n_genes"]
        assert n_genes == self.adata.shape[1]
        return n_genes

    @property
    def n_batch(self) -> int:
        return self.adata.uns["scvi_summary_stats"]["n_batch"]

    @property
    def protein_names(self) -> List[str]:
        """Returns list of protein names
        """
        if "protein_names" in self.adata.uns["scvi_summary_stats"].keys():
            return self.adata.uns["scvi_summary_stats"]["protein_names"]
        else:
            return None

    @property
    def protein_expression(self) -> np.ndarray:
        if "protein_expression" in self.adata.uns["scvi_data_registry"].keys():
            pe = get_from_registry(self.adata, "protein_expression")
            return pe.to_numpy() if type(pe) is pd.DataFrame else pe
        else:
            return None

    @property
    def X(self) -> np.ndarray:
        dtype = self.attributes_and_types[_X_KEY]
        data = get_from_registry(self.adata, _X_KEY)
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        elif scipy.sparse.issparse(data):
            data = data.toarray().astype(dtype)
        return data.astype(dtype)

    @property
    def labels(self) -> np.ndarray:
        dtype = self.attributes_and_types[_LABELS_KEY]
        data = get_from_registry(self.adata, _LABELS_KEY)
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        elif scipy.sparse.issparse(data):
            data = data.toarray().astype(dtype)
        return data.astype(dtype)

    def to_anndata(self,) -> anndata.AnnData:
        return self.adata.copy()
