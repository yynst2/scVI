import csv
import anndata
import pandas as pd
import logging
import os
import numpy as np

from scvi.dataset._utils import _download
from scvi.dataset import setup_anndata

logger = logging.getLogger(__name__)


def cortex(save_path: str = "data/", run_setup_anndata=True):
    """
    Loads cortex dataset
    """
    save_path = os.path.abspath(save_path)
    url = "https://storage.googleapis.com/linnarsson-lab-www-blobs/blobs/cortex/expression_mRNA_17-Aug-2014.txt"
    save_fn = "expression.bin"
    _download(url, save_path, save_fn)
    adata = _load_cortex_txt(os.path.join(save_path, save_fn))
    if run_setup_anndata:
        setup_anndata(adata, labels_key="labels")
    return adata


def _load_cortex_txt(path_to_file):
    logger.info("Loading Cortex data from {}".format(path_to_file))
    rows = []
    gene_names = []
    with open(path_to_file, "r") as csvfile:
        data_reader = csv.reader(csvfile, delimiter="\t")
        for i, row in enumerate(data_reader):
            if i == 1:
                precise_clusters = np.asarray(row, dtype=str)[2:]
            if i == 8:
                clusters = np.asarray(row, dtype=str)[2:]
            if i >= 11:
                rows.append(row[1:])
                gene_names.append(row[0])
    cell_types, labels = np.unique(clusters, return_inverse=True)
    _, precise_labels = np.unique(precise_clusters, return_inverse=True)
    X = np.asarray(rows, dtype=np.int).T[1:]
    gene_names = np.asarray(gene_names, dtype=np.str)
    gene_indices = []

    extra_gene_indices = []
    gene_indices = np.concatenate([gene_indices, extra_gene_indices]).astype(np.int32)
    if gene_indices.size == 0:
        gene_indices = slice(None)

    X = X[:, gene_indices]
    gene_names = gene_names[gene_indices]
    X_df = pd.DataFrame(X, columns=gene_names)
    adata = anndata.AnnData(X=X_df)
    adata.obs["labels"] = labels
    adata.obs["precise_labels"] = precise_clusters
    adata.obs["cell_type"] = clusters
    logger.info("Finished loading Cortex data")
    return adata
