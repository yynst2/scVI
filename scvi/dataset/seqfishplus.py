import os
import zipfile

import pandas as pd
import anndata

from scvi.dataset._utils import _download


def seqfishplus(save_path="data/", tissue_region="subventricular cortex"):
    save_path = os.path.abspath(save_path)
    url = "https://github.com/CaiGroup/seqFISH-PLUS/raw/master/sourcedata.zip"
    save_fn = "seqfishplus.zip"
    _download(url, save_path, save_fn)
    adata = _load_seqfishplus(
        os.path.join(save_path, save_fn), delimiter="\t", gene_by_cell=False
    )
    return adata


def _load_seqfishplus(self):
    counts_filename = "sourcedata/{}_counts.csv".format(self.file_prefix)
    coordinates_filename = "sourcedata/{}_cellcentroids.csv".format(self.file_prefix)
    data_path = os.path.join(self.save_path, "seqfishplus")
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    with zipfile.ZipFile(os.path.join(self.save_path, self.filenames[0])) as f:
        f.extract(counts_filename, path=data_path)
        f.extract(coordinates_filename, path=data_path)
    df_counts = pd.read_csv(os.path.join(data_path, counts_filename))

    adata = anndata.AnnData(df_counts)
    df_coordinates = pd.read_csv(os.path.join(data_path, coordinates_filename))

    adata.obs["X"] = df_coordinates["X"].values
    adata.obs["Y"] = df_coordinates["Y"].values
    adata.obs["Cell ID"] = df_coordinates["cell_id"].values
    adata.obs["Field of View"] = df_coordinates["field_of_view"].values

    return adata
