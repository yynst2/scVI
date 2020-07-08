import os
import zipfile

import pandas as pd

from scvi.dataset._utils import _download
from scvi.dataset import setup_anndata
from scvi.dataset import DownloadableDataset, CellMeasurement


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
    df_coordinates = pd.read_csv(os.path.join(data_path, coordinates_filename))
    coordinates = CellMeasurement(
        name="coords",
        data=df_coordinates[["X", "Y"]],
        columns_attr_name="axis",
        columns=["x", "y"],
    )
    cell_attributes_name_mapping = {
        "Cell ID": "cell_id",
        "Field of View": "field_of_view",
    }
    if self.tissue_region == "subventricular cortex":
        cell_attributes_name_mapping.update({"Region": "region"})
    cell_attributes_dict = {}
    for column_name, attribute_name in cell_attributes_name_mapping.items():
        cell_attributes_dict[attribute_name] = df_coordinates[column_name]
    self.populate_from_data(
        X=df_counts.values,
        gene_names=df_counts.columns,
        Ys=[coordinates],
        cell_attributes_dict=cell_attributes_dict,
    )
