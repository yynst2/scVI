import logging
import pdb
import os
import pandas as pd
import loompy
import numpy as np
import anndata

from scvi.dataset._utils import _download
from scvi.dataset import setup_anndata

logger = logging.getLogger(__name__)


def smfish(save_path="data/"):
    save_path = os.path.abspath(save_path)
    url = "http://linnarssonlab.org/osmFISH/osmFISH_SScortex_mouse_all_cells.loom"
    save_fn = "osmFISH_SScortex_mouse_all_cell.loom"
    _download(url, save_path, save_fn)
    adata = _load_smfish(os.path.join(save_path, save_fn))
    setup_anndata(adata)
    return adata


def _load_smfish(path_to_file):
    logger.info("Loading smFISH dataset")
    ds = loompy.connect(path_to_file)
    gene_names = ds.ra["Gene"].astype(np.str)
    labels = ds.ca["ClusterID"].reshape(-1, 1)
    tmp_cell_types = np.asarray(ds.ca["ClusterName"])

    u_labels, u_index = np.unique(labels.ravel(), return_index=True)
    cell_types = ["" for _ in range(max(u_labels) + 1)]
    for i, index in zip(u_labels, u_index):
        cell_types[i] = tmp_cell_types[index]
    cell_types = np.asarray(cell_types, dtype=np.str)

    x_coord, y_coord = ds.ca["X"], ds.ca["Y"]
    data = ds[:, :].T

    pdb.set_trace()

    adata = anndata.AnnData(
        X=data,
        obs={"x_coord": x_coord, "y_coord": y_coord},
        uns={"cell_types": cell_types},
    )
    adata.var_names = gene_names
    return adata
    # adata.
    # self.populate_from_data(
    #     X=data,
    #     labels=labels,
    #     gene_names=gene_names,
    #     cell_types=cell_types,
    #     cell_attributes_dict=
    #     remap_attributes=False,
    # )
    # major_clusters = dict(
    #     [
    #         ((3, 2), "Astrocytes"),
    #         ((7, 26), "Endothelials"),
    #         ((18, 17, 14, 19, 15, 16, 20), "Inhibitory"),
    #         ((29, 28), "Microglias"),
    #         ((32, 33, 30, 22, 21), "Oligodendrocytes"),
    #         ((9, 8, 10, 6, 5, 4, 12, 1, 13), "Pyramidals"),
    #     ]
    # )
    # if self.use_high_level_cluster:
    #     self.map_cell_types(major_clusters)
    #     self.filter_cell_types(
    #         [
    #             "Astrocytes",
    #             "Endothelials",
    #             "Inhibitory",
    #             "Microglias",
    #             "Oligodendrocytes",
    #             "Pyramidals",
    #         ]
    #     )

    # self.remap_categorical_attributes()
