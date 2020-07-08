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

_subtype_to_high_level_mapping = {
    "Astrocytes": ("Astrocyte Gfap", "Astrocyte Mfge8"),
    "Endothelials": ("Endothelial", "Endothelial 1"),
    "Inhibitory": (
        "Inhibitory Cnr1",
        "Inhibitory Kcnip2",
        "Inhibitory Pthlh",
        "Inhibitory Crhbp",
        "Inhibitory CP",
        "Inhibitory IC",
        "Inhibitory Vip",
    ),
    "Microglias": ("Perivascular Macrophages", "Microglia"),
    "Oligodendrocytes": (
        "Oligodendrocyte Precursor cells",
        "Oligodendrocyte COP" "Oligodendrocyte NF",
        "Oligodendrocyte Mature",
        "Oligodendrocyte MF",
    ),
    "Pyramidals": (
        "Pyramidal L2-3",
        "Pyramidal Cpne5",
        "Pyramidal L2-3 L5",
        "pyramidal L4",
        "Pyramidal L3-4",
        "Pyramidal Kcnip2",
        "Pyramidal L6",
        "Pyramidal L5",
        "Hippocampus",
    ),
}


def smfish(save_path="data/", use_high_level_cluster=True):
    save_path = os.path.abspath(save_path)
    url = "http://linnarssonlab.org/osmFISH/osmFISH_SScortex_mouse_all_cells.loom"
    save_fn = "osmFISH_SScortex_mouse_all_cell.loom"
    _download(url, save_path, save_fn)
    adata = _load_smfish(
        os.path.join(save_path, save_fn), use_high_level_cluster=use_high_level_cluster
    )
    return adata


def _load_smfish(path_to_file, use_high_level_cluster):
    logger.info("Loading smFISH dataset")
    ds = loompy.connect(path_to_file)
    gene_names = ds.ra["Gene"].astype(np.str)
    labels = ds.ca["ClusterID"].reshape(-1, 1)
    cell_types = np.asarray(ds.ca["ClusterName"])
    if use_high_level_cluster:
        pdb.set_trace()
        for high_level_cluster, subtypes in _subtype_to_high_level_mapping.items():
            for subtype in subtypes:
                idx = np.where(cell_types == subtype)
                cell_types[idx] = high_level_cluster
        pdb.set_trace()
        # for old_cell_type_idx, new_cell_type in major_clusters:
        #     pdb.set_trace()
    u_labels, u_index = np.unique(labels.ravel(), return_index=True)
    cell_types = ["" for _ in range(max(u_labels) + 1)]
    for i, index in zip(u_labels, u_index):
        cell_types[i] = tmp_cell_types[index]
    cell_types = np.asarray(cell_types, dtype=np.str)
    pdb.set_trace()

    x_coord, y_coord = ds.ca["X"], ds.ca["Y"]
    data = ds[:, :].T

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
