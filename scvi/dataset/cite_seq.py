import numpy as np
import pandas as pd
import scanpy as sc


def pbmcs_10x_cite_seq(protein_join="inner"):
    dataset1 = sc.read(
        "pbmc_10k_protein.h5ad",
        backup_url="https://github.com/YosefLab/scVI-data/raw/master/pbmc_10k_protein_v3.h5ad?raw=true",
    )
    dataset2 = sc.read(
        "pbmc_5k_protein.h5ad",
        backup_url="https://github.com/YosefLab/scVI-data/raw/master/pbmc_5k_protein_v3.h5ad?raw=true",
    )
    common_genes = dataset1.var_names.intersection(dataset2.var_names)
    dataset1 = dataset1[:, common_genes]
    dataset2 = dataset2[:, common_genes]
    dataset1.obsm["protein_df"] = pd.DataFrame(
        dataset1.obsm["protein_expression"],
        columns=dataset1.uns["protein_names"],
        index=dataset1.obs_names,
    )
    dataset2.obsm["protein_df"] = pd.DataFrame(
        dataset2.obsm["protein_expression"],
        columns=dataset2.uns["protein_names"],
        index=dataset2.obs_names,
    )
    dataset = dataset1.concatenate(dataset2, join=protein_join)
    dataset.obsm["protein_expression"] = dataset.obsm["protein_df"].values
    dataset.uns["protein_names"] = np.array(dataset.obsm["protein_df"].columns)
    del dataset.obsm["protein_df"]
    return dataset
