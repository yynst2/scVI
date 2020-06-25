import numpy as np
import logging
import pandas as pd
import anndata
import scipy.sparse as sp_sparse
import scanpy as sc

from typing import Optional

logger = logging.getLogger(__name__)


def _seurat_v3_highly_variable_genes(
    adata, n_top_genes: int = 4000, batch_key: str = None
):
    """An adapted implementation of the "vst" feature selection in Seurat v3.

        This function will be replaced once it's full implemented in scanpy
        https://github.com/theislab/scanpy/pull/1182

        For further details of the arithmetic see https://www.overleaf.com/read/ckptrbgzzzpg

    Parameters
    ----------
    adata
        anndata object
    n_top_genes
        How many variable genes to return
    batch_key
        key in adata.obs that contains batch info. If None, do not use batch info

    """

    from scanpy.preprocessing._utils import _get_mean_var
    from skmisc.loess import loess

    if batch_key is None:
        batch_info = pd.Categorical(np.zeros(adata.shape[0], dtype=int))
    else:
        batch_info = adata.obs[batch_key]

    norm_gene_vars = []
    for b in np.unique(batch_info):

        mean, var = _get_mean_var(adata[batch_info == b].X)
        not_const = var > 0
        estimat_var = np.zeros(adata.shape[1])

        y = np.log10(var[not_const])
        x = np.log10(mean[not_const])

        model = loess(x, y, span=0.3, degree=2)
        model.fit()
        estimat_var[not_const] = model.outputs.fitted_values
        reg_std = np.sqrt(10 ** estimat_var)

        batch_counts = adata[batch_info == b].X.astype(np.float64).copy()
        # clip large values as in Seurat
        N = np.sum(batch_info == b)
        vmax = np.sqrt(N)
        clip_val = reg_std * vmax + mean
        if sp_sparse.issparse(batch_counts):
            batch_counts = sp_sparse.csr_matrix(batch_counts)
            mask = batch_counts.data > clip_val[batch_counts.indices]
            batch_counts.data[mask] = clip_val[batch_counts.indices[mask]]
        else:
            clip_val_broad = np.broadcast_to(clip_val, batch_counts.shape)
            np.putmask(batch_counts, batch_counts > clip_val_broad, clip_val_broad)

        if sp_sparse.issparse(batch_counts):
            squared_batch_counts_sum = np.array(batch_counts.power(2).sum(axis=0))
            batch_counts_sum = np.array(batch_counts.sum(axis=0))
        else:
            squared_batch_counts_sum = np.square(batch_counts).sum(axis=0)
            batch_counts_sum = batch_counts.sum(axis=0)

        norm_gene_var = (1 / ((N - 1) * np.square(reg_std))) * (
            (N * np.square(mean))
            + squared_batch_counts_sum
            - 2 * batch_counts_sum * mean
        )
        norm_gene_vars.append(norm_gene_var.reshape(1, -1))

    norm_gene_vars = np.concatenate(norm_gene_vars, axis=0)
    # argsort twice gives ranks
    ranked_norm_gene_vars = np.argsort(np.argsort(norm_gene_vars, axis=1), axis=1)
    median_norm_gene_vars = np.median(norm_gene_vars, axis=0)
    median_ranked = np.median(ranked_norm_gene_vars, axis=0)

    num_batches_high_var = np.sum(
        ranked_norm_gene_vars >= (adata.X.shape[1] - n_top_genes), axis=0
    )
    df = pd.DataFrame(index=np.array(adata.var_names))
    df["highly_variable_nbatches"] = num_batches_high_var
    df["highly_variable_median_rank"] = median_ranked

    df["highly_variable_median_variance"] = median_norm_gene_vars
    df.sort_values(
        ["highly_variable_nbatches", "highly_variable_median_rank"],
        ascending=False,
        na_position="last",
        inplace=True,
    )
    df["highly_variable"] = False
    df.loc[:n_top_genes, "highly_variable"] = True
    df = df.loc[adata.var_names]

    adata.var["highly_variable"] = df["highly_variable"].values
    if len(np.unique(batch_info)) > 1:
        batches = adata.obs[batch_key].cat.categories
        adata.var["highly_variable_nbatches"] = df["highly_variable_nbatches"].values
        adata.var["highly_variable_intersection"] = df[
            "highly_variable_nbatches"
        ] == len(batches)
    adata.var["highly_variable_median_rank"] = df["highly_variable_median_rank"].values
    adata.var["highly_variable_median_variance"] = df[
        "highly_variable_median_variance"
    ].values


def highly_variable_genes(
    adata: anndata.AnnData,
    n_top_genes: int = None,
    flavor: Optional[str] = "seurat_v3",
    batch_correction: Optional[bool] = True,
    **highly_var_genes_kwargs,
) -> pd.DataFrame:
    """\
    Code adapted from the scanpy package
    Annotate highly variable genes [Satija15]_ [Zheng17]_ [Stuart19]_.
    Depending on `flavor`, this reproduces the R-implementations of Seurat v2 and earlier
    [Satija15]_ and Cell Ranger [Zheng17]_, and Seurat v3 [Stuart19]_.

    Parameters
    ----------
    n_top_genes
        Number of highly-variable genes to keep. Mandatory for Seurat v3
    flavor
        Choose the flavor for computing normalized dispersion. One of "seurat_v2", "cell_ranger",
        "seurat_v3". In their default workflows, Seurat v2 passes the cutoffs whereas Cell Ranger passes
        `n_top_genes`.
    batch_correction
        Whether batches should be taken into account during procedure
    **highly_var_genes_kwargs
        Kwargs to feed to highly_variable_genes when using
        the Seurat V2 flavor.

    Returns
    -------
    type
        scanpy .var DataFrame providing genes information including means, dispersions
        and whether the gene is tagged highly variable (key `highly_variable`)
        (see scanpy highly_variable_genes documentation)

    """

    if flavor not in ["seurat_v2", "seurat_v3", "cell_ranger"]:
        raise ValueError(
            "Choose one of the following flavors: 'seurat_v2', 'seurat_v3', 'cell_ranger'"
        )

    if flavor == "seurat_v3" and n_top_genes is None:
        raise ValueError("n_top_genes must not be None with flavor=='seurat_v3'")

    logger.info("extracting highly variable genes using {} flavor".format(flavor))

    # Creating AnnData structure
    obs = pd.DataFrame(
        data=dict(batch=self.batch_indices.squeeze()), index=np.arange(self.nb_cells),
    ).astype("category")

    counts = self.X.copy()
    adata = sc.AnnData(X=counts, obs=obs)
    batch_key = "batch" if (batch_correction and self.n_batches >= 2) else None
    if flavor != "seurat_v3":
        if flavor == "seurat_v2":
            # name expected by scanpy
            flavor = "seurat"
        # Counts normalization
        sc.pp.normalize_total(adata, target_sum=1e4)
        # logarithmed data
        sc.pp.log1p(adata)

        # Finding top genes
        sc.pp.highly_variable_genes(
            adata=adata,
            n_top_genes=n_top_genes,
            flavor=flavor,
            batch_key=batch_key,
            inplace=True,  # inplace=False looks buggy
            **highly_var_genes_kwargs,
        )
    elif flavor == "seurat_v3":
        _seurat_v3_highly_variable_genes(
            adata, n_top_genes=n_top_genes, batch_key=batch_key
        )
    else:
        raise ValueError(
            "flavor should be one of 'seurat_v2', 'cell_ranger', 'seurat_v3'"
        )

    return adata.var


def corrupt(self, rate: float = 0.1, corruption: str = "uniform"):
    """Forms a corrupted_X attribute containing a corrupted version of X.

    Sub-samples ``rate * self.X.shape[0] * self.X.shape[1]`` entries
    and perturbs them according to the ``corruption`` method.
    Namely:
        - "uniform" multiplies the count by a Bernouilli(0.9)
        - "binomial" replaces the count with a Binomial(count, 0.2)
    A corrupted version of ``self.X`` is stored in ``self.corrupted_X``.

    Parameters
    ----------
    rate
        Rate of corrupted entries.
    corruption
        Corruption method.
    """
    self.corrupted_X = copy.deepcopy(self.X)
    if corruption == "uniform":  # multiply the entry n with a Ber(0.9) random variable.
        i, j = self.X.nonzero()
        ix = np.random.choice(len(i), int(np.floor(rate * len(i))), replace=False)
        i, j = i[ix], j[ix]
        self.corrupted_X[i, j] = np.squeeze(
            np.asarray(
                np.multiply(
                    self.X[i, j],
                    np.random.binomial(n=np.ones(len(ix), dtype=np.int32), p=0.9),
                )
            )
        )
    elif (
        corruption == "binomial"
    ):  # replace the entry n with a Bin(n, 0.2) random variable.
        i, j = (k.ravel() for k in np.indices(self.X.shape))
        ix = np.random.choice(len(i), int(np.floor(rate * len(i))), replace=False)
        i, j = i[ix], j[ix]
        self.corrupted_X[i, j] = np.squeeze(
            np.asarray(np.random.binomial(n=(self.X[i, j]).astype(np.int32), p=0.2))
        )
    else:
        raise NotImplementedError("Unknown corruption method.")


def organize_cite_seq_cell_ranger(adata):

    raise NotImplementedError
