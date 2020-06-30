import numpy as np
import logging
import pandas as pd
import anndata
import scipy.sparse as sp_sparse
import copy

from typing import Optional

from ._utils import _check_nonnegative_integers

logger = logging.getLogger(__name__)


def highly_variable_genes_seurat_v3(
    adata: anndata.AnnData,
    layer: Optional[str] = None,
    n_top_genes: int = 2000,
    batch_key: Optional[str] = None,
    span: Optional[float] = 0.3,
    subset: bool = False,
    inplace: bool = True,
) -> Optional[pd.DataFrame]:
    """\
    Implements highly variable gene selection using the `vst` method in Seurat v3.

    Expects count data.

    This is a temporary implementation that will be replaced by the one in Scanpy.
    For further implemenation details see https://www.overleaf.com/read/ckptrbgzzzpg

    Parameters
    ----------
    adata
        The annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond
        to cells and columns to genes.
    layer
        If provided, use `adata.layers[layer]` for expression values instead of `adata.X`.
    n_top_genes
        Number of highly-variable genes to keep. Mandatory if `flavor='seurat_v3'`.
    span
        The fraction of the data (cells) used when estimating the variance in the loess
        model fit if `flavor='seurat_v3'`.
    subset
        Inplace subset to highly-variable genes if `True` otherwise merely indicate
        highly variable genes.
    inplace
        Whether to place calculated metrics in `.var` or return them.
    batch_key
        If specified, highly-variable genes are selected within each batch separately and merged.
        This simple process avoids the selection of batch-specific genes and acts as a
        lightweight batch correction method. Genes are first sorted by how many batches
        they are a HVG. Ties are broken by the median (across batches) rank based on
        within-batch normalized variance.

    Returns
    -------
    Depending on `inplace` returns calculated metrics (:class:`~pd.DataFrame`) or
    updates `.var` with the following fields

    highly_variable : bool
        boolean indicator of highly-variable genes
    **means**
        means per gene
    **variances**
        variance per gene
    **variances_norm**
        normalized variance per gene, averaged in the case of multiple batches
    highly_variable_rank : float
        Rank of the gene according to normalized variance, median rank in the case of multiple batches
    highly_variable_nbatches : int
        If batch_key is given, this denotes in how many batches genes are detected as HVG
    """
    from scanpy.preprocessing._utils import _get_mean_var

    try:
        from skmisc.loess import loess
    except ImportError:
        raise ImportError(
            "Please install skmisc package via `pip install --user scikit-misc"
        )

    X = adata.layers[layer] if layer is not None else adata.X
    if _check_nonnegative_integers(X) is False:
        raise ValueError(
            "`pp.highly_variable_genes` with `flavor='seurat_v3'` expects "
            "raw count data."
        )

    if batch_key is None:
        batch_info = pd.Categorical(np.zeros(adata.shape[0], dtype=int))
    else:
        batch_info = adata.obs[batch_key].values

    norm_gene_vars = []
    for b in np.unique(batch_info):

        ad = adata[batch_info == b]
        X = ad.layers[layer] if layer is not None else ad.X

        mean, var = _get_mean_var(X)
        not_const = var > 0
        estimat_var = np.zeros(adata.shape[1], dtype=np.float64)

        y = np.log10(var[not_const])
        x = np.log10(mean[not_const])
        model = loess(x, y, span=span, degree=2)
        model.fit()
        estimat_var[not_const] = model.outputs.fitted_values
        reg_std = np.sqrt(10 ** estimat_var)

        batch_counts = X.astype(np.float64).copy()
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
    # argsort twice gives ranks, small rank means most variable
    ranked_norm_gene_vars = np.argsort(np.argsort(-norm_gene_vars, axis=1), axis=1)

    # this is done in SelectIntegrationFeatures() in Seurat v3
    ranked_norm_gene_vars = ranked_norm_gene_vars.astype(np.float32)
    ranked_norm_gene_vars[ranked_norm_gene_vars >= n_top_genes] = np.nan
    median_ranked = np.nanmedian(ranked_norm_gene_vars, axis=0)

    with np.errstate(invalid="ignore"):
        num_batches_high_var = np.sum(
            (ranked_norm_gene_vars < n_top_genes).astype(int), axis=0
        )
    df = pd.DataFrame(index=np.array(adata.var_names))
    df["highly_variable_nbatches"] = num_batches_high_var
    df["highly_variable_rank"] = median_ranked
    df["variances_norm"] = np.mean(norm_gene_vars, axis=0)
    df["means"] = mean
    df["variances"] = var

    df.sort_values(
        ["highly_variable_rank", "highly_variable_nbatches"],
        ascending=[True, False],
        na_position="last",
        inplace=True,
    )
    df["highly_variable"] = False
    df.loc[: int(n_top_genes), "highly_variable"] = True
    df = df.loc[adata.var_names]

    if inplace or subset:
        adata.uns["hvg"] = {"flavor": "seurat_v3"}
        logger.info(
            "added\n"
            "    'highly_variable', boolean vector (adata.var)\n"
            "    'highly_variable_rank', float vector (adata.var)\n"
            "    'means', float vector (adata.var)\n"
            "    'variances', float vector (adata.var)\n"
            "    'variances_norm', float vector (adata.var)"
        )
        adata.var["highly_variable"] = df["highly_variable"].values
        adata.var["highly_variable_rank"] = df["highly_variable_rank"].values
        adata.var["means"] = df["means"].values
        adata.var["variances"] = df["variances"].values
        adata.var["variances_norm"] = df["variances_norm"].values.astype(
            "float64", copy=False
        )
        if batch_key is not None:
            adata.var["highly_variable_nbatches"] = df[
                "highly_variable_nbatches"
            ].values
        if subset:
            adata._inplace_subset_var(df["highly_variable"].values)
    else:
        if batch_key is None:
            df = df.drop(["highly_variable_nbatches"], axis=1)
        return df


def corrupt(
    adata: anndata.AnnData,
    layer: Optional[str] = None,
    rate: Optional[float] = 0.1,
    corruption: Optional[str] = "uniform",
    layer_key_added: Optional[str] = "corrupted_X",
):
    """Forms a `corrupted_X` layer containing a corrupted version of X.

    Sub-samples ``rate * adata.shape[0] * adata.shape[1]`` entries
    and perturbs them according to the ``corruption`` method.
    Namely:
        - "uniform" multiplies the count by a Bernouilli(0.9)
        - "binomial" replaces the count with a Binomial(count, 0.2)

    Parameters
    ----------
    adata
        The annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond
        to cells and columns to genes.
    layer
        If provided, use `adata.layers[layer]` for expression values instead of `adata.X`.
    rate
        Rate of corrupted entries.
    corruption
        Corruption method.
    layer_key_added
        key added to `adata.layers`

    Returns
    -------
    Adds `.layers[layer_key_added]` with corrupted version of the data.

    """
    X = adata.layers[layer] if layer is not None else adata.X
    corrupted_X = copy.deepcopy(X)
    if corruption == "uniform":  # multiply the entry n with a Ber(0.9) random variable.
        i, j = X.nonzero()
        ix = np.random.choice(len(i), int(np.floor(rate * len(i))), replace=False)
        i, j = i[ix], j[ix]
        corrupted_X[i, j] = np.squeeze(
            np.asarray(
                np.multiply(
                    X[i, j],
                    np.random.binomial(n=np.ones(len(ix), dtype=np.int32), p=0.9),
                )
            )
        )
    elif (
        corruption == "binomial"
    ):  # replace the entry n with a Bin(n, 0.2) random variable.
        i, j = (k.ravel() for k in np.indices(X.shape))
        ix = np.random.choice(len(i), int(np.floor(rate * len(i))), replace=False)
        i, j = i[ix], j[ix]
        corrupted_X[i, j] = np.squeeze(
            np.asarray(np.random.binomial(n=(X[i, j]).astype(np.int32), p=0.2))
        )
    else:
        raise NotImplementedError("Unknown corruption method.")

    adata.layers[layer_key_added] = corrupted_X


def organize_cite_seq_cell_ranger(adata):

    raise NotImplementedError
