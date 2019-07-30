from . import GeneExpressionDataset
from .anndataset import AnnDatasetFromAnnData, DownloadableAnnDataset
import torch
import pickle
import os
import numpy as np

import pandas as pd
import anndata


class AnnDatasetKeywords(GeneExpressionDataset):
    def __init__(self, data, select_genes_keywords=[]):

        super().__init__()

        if isinstance(data, str):
            anndataset = anndata.read(data)
        else:
            anndataset = data

        idx_and_gene_names = [
            (idx, gene_name) for idx, gene_name in enumerate(list(anndataset.var.index))
        ]
        for keyword in select_genes_keywords:
            idx_and_gene_names = [
                (idx, gene_name)
                for idx, gene_name in idx_and_gene_names
                if keyword.lower() in gene_name.lower()
            ]

        gene_indices = np.array([idx for idx, _ in idx_and_gene_names])
        gene_names = np.array([gene_name for _, gene_name in idx_and_gene_names])

        expression_mat = np.array(anndataset.X[:, gene_indices].todense())

        select_cells = expression_mat.sum(axis=1) > 0
        expression_mat = expression_mat[select_cells, :]

        select_genes = (expression_mat > 0).mean(axis=0) > 0.21
        gene_names = gene_names[select_genes]
        expression_mat = expression_mat[:, select_genes]

        print("Final dataset shape :", expression_mat.shape)

        self.populate_from_data(X=expression_mat, gene_names=gene_names)


class ZhengDataset(AnnDatasetKeywords):
    def __init__(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        zheng = anndata.read(os.path.join(current_dir, "zheng_gemcode_control.h5ad"))

        super(ZhengDataset, self).__init__(zheng, select_genes_keywords=["ercc"])


class MacosDataset(AnnDatasetKeywords):
    def __init__(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        macos = anndata.read(os.path.join(current_dir, "macosko_dropseq_control.h5ad"))

        super(MacosDataset, self).__init__(macos, select_genes_keywords=["ercc"])


class KleinDataset(AnnDatasetKeywords):
    def __init__(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        klein = anndata.read(
            os.path.join(current_dir, "klein_indrops_control_GSM1599501.h5ad")
        )

        super(KleinDataset, self).__init__(klein, select_genes_keywords=["ercc"])


class Sven1Dataset(AnnDatasetKeywords):
    def __init__(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        svens = anndata.read(
            os.path.join(current_dir, "svensson_chromium_control.h5ad")
        )

        sven1 = svens[svens.obs.query('sample == "20311"').index]
        super(Sven1Dataset, self).__init__(sven1, select_genes_keywords=["ercc"])


class Sven2Dataset(AnnDatasetKeywords):
    def __init__(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        svens = anndata.read(
            os.path.join(current_dir, "svensson_chromium_control.h5ad")
        )

        sven2 = svens[svens.obs.query('sample == "20312"').index]
        super(Sven2Dataset, self).__init__(sven2, select_genes_keywords=["ercc"])


class AnnDatasetRNA(GeneExpressionDataset):
    def __init__(self, data, n_genes=100):

        super().__init__()

        if isinstance(data, str):
            anndataset = anndata.read(data)
        else:
            anndataset = data

        # Select RNA genes
        idx_and_gene_names = [
            (idx, gene_name)
            for idx, gene_name in enumerate(list(anndataset.var.index))
            if "ercc" not in gene_name.lower()
        ]

        gene_indices = np.array([idx for idx, _ in idx_and_gene_names])
        gene_names = np.array([gene_name for _, gene_name in idx_and_gene_names])

        expression_mat = np.array(anndataset.X[:, gene_indices].todense())

        # Find n_genes most expressed genes (wrt average gene expression)
        argsort_genes_exp = np.argsort(np.mean(expression_mat, axis=0))
        expression_mat = expression_mat[:, argsort_genes_exp[-n_genes:]]
        gene_names = gene_names[argsort_genes_exp[-n_genes:]]

        # Remove zero cells, then zero genes
        select_cells = expression_mat.sum(axis=1) > 0
        expression_mat = expression_mat[select_cells, :]

        select_genes = (expression_mat > 0).mean(axis=0) >= 0.21
        gene_names = gene_names[select_genes]
        expression_mat = expression_mat[:, select_genes]

        print("Final dataset shape :", expression_mat.shape)

        self.populate_from_data(X=expression_mat, gene_names=gene_names)


class KleinDatasetRNA(AnnDatasetRNA):
    def __init__(self, n_genes=100):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        klein = anndata.read(
            os.path.join(current_dir, "klein_indrops_control_GSM1599501.h5ad")
        )

        super(KleinDatasetRNA, self).__init__(klein, n_genes=n_genes)


class Sven1DatasetRNA(AnnDatasetRNA):
    def __init__(self, n_genes=100):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        svens = anndata.read(
            os.path.join(current_dir, "svensson_chromium_control.h5ad")
        )

        sven1 = svens[svens.obs.query('sample == "20311"').index]
        super(Sven1DatasetRNA, self).__init__(sven1, n_genes=n_genes)


class Sven2DatasetRNA(AnnDatasetRNA):
    def __init__(self, n_genes=100):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        svens = anndata.read(
            os.path.join(current_dir, "svensson_chromium_control.h5ad")
        )

        sven2 = svens[svens.obs.query('sample == "20312"').index]
        super(Sven2DatasetRNA, self).__init__(sven2, n_genes=n_genes)


class AnnDatasetMixed(GeneExpressionDataset):
    def __init__(self, data, matching_func="l2", n_matches=3, threshold=0.01):

        super().__init__()

        assert matching_func in [
            "l2",
            "l2_sort",
            "means",
            "cosine",
            "cosine_sort",
            "random",
        ]
        self.matching_func = matching_func
        self.n_matches = n_matches

        if isinstance(data, str):
            anndataset = anndata.read(data)
        else:
            anndataset = data

        expression_mat = np.array(anndataset.X.todense())

        # Select ERCC genes
        ercc_idx_and_gene_names = [
            (idx, gene_name)
            for idx, gene_name in enumerate(list(anndataset.var.index))
            if "ercc" in gene_name.lower()
        ]

        # Eliminate zero cells and zero genes
        select_cells = expression_mat.sum(axis=1) > 0
        expression_mat = expression_mat[select_cells, :]
        select_genes = expression_mat.sum(axis=0) > 0
        expression_mat = expression_mat[:, select_genes]

        # Select ERCC genes
        gene_names = np.array(
            [
                gene_name
                for idx, gene_name in enumerate(list(anndataset.var.index))
                if select_genes[idx]
            ]
        )

        ercc_gene_indices = np.array(
            [
                idx
                for idx, gene_name in enumerate(gene_names)
                if "ercc" in gene_name.lower()
            ]
        )

        # Match ERCC genes with RNA genes, select matched genes
        selected_matched_genes = self._match_genes(expression_mat, ercc_gene_indices)
        expression_mat = expression_mat[:, selected_matched_genes]
        gene_names = gene_names[selected_matched_genes]

        # Remove induced zero cells and keep only genes present in at least 21% of cells
        select_cells = expression_mat.sum(axis=1) > 0
        expression_mat = expression_mat[select_cells, :]

        select_genes = (expression_mat > 0).mean(axis=0) >= threshold
        gene_names = gene_names[select_genes]
        expression_mat = expression_mat[:, select_genes]

        print("Final dataset shape :", expression_mat.shape)
        print(
            "ERCC genes :",
            len([gene_name for gene_name in gene_names if "ercc" in gene_name.lower()]),
        )

        self.is_ercc = np.array(
            ["ercc" in gene_name.lower() for gene_name in gene_names]
        )

        self.populate_from_data(X=expression_mat, gene_names=gene_names)

    def _matching_func(self, ref_col, mat):

        if self.matching_func == "l2":
            return np.linalg.norm(mat - ref_col, axis=0)
        elif self.matching_func == "l2_sort":
            return np.linalg.norm(
                np.sort(mat, axis=0) - np.sort(ref_col, axis=0), axis=0
            )
        elif self.matching_func == "means":
            return np.abs(np.mean(mat, axis=0) - np.mean(ref_col))
        elif self.matching_func == "cosine":
            return 1.0 - np.sum(mat * ref_col, axis=0) / (
                np.linalg.norm(mat, axis=0) * np.linalg.norm(ref_col)
            )
        elif self.matching_func == "cosine_sort":
            return 1.0 - np.sum(
                np.sort(mat, axis=0) * np.sort(ref_col, axis=0), axis=0
            ) / (np.linalg.norm(mat, axis=0) * np.linalg.norm(ref_col))
        elif self.matching_func == "random":
            np.random.seed(0)
            return np.random.uniform(0.0, 100.0, size=(mat.shape[1],))
        else:
            raise Exception("Matching function not recognized")

    def _match_given_gene(self, expression_mat, ref_gene_index, selected_genes):

        scores = self._matching_func(
            expression_mat[:, ref_gene_index][:, np.newaxis], expression_mat
        )
        scores[selected_genes] = np.inf
        new_matches = np.arange(expression_mat.shape[1])[
            np.argsort(scores)[: self.n_matches]
        ]
        selected_genes[new_matches] = True

        return selected_genes

    def _match_genes(self, expression_mat, ref_gene_indices):

        selected_genes = np.zeros(shape=(expression_mat.shape[1],), dtype=bool)
        selected_genes[ref_gene_indices] = True

        if self.n_matches > 0:
            for ref_gene_index in ref_gene_indices:
                selected_genes = self._match_given_gene(
                    expression_mat, ref_gene_index, selected_genes
                )

        return selected_genes


class KleinDatasetMixed(AnnDatasetMixed):
    def __init__(self, matching_func="l2", n_matches=3, threshold=0.01):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        klein = anndata.read(
            os.path.join(current_dir, "klein_indrops_control_GSM1599501.h5ad")
        )

        super(KleinDatasetMixed, self).__init__(
            klein, matching_func=matching_func, n_matches=n_matches, threshold=threshold
        )


class Sven1DatasetMixed(AnnDatasetMixed):
    def __init__(self, matching_func="l2", n_matches=3, threshold=0.01):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        svens = anndata.read(
            os.path.join(current_dir, "svensson_chromium_control.h5ad")
        )

        sven1 = svens[svens.obs.query('sample == "20311"').index]
        super(Sven1DatasetMixed, self).__init__(
            sven1, matching_func=matching_func, n_matches=n_matches, threshold=threshold
        )


class Sven2DatasetMixed(AnnDatasetMixed):
    def __init__(self, matching_func="l2", n_matches=3, threshold=0.01):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        svens = anndata.read(
            os.path.join(current_dir, "svensson_chromium_control.h5ad")
        )

        sven2 = svens[svens.obs.query('sample == "20312"').index]
        super(Sven2DatasetMixed, self).__init__(
            sven2, matching_func=matching_func, n_matches=n_matches, threshold=threshold
        )
