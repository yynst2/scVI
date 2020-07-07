from unittest import TestCase
import numpy as np
from scvi.dataset import setup_anndata
from scvi.dataset._datasets import synthetic_iid
from scvi.inference import UnsupervisedTrainer
from scvi.models import VAE


class TestVAE(TestCase):
    def test_posterior(self):
        use_cuda = False
        adata = synthetic_iid()
        setup_anndata(
            adata,
            batch_key="batch",
            labels_key="labels",
            protein_expression_obsm_key="protein_expression",
            protein_names_uns_key="protein_names",
        )
        stats = adata.uns["scvi_summary_stats"]
        vae = VAE(stats["n_genes"], stats["n_batch"], stats["n_labels"])
        trainer = UnsupervisedTrainer(vae, adata, train_size=0.5, use_cuda=use_cuda)
        trainer.train(n_epochs=1)
        full = trainer.create_posterior(
            trainer.model, adata, indices=np.arange(stats["n_genes"])
        )
        full = full.update_batch_size(batch_size=32)
        latent, batch_indices, labels = full.sequential().get_latent()
        batch_indices = batch_indices.ravel()

        # imputed_values = full.sequential().imputation()
        # normalized_values = full.sequential().get_sample_scale()

        # cell_types = adata.cell_types
        # print(gene_dataset.cell_types)
        # # oligodendrocytes (#4) VS pyramidal CA1 (#5)
        # couple_celltypes = (4, 5)  # the couple types on which to study DE

        # print(
        #     "\nDifferential Expression A/B for cell types\nA: %s\nB: %s\n"
        #     % tuple((cell_types[couple_celltypes[i]] for i in [0, 1]))
        # )

        # cell_idx1 = gene_dataset.labels.ravel() == couple_celltypes[0]
        # cell_idx2 = gene_dataset.labels.ravel() == couple_celltypes[1]

    def test_totalVI_posterior(self,):
        pass

    def test_scanVI(self,):
        pass
