from unittest import TestCase
import pdb
import numpy as np
from scvi.dataset import setup_anndata
from scvi.dataset._datasets import synthetic_iid
from scvi.inference import UnsupervisedTrainer
from scvi.inference import TotalTrainer, TotalPosterior
from scvi.models import VAE, TOTALVI


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
            trainer.model, adata, indices=np.arange(stats["n_cells"])
        )
        full = full.update_batch_size(batch_size=32)
        latent, batch_indices, labels = full.sequential().get_latent()
        batch_indices = batch_indices.ravel()
        imputed_values = full.imputation()
        assert imputed_values.shape == (400, 100)
        normalized_values = full.get_sample_scale()
        assert normalized_values.shape == (400, 100)
        adata.layers["scVI_normalized"] = normalized_values
        rand_cell_types = adata.obs.batch.values
        cell_idx1 = rand_cell_types == 1
        cell_idx2 = rand_cell_types == 0
        de_res = full.differential_expression_score(cell_idx1, cell_idx2)

    def test_totalVI_posterior(self,):
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
        totalvae = TOTALVI(
            stats["n_genes"], stats["n_proteins"], n_batch=stats["n_batch"]
        )
        use_cuda = False
        # totalVI is trained on 90% of the data
        # Early stopping does not comply with our automatic notebook testing so we disable it when testing
        trainer = TotalTrainer(
            totalvae,
            adata,
            train_size=0.90,
            test_size=0.10,
            use_cuda=use_cuda,
            frequency=1,
            batch_size=256,
        )
        trainer.train(n_epochs=1)

        full_posterior = trainer.create_posterior(type_class=TotalPosterior)
        # fix this
        full_posterior.sampler_kwargs.update({"batch_size": 32})

        # extract latent space
        latent_mean, batch_index, label, library_gene = full_posterior.get_latent()

        # Number of Monte Carlo samples to average over
        n_samples = 25
        # Probability of background for each (cell, protein)
        py_mixing = full_posterior.get_sample_mixing(
            n_samples=n_samples, give_mean=True, transform_batch=[0, 1]
        )
        parsed_protein_names = [
            p.split("_")[0] for p in adata.obsm["protein_expression"].columns
        ]

        (
            denoised_genes,
            denoised_proteins,
        ) = full_posterior.get_normalized_denoised_expression(
            n_samples=n_samples, give_mean=True, transform_batch=[0, 1]
        )

    def test_scanVI(self,):
        pass
