import logging

import numpy as np
import torch
from scvi.inference import Posterior
from scvi.inference import Trainer
from scvi.models import CVAE
from torch.utils.data import DataLoader

from scvi.dataset import SpatialDataset

logger = logging.getLogger(__name__)


class SpatialPosterior(Posterior):
    def __init__(self, model: CVAE, gene_dataset: SpatialDataset, *args, **kwargs):
        if "data_loader_kwargs" not in kwargs:
            kwargs["data_loader_kwargs"] = {"batch_size": 1024}
        super().__init__(model, gene_dataset, *args, **kwargs)
        self.data_loader_kwargs.update(
            {
                "collate_fn": gene_dataset.collate_fn_builder(
                    dict(
                        [
                            ("X", np.float32),
                            ("edge_indices", np.int64),
                            ("neighbor_scrna", np.float32),
                            ("edge_weights", np.float32),
                            ("batch_indices", np.int64),
                            ("labels", np.int64),
                        ]
                    ),
                    override=True,
                )
            }
        )
        self.data_loader = DataLoader(gene_dataset, **self.data_loader_kwargs)

    @torch.no_grad()
    def elbo(self):
        elbo = []
        for tensors in self:
            (
                sample_batch,
                edge_indices,
                neighbor_scrna,
                edge_weights,
                _,
                labels,
            ) = tensors
            elbo += [self.model.elbo(sample_batch).cpu()]
        return np.mean(np.array(torch.cat(elbo)))

    @torch.no_grad()
    def total_elbo(self):
        """
        computes the whole loss
        """
        elbo = []
        for tensors in self:
            (
                sample_batch,
                edge_indices,
                neighbor_scrna,
                edge_weights,
                _,
                labels,
            ) = tensors
            elbo += [self.model.elbo(sample_batch).cpu()]
        return np.mean(np.array(torch.cat(elbo)))

    @torch.no_grad()
    def get_latent(self, sample=False):
        latent = []
        for tensors in self:
            (
                sample_batch,
                edge_indices,
                neighbor_scrna,
                edge_weights,
                _,
                labels,
            ) = tensors
            latent += [self.model.sample_from_posterior_z(sample_batch).cpu()]
        return np.array(torch.cat(latent))

    @torch.no_grad()
    def get_sample_scale(self):
        px_scales = []
        for tensors in self:
            (
                sample_batch,
                local_l_mean,
                local_l_var,
                neighbors,
                positions,
                batch_index,
                labels,
            ) = tensors
            px_scales += [
                np.array(
                    (
                        self.model.get_sample_scale(
                            sample_batch,
                            neighbors=neighbors,
                            positions=positions,
                            batch_index=batch_index,
                            y=labels,
                            n_samples=1,
                        )
                    ).cpu()
                )
            ]
        return np.concatenate(px_scales)


class SpatialUnsupervisedTrainer(Trainer):
    default_metrics_to_monitor = ["elbo"]

    def __init__(
        self,
        model,
        gene_dataset,
        cut_minibatch=0.1,
        train_size=1.0,
        test_size=None,
        n_samples=1,
        n_edges=1,
        **kwargs
    ):
        """
        :param model: model to be used
        :param cut_minibatch: the total batch size has to cut between iid and correlated part
        """
        super().__init__(model, gene_dataset, **kwargs)
        self.cut_minibatch = cut_minibatch
        if type(self) is SpatialUnsupervisedTrainer:
            (
                self.train_set,
                self.test_set,
                self.validation_set,
            ) = self.train_test_validation(
                model, gene_dataset, train_size, test_size, type_class=SpatialPosterior
            )
            self.train_set.to_monitor = ["reconstruction_error"]
            self.test_set.to_monitor = ["reconstruction_error"]
            self.validation_set.to_monitor = ["reconstruction_error"]
            self.n_samples = n_samples
            self.n_edges = n_edges

    @property
    def posteriors_loop(self):
        return ["train_set"]

    def loss(self, tensors):
        # get data with neighbors
        (
            sample_batch,
            edge_indices,
            neighbor_scrna,
            edge_weight,
            batch_index,
            labels,
        ) = tensors
        # first get minibatch for iid part of the ELBO
        T = int(sample_batch.shape[0] * self.cut_minibatch)
        x = sample_batch[:T]

        # second, get minibatches for the correlated part
        x1 = sample_batch[T:]
        neighbor_scrna = neighbor_scrna[T:]
        edge_weight = edge_weight[T:]

        # get a random neighbor for each cell in minibatch
        uniform = torch.distributions.one_hot_categorical.OneHotCategorical(
            torch.ones_like(neighbor_scrna[0, :, 0])
        )
        one_hot = uniform.sample((neighbor_scrna.shape[0],))

        # now select only that random neighbor
        x2 = torch.sum(neighbor_scrna * one_hot.unsqueeze(2), 1)
        w = torch.sum(edge_weight * one_hot, 1)
        # print(w)
        # finally get the loss
        reconstruction_loss, kl_z, graph_corr = self.model(x, x1, x2, w)
        loss = torch.mean(
            reconstruction_loss + kl_z
        ) + self.n_edges / self.n_samples * torch.mean(graph_corr)
        print(
            loss.item(),
            (
                torch.mean(kl_z)
                + self.n_edges / self.n_samples * torch.mean(graph_corr)
            ).item(),
        )
        return loss

    def create_posterior(
        self,
        model=None,
        gene_dataset=None,
        shuffle=False,
        indices=None,
        type_class=SpatialPosterior,
    ):
        return super().create_posterior(
            model=model,
            gene_dataset=gene_dataset,
            shuffle=shuffle,
            indices=indices,
            type_class=type_class,
        )
