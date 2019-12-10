# -*- coding: utf-8 -*-
"""Main module."""

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence as kl

from scvi.models.log_likelihood import log_zinb_positive, log_nb_positive
from scvi.models.modules import Encoder, DecoderSCVI

torch.backends.cudnn.benchmark = True


# CVAE model
class CVAE(nn.Module):
    r"""Conditional Variational auto-encoder model, which takes into account cell to cell correlations

    :param n_input: Number of input genes
    :param n_batch: Number of batches
    :param n_labels: Number of labels
    :param n_hidden: Number of nodes per hidden layer
    :param n_latent: Dimensionality of the latent space
    :param n_layers: Number of hidden layers used for encoder and decoder NNs
    :param dropout_rate: Dropout rate for neural networks

    :param log_variational: Log(data+1) prior to encoding for numerical stability. Not normalization.
    :param reconstruction_loss:  One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution

    Examples:
        >>> gene_dataset = CortexDataset()
        >>> vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=gene_dataset.n_labels)

    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        log_variational: bool = True,
        reconstruction_loss: str = "zinb",
        tau: float = 0.5,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.reconstruction_loss = reconstruction_loss
        # Automatically deactivate if useless
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.tau = tau
        self.px_r = torch.nn.Parameter(torch.randn(n_input))

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        # decoder goes from n_latent-dimensional space to n_input-d data
        self.decoder = DecoderSCVI(
            n_latent,
            n_input,
            n_cat_list=[n_batch],
            n_layers=n_layers,
            n_hidden=n_hidden,
        )

    def get_latent(self, x, y=None):
        r""" returns the result of ``sample_from_posterior_z`` inside a list

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :return: one element list of tensor
        :rtype: list of :py:class:`torch.Tensor`
        """
        return [self.sample_from_posterior_z(x, y)]

    def sample_from_posterior_z(self, x, y=None, give_mean=False):
        r""" samples the tensor of latent values from the posterior
        #doesn't really sample, returns the means of the posterior distribution

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param give_mean: is True when we want the mean of the posterior  distribution rather than sampling
        :return: tensor of shape ``(batch_size, n_latent)``
        :rtype: :py:class:`torch.Tensor`
        """
        if self.log_variational:
            x = torch.log(1 + x)
        qz_m, qz_v, z = self.z_encoder(x, y)  # y only used in VAEC
        if give_mean:
            z = qz_m
        return z

    def get_sample_scale(self, x, batch_index=None, y=None, n_samples=1):
        r"""Returns the tensor of predicted frequencies of expression

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param n_samples: number of samples
        :return: tensor of predicted frequencies of expression with shape ``(batch_size, n_input)``
        :rtype: :py:class:`torch.Tensor`
        """
        return self.inference(x, batch_index=batch_index, y=y, n_samples=n_samples)[
            "px_scale"
        ]

    def get_sample_rate(self, x, batch_index=None, y=None, n_samples=1):
        r"""Returns the tensor of means of the negative binomial distribution

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param n_samples: number of samples
        :return: tensor of means of the negative binomial distribution with shape ``(batch_size, n_input)``
        :rtype: :py:class:`torch.Tensor`
        """
        return self.inference(x, batch_index=batch_index, y=y, n_samples=n_samples)[
            "px_rate"
        ]

    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout, **kwargs):
        # Reconstruction Loss
        if self.reconstruction_loss == "zinb":
            reconst_loss = -log_zinb_positive(x, px_rate, px_r, px_dropout).sum(dim=-1)
        elif self.reconstruction_loss == "nb":
            reconst_loss = -log_nb_positive(x, px_rate, px_r).sum(dim=-1)
        return reconst_loss

    def scale_from_z(self, sample_batch, fixed_batch):
        if self.log_variational:
            sample_batch = torch.log(1 + sample_batch)
        qz_m, qz_v, z = self.z_encoder(sample_batch)
        batch_index = fixed_batch * torch.ones_like(sample_batch[:, [0]])
        library = 4.0 * torch.ones_like(sample_batch[:, [0]])
        px_scale, _, _, _ = self.decoder("gene", z, library, batch_index)
        return px_scale

    def inference(self, x, x1=None, x2=None, batch_index=None, y=None, n_samples=1):

        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        # Sampling
        qz_m, qz_v, z = self.z_encoder(x_, y)
        library = torch.log(x_.sum(1)).unsqueeze(1)

        if x1 is not None:
            x1_ = x1
            x2_ = x2
            if self.log_variational:
                x1_ = torch.log(1 + x1_)
                x2_ = torch.log(1 + x2_)

            qz1_m, qz1_v, z1 = self.z_encoder(x1_, y)
            qz2_m, qz2_v, z2 = self.z_encoder(x2_, y)

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            z = Normal(qz_m, qz_v.sqrt()).sample()

        px_scale, _, px_rate, px_dropout = self.decoder(
            "gene", z, library, batch_index, y
        )
        px_r = torch.exp(self.px_r)

        res = dict(
            px_scale=px_scale,
            px_r=px_r,
            px_rate=px_rate,
            px_dropout=px_dropout,
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            library=library,
        )
        if x1 is not None:
            res["qz1_m"] = qz1_m
            res["qz1_v"] = qz1_v
            res["qz2_m"] = qz2_m
            res["qz2_v"] = qz2_v
        return res

    def elbo(self, x):
        outputs = self.inference(x)

        # first derive the iid part of the ELBO
        qz_m = outputs["qz_m"]
        qz_v = outputs["qz_v"]
        px_rate = outputs["px_rate"]
        px_r = outputs["px_r"]
        px_dropout = outputs["px_dropout"]
        kl_divergence_z = kl(
            Normal(qz_m, torch.sqrt(qz_v)),
            Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v)),
        ).sum(dim=1)
        reconst_loss = self.get_reconstruction_loss(x, px_rate, px_r, px_dropout)
        return reconst_loss + kl_divergence_z

    def forward(self, x, x1, x2, w, y=None):
        r""" Returns the reconstruction loss and the Kullback divergences

        :param x: tensor of values with shape (batch_size1, n_input)
        :param x1: tensor of values with shape (batch_size2, n_input)
        :param x2: tensor of values with shape (batch_size2, n_input)
        :param w: tensor of values with shape (n_input)
        :return: the reconstruction loss and the Kullback divergences
        :rtype: 2-tuple of :py:class:`torch.FloatTensor`
        """
        # Parameters for z latent distribution
        outputs = self.inference(x, x1, x2, y)

        # first derive the iid part of the ELBO
        qz_m = outputs["qz_m"]
        qz_v = outputs["qz_v"]
        px_rate = outputs["px_rate"]
        px_r = outputs["px_r"]
        px_dropout = outputs["px_dropout"]

        kl_divergence_z = 0.5 * torch.sum(qz_m ** 2 + qz_v - torch.log(qz_v) - 1, dim=1)
        reconst_loss = self.get_reconstruction_loss(x, px_rate, px_r, px_dropout)

        # second get the correlated part
        qz1_m = outputs["qz1_m"]
        qz1_v = outputs["qz1_v"]
        qz2_m = outputs["qz2_m"]
        qz2_v = outputs["qz2_v"]

        kl_correlation = (
            0.5
            * w
            * torch.sum(
                (qz1_m ** 2 + qz2_m ** 2 - 2 * qz1_m * qz2_m * self.tau + qz1_v + qz2_v)
                / (1.0 - self.tau ** 2)
                - (qz1_m ** 2 + qz2_m ** 2 + qz1_v + qz2_v)
                + torch.log(1.0 - torch.ones_like(qz_m[0, 0]) * self.tau ** 2),
                1,
            )
        )

        return reconst_loss, kl_divergence_z, kl_correlation
