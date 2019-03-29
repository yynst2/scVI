# -*- coding: utf-8 -*-
"""Main module."""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence as kl
from scipy.linalg import sqrtm
from scvi.models.modules import Encoder, Decoder
from scvi.models.utils import one_hot

torch.backends.cudnn.benchmark = True


# Linear Gaussian with fixed variance model
class LinearGaussian(nn.Module):
    r"""Variational encoder model. Support full covariances as long as it is not learned.

    :param n_input: Number of input genes
    :param n_hidden: Number of nodes per hidden layer
    :param n_latent: Dimensionality of the latent space
    :param n_layers: Number of hidden layers used for encoder NN
    :param dropout_rate: Dropout rate for neural networks

    """

    def __init__(self, A_param, px_condz_var, n_input: int, n_hidden: int = 128, n_latent: int = 10,
                 n_layers: int = 1, dropout_rate: float = 0.1):
        super().__init__()

        self.n_latent = n_latent
        self.encoder = Encoder(n_input, n_latent, n_layers=n_layers, n_hidden=n_hidden,
                               dropout_rate=dropout_rate)
        self.A = torch.from_numpy(np.array(A_param, dtype=np.float32)).cuda()

        log_det = np.log(np.linalg.det(px_condz_var))
        self.log_det_px_z = torch.tensor(log_det, requires_grad=False, dtype=torch.float).cuda()

        inv_sqrt = sqrtm(np.linalg.inv(px_condz_var))
        self.inv_sqrt_px_z = torch.from_numpy(np.array(inv_sqrt, dtype=np.float32)).cuda()

    def get_latents(self, x):
        r""" returns the result of ``sample_from_posterior`` inside a list

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :return: one element list of tensor
        :rtype: list of :py:class:`torch.Tensor`
        """
        return [self.sample_from_posterior(x)]

    def sample_from_posterior(self, x, give_mean=False):
        r""" samples the tensor of latent values from the posterior
        #doesn't really sample, returns the means of the posterior distribution

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param give_mean: is True when we want the mean of the posterior  distribution rather than sampling
        :return: tensor of shape ``(batch_size, n_latent)``
        :rtype: :py:class:`torch.Tensor`
        """
        qz_m, qz_v, z = self.encoder(x, None)  # y only used in VAEC
        if give_mean:
            z = qz_m
        return z

    def inference(self, x, n_samples=1):
        # Sampling
        qz_m, qz_v, z = self.encoder(x, None)
        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            z = Normal(qz_m, qz_v.sqrt()).sample()

        px_mean = torch.matmul(z, torch.transpose(self.A, 0, 1))

        return px_mean, qz_m, qz_v, z

    def forward(self, x):
        r""" Returns the reconstruction loss and the Kullback divergences

        :param x: tensor of values with shape (batch_size, n_input)
        :return: the reconstruction loss and the Kullback divergences
        :rtype: 2-tuple of :py:class:`torch.FloatTensor`
        """
        # Parameters for z latent distribution

        px_mean, qz_m, qz_v, z = self.inference(x)

        # KL Divergence
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        kl_divergence = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(dim=1)

        reconst_loss = self.log_normal_full(x, px_mean, self.log_det_px_z, self.inv_sqrt_px_z)

        return reconst_loss + kl_divergence

    @staticmethod
    def log_normal_full(x, mean, log_det, inv_sqrt):
        # copying code from NUMPY
        d = x.shape[1]

        log_lik = torch.zeros((1,), dtype=torch.float).cuda()
        log_lik += d * np.log(2 * np.array(np.pi, dtype=np.float32))
        log_lik += log_det
        vec_ = torch.matmul(x - mean, inv_sqrt)
        log_lik += 0.5 * torch.mul(vec_, vec_).sum(dim=-1).mean(dim=-1)
        return -0.5 * log_lik
