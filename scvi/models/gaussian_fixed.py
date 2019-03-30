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

    def __init__(self, A_param, px_condz_var, n_input: int, learn_var: bool = False, n_hidden: int = 128, n_latent: int = 10,
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

        self.learn_var = learn_var
        self.px_log_diag_var = torch.nn.Parameter(torch.randn(1, n_input))

    def get_std(self):
        return torch.sqrt(torch.exp(self.px_log_diag_var))

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

        return px_mean, torch.exp(self.px_log_diag_var), qz_m, qz_v, z

    def log_ratio(self, x, px_mean, px_var, qz_m, qz_v, z, return_full=False):
        log_px_given_z = self.reconstruction(x.repeat(px_mean.shape[0], 1),
                                             px_mean.view((-1, x.shape[-1])),
                                             px_var
                                             ).view((px_mean.shape[0], -1))

        log_pz = Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v)).log_prob(z).sum(dim=-1)
        log_qz_given_x = Normal(qz_m, qz_v.sqrt()).log_prob(z).sum(dim=-1)
        log_ratio = log_px_given_z + log_pz - log_qz_given_x
        if return_full:
            return log_ratio, log_px_given_z, log_pz, log_qz_given_x
        else:
            return log_ratio

    def neg_iwelbo(self, x, n_samples_mc):
        # tile vectors from (B, d) to (n_samples_mc, B, d)
        px_mean, px_var, qz_m, qz_v, z = self.inference(x, n_samples=n_samples_mc)
        log_ratio = self.log_ratio(x, px_mean, px_var, qz_m, qz_v, z)

        iwelbo = torch.logsumexp(log_ratio, dim=0) - np.log(n_samples_mc)
        return -iwelbo

    def cubo(self, x, n_samples_mc):
        # computes the naive cubo from chi2 upper bound
        # tile vectors from (B, d) to (n_samples_mc, B, d)
        px_mean, px_var, qz_m, qz_v, z = self.inference(x, n_samples=n_samples_mc)
        log_ratio = self.log_ratio(x, px_mean, px_var, qz_m, qz_v, z)

        cubo = torch.logsumexp(2 * log_ratio, dim=0) - np.log(n_samples_mc)
        return 0.5 * cubo

    def iwrevkl_obj(self, x, n_samples_mc):
        # computes the importance sampled objective for reverse KL EP (revisited reweighted wake-sleep)
        # tile vectors from (B, d) to (n_samples_mc, B, d)
        px_mean, px_var, qz_m, qz_v, z = self.inference(x, n_samples=n_samples_mc)
        log_ratio, _, _, log_qz_given_x = self.log_ratio(x, px_mean, px_var, qz_m, qz_v, z, return_full=True)

        rev_kl = torch.nn.softmax(log_ratio, dim=0) * (-1) * log_qz_given_x
        return rev_kl.sum(dim=0)

    def vr_max(self, x, n_samples_mc):
        # computes the naive MC from VR-max bound
        # tile vectors from (B, d) to (n_samples_mc, B, d)
        px_mean, px_var, qz_m, qz_v, z = self.inference(x, n_samples=n_samples_mc)
        log_ratio = self.log_ratio(x, px_mean, px_var, qz_m, qz_v, z)

        return log_ratio.max(dim=0)[0]

    def neg_elbo(self, x):
        px_mean, px_var, qz_m, qz_v, z = self.inference(x)

        # KL Divergence
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        kl_divergence = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(dim=1)
        log_px_given_z = self.reconstruction(x, px_mean, px_var)
        # minimize the neg_elbo
        return - log_px_given_z + kl_divergence

    def forward(self, x, param):
        if param == "ELBO":
            return self.neg_elbo(x)
        if param == "CUBO":
            return self.cubo(x, n_samples_mc=30)

    def reconstruction(self, x, px_mean, px_var):
        if self.learn_var:
            return Normal(px_mean, torch.sqrt(px_var)).log_prob(x).sum(dim=1)
        else:
            return self.log_normal_full(x, px_mean, self.log_det_px_z, self.inv_sqrt_px_z)

    @staticmethod
    def log_normal_full(x, mean, log_det, inv_sqrt):
        # copying code from NUMPY
        d = x.shape[1]

        log_lik = torch.zeros((x.shape[0],), dtype=torch.float).cuda()
        log_lik += d * np.log(2 * np.array(np.pi, dtype=np.float32))
        log_lik += log_det
        vec_ = torch.matmul(x - mean, inv_sqrt)
        log_lik += torch.mul(vec_, vec_).sum(dim=-1)
        return -0.5 * log_lik
