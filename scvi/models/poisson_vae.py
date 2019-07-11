import torch
from torch import nn as nn
from torch.distributions import kl_divergence as kl, Normal
import numpy as np
from tqdm import tqdm

from scvi.models.modules import DecoderPoisson
from scvi.models.vae import NormalEncoderVAE


# TODO: Refactor Log Ratio
# TODO: BEWARE OF log transformation

class LogNormalPoissonVAE(NormalEncoderVAE):
    """Variational auto-encoder model for LogPoisson latent- Poisson gene expressions.

    :param n_input: Number of input genes
    :param n_batch: Number of batches
    :param n_labels: Number of labels
    :param n_hidden: Number of nodes per hidden layer
    :param n_latent: Dimensionality of the latent space
    :param n_layers: Number of hidden layers used for encoder and decoder NNs
    :param dropout_rate: Dropout rate for neural networks

    :param log_variational: Log variational distribution
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
        full_cov=False,
        autoregressive=False,
        gt_decoder: nn.Module = None,
        learn_prior_scale: bool = False
    ):
        self.trained_decoder = gt_decoder is None
        if self.trained_decoder:
            log_p_z = None
        else:
            log_p_z = gt_decoder.log_p_z
        super().__init__(
            n_input=n_input,
            n_hidden=n_hidden,
            n_latent=n_latent,
            dropout_rate=dropout_rate,
            log_variational=log_variational,
            full_cov=full_cov,
            autoregresssive=autoregressive,
            log_p_z=log_p_z,
            learn_prior_scale=learn_prior_scale
        )

        # decoder goes from n_latent-dimensional space to n_input-d data
        if self.trained_decoder:
            self.decoder = DecoderPoisson(
                n_latent,
                n_input,
                n_cat_list=[n_batch],
                n_layers=n_layers,
                n_hidden=n_hidden,
            )
        else:
            self.decoder = gt_decoder

        self.n_latent = n_latent
        self.log_variational = log_variational
        # Automatically deactivate if useless
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_latent_layers = 1  # not sure what this is for, no usages?

    def get_sample_scale(self, x, batch_index=None, y=None, n_samples=1):
        raise NotImplementedError

    def get_sample_rate(self, x, batch_index=None, y=None, n_samples=1):
        raise NotImplementedError

    def get_log_ratio(self, x, batch_index=None, y=None, n_samples=1):
        r"""Returns the tensor of log_pz + log_px_z - log_qz_x

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param n_samples: number of samples
        :return: tensor of predicted frequencies of expression with shape ``(batch_size, n_input)``
        :rtype: :py:class:`torch.Tensor`
        """
        # TODO: Implement
        raise NotImplementedError

    @staticmethod
    def get_reconstruction_loss(x, rate):
        rl = -torch.distributions.Poisson(rate).log_prob(x)
        assert rl.dim() == rate.dim()  # rl should be (n_batch, n_input)
        # or (n_samples, n_batch, n_input)
        return torch.sum(rl, dim=-1)

    def scale_from_z(self, sample_batch, fixed_batch):
        raise NotImplementedError

    def forward(self, x, local_l_mean, local_l_var, batch_index=None, y=None):
        r""" Returns the reconstruction loss and the Kullback divergences

        :param x: tensor of values with shape (batch_size, n_input)
        :param local_l_mean: tensor of means of the prior distribution of latent variable l
         with shape (batch_size, 1)
        :param local_l_var: tensor of variancess of the prior distribution of latent variable l
         with shape (batch_size, 1)
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param y: tensor of cell-types labels with shape (batch_size, n_labels)
        :return: the reconstruction loss and the Kullback divergences
        :rtype: 2-tuple of :py:class:`torch.FloatTensor`
        """
        # assert self.trained_decoder, "If you train the encoder alone please use the `ratio_loss`" \
        #                              "In `forward`, the KL terms are wrong"

        px_rate, qz_m, qz_v, z, ql_m, ql_v, library = self.inference(x, batch_index, y)

        # KL Divergence
        mean, scale = self.get_prior_params(device=qz_m.device)
        kl_divergence_z = kl(
            self.z_encoder.distrib(qz_m, qz_v), self.z_encoder.distrib(mean, scale)
        )
        if len(kl_divergence_z.size()) == 2:
            kl_divergence_z = kl_divergence_z.sum(dim=1)
        kl_divergence_l = kl(
            Normal(ql_m, torch.sqrt(ql_v)),
            Normal(local_l_mean, torch.sqrt(local_l_var)),
        ).sum(dim=1)
        kl_divergence = kl_divergence_z
        reconst_loss = self.get_reconstruction_loss(x, px_rate)
        return reconst_loss + kl_divergence_l, kl_divergence

    def inference(self, x, batch_index=None, y=None, n_samples=1):
        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        # Sampling
        qz_m, qz_v, z = self.z_encoder(x_, y)
        ql_m, ql_v, library = self.l_encoder(x_)

        n_batches, n_dim = qz_m.shape
        if n_samples > 1:
            z = self.z_encoder.reparameterize(qz_m, qz_v, sample_size=(n_samples,))
            library = self.l_encoder.reparameterize(ql_m, ql_v, sample_size=(n_samples,))

            assert z.shape == (n_samples, n_batches, n_dim)
            px_rate = torch.zeros(size=(n_samples, n_batches, self.n_input), device=z.device)
            for sample_id in range(n_samples):
                px_rate[sample_id, :] = self.decoder(
                    z[sample_id, :],
                    library[sample_id, :],
                    batch_index,
                    y
                )

        else:
            px_rate = self.decoder(z, library, batch_index, y)
        return px_rate, qz_m, qz_v, z, ql_m, ql_v, library

    def log_ratio(self, x, z, library=None, batch_index=None, y=None):
        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        qz_m, qz_v, _ = self.z_encoder(x_, y)
        ql_m, ql_v, library = self.l_encoder(x_)

        px_rate = self.decoder(z, library, batch_index, y)
        # TODO: Refactor compute_log_ratio
        log_px_zl = -self.get_reconstruction_loss(x, px_rate)
        log_pz = self.log_p_z(z)
        # print("qz_m", qz_m)
        # print("qz_v", qz_v)
        log_qz_x = self.z_encoder.distrib(qz_m, qz_v).log_prob(z)  # .sum(dim=-1)

        # print('log_px_zl', log_px_zl)
        # print('log_pz', log_pz)
        # print('log_qz_x', log_qz_x)
        if log_pz.dim() == 2:
            log_pz = log_pz.sum(-1)
            # raise ValueError
        if log_qz_x.dim() == 2:
            log_qz_x = log_qz_x.sum(-1)
            # raise ValueError
        assert (
            log_px_zl.shape
            == log_pz.shape
            == log_qz_x.shape
        ), (log_px_zl.shape, log_pz.shape, log_qz_x.shape)
        # log_ratio = log_px_zl + log_pz + log_pl - log_qz_x - log_ql_x
        log_ratio = log_px_zl + log_pz - log_qz_x
        return log_ratio

    def compute_log_ratio(self, x, local_l_mean, local_l_var, batch_index=None, y=None, n_samples=1):
        """
        Computes log p(x, latents) / q(latents | x) for each element in x
        :param x:
        :param local_l_mean:
        :param local_l_var:
        :param batch_index:
        :param y:
        :param n_samples:
        :return:
        """
        (px_rate, qz_m, qz_v, z, ql_m, ql_v, library) = self.inference(
            x, batch_index, y, n_samples=n_samples
        )

        # KL Divergence
        # z_prior_m, z_prior_v = self.get_prior_params(device=qz_m.device)

        log_px_zl = -self.get_reconstruction_loss(x, px_rate)
        log_pl = (
            Normal(local_l_mean, torch.sqrt(local_l_var)).log_prob(library).sum(dim=-1)
        )

        log_pz = self.log_p_z(z)
        log_qz_x = self.z_encoder.distrib(qz_m, qz_v).log_prob(z)  # .sum(dim=-1)

        # assert log_qz_x.shape == log_pz.shape, (log_qz_x.shape, log_pz.shape)

        # TODO: Find cleaner and safer way
        # Below should not be useful now
        if log_pz.dim() == 2 and n_samples == 1:
            log_pz = log_pz.sum(-1)
            # raise ValueError
        if log_qz_x.dim() == 2 and n_samples == 1:
            log_qz_x = log_qz_x.sum(-1)
            # raise ValueError
        if log_qz_x.dim() == 3 and n_samples > 1:
            log_qz_x = log_qz_x.sum(-1)
            # raise ValueError
        if log_pz.dim() == 3 and n_samples > 1:
            log_pz = log_pz.sum(-1)

        log_ql_x = Normal(ql_m, torch.sqrt(ql_v)).log_prob(library).sum(dim=-1)

        assert (
            log_px_zl.shape
            == log_pl.shape
            == log_pz.shape
            == log_qz_x.shape
            == log_ql_x.shape
        ), (log_px_zl.shape, log_pl.shape, log_pz.shape, log_qz_x.shape, log_ql_x.shape)
        # log_ratio = log_px_zl + log_pz + log_pl - log_qz_x - log_ql_x
        log_ratio = log_px_zl + log_pz - log_qz_x
        return log_ratio

    def ratio_loss(self, x, local_l_mean, local_l_var, batch_index=None, y=None):
        """
        Compute estimate of E_q log(p(x, latents) / q(latents|x)) using Variationnal EM

        :param x:
        :param local_l_mean:
        :param local_l_var:
        :param batch_index:
        :param y:
        :return:
        """
        log_ratios = self.compute_log_ratio(x, local_l_mean, local_l_var, batch_index=None, y=None)
        neg_elbo = -log_ratios.mean(dim=0)
        return neg_elbo

    @torch.no_grad()
    def marginal_ll(self, posterior, n_samples_mc=100):
        """
        Computes estimate of marginal log likelihood E_q(z) [log ratio]

        :param posterior:
        :param n_samples_mc:
        :return:
        """
        log_lkl = 0.0
        for i_batch, tensors in enumerate(tqdm(posterior)):
            x, local_l_mean, local_l_var, batch_index, labels = tensors
            n_batches = x.shape[0]
            log_ratios = self.compute_log_ratio(
                x,
                local_l_mean,
                local_l_var,
                batch_index=batch_index,
                y=labels,
                n_samples=n_samples_mc
            )
            assert log_ratios.shape == (n_samples_mc, n_batches)
            batch_log_lkl = torch.logsumexp(log_ratios, dim=0) - np.log(n_samples_mc)
            log_lkl += torch.sum(batch_log_lkl).item()
        return log_lkl

