import torch
from torch import nn as nn
from torch.distributions import kl_divergence as kl, Normal

from scvi.models.modules import DecoderPoisson
from scvi.models.vae import NormalEncoderVAE


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
    def _reconstruction_loss(x, rate):
        rl = -torch.distributions.Poisson(rate).log_prob(x)
        assert rl.dim() == 2
        return torch.sum(rl, dim=-1)

    def scale_from_z(self, sample_batch, fixed_batch):
        raise NotImplementedError

    def inference(self, x, batch_index=None, y=None, n_samples=1):
        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        # Sampling
        qz_m, qz_v, z = self.z_encoder(x_, y)
        ql_m, ql_v, library = self.l_encoder(x_)

        if n_samples > 1:
            assert not self.z_full_cov
            qz_m = qz_m.unsqueeze(0).expand([n_samples] + list(qz_m.size()))
            qz_v = qz_v.unsqueeze(0).expand([n_samples] + list(qz_v.size()))
            ql_m = ql_m.unsqueeze(0).expand([n_samples] + list(ql_m.size()))
            ql_v = ql_v.unsqueeze(0).expand([n_samples] + list(ql_v.size()))
            z = self.z_encoder.sample(qz_m, qz_v)
            library = self.l_encoder.sample(ql_m, ql_v)

        px_rate = self.decoder(z, library, batch_index, y)
        return px_rate, qz_m, qz_v, z, ql_m, ql_v, library

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
        reconst_loss = self._reconstruction_loss(x, px_rate)
        return reconst_loss + kl_divergence_l, kl_divergence

    def ratio_loss(self, x, local_l_mean, local_l_var, batch_index=None, y=None):
        (px_rate, qz_m, qz_v, z, ql_m, ql_v, library) = self.inference(
            x, batch_index, y
        )

        # KL Divergence
        # z_prior_m, z_prior_v = self.get_prior_params(device=qz_m.device)

        log_px_zl = -self._reconstruction_loss(x, px_rate)
        log_pl = (
            Normal(local_l_mean, torch.sqrt(local_l_var)).log_prob(library).sum(dim=-1)
        )

        log_pz = self.log_p_z(z)
        log_qz_x = self.z_encoder.distrib(qz_m, qz_v).log_prob(z)

        # assert log_qz_x.shape == log_pz.shape, (log_qz_x.shape, log_pz.shape)
        if log_pz.dim() == 2:
            log_pz = log_pz.sum(dim=1)
        if log_qz_x.dim() == 2:
            log_qz_x = log_qz_x.sum(dim=1)

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
        neg_elbo = -log_ratio.mean(dim=0)
        return neg_elbo
