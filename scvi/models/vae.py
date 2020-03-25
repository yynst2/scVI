# -*- coding: utf-8 -*-
"""Main module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence as kl

from scvi.models.log_likelihood import log_zinb_positive, log_nb_positive
from scvi.models.modules import Encoder, DecoderSCVI, LinearDecoderSCVI
from scvi.models.utils import one_hot

torch.backends.cudnn.benchmark = True


class NormalEncoderVAE(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        log_variational: bool = True,
        full_cov: bool = False,
        autoregresssive: bool = False,
        log_p_z=None,
        learn_prior_scale: bool = False,
    ):
        """
        Serves as model class for any VAE with Gaussian latent variables for scVI

        :param n_input:
        :param n_hidden:
        :param n_latent:
        :param n_layers:
        :param dropout_rate:
        :param log_variational:
        :param full_cov: Train full posterior cov matrices for variational posteriors
        :param autoregresssive: Train posterior cov matrices using Inverse Autoregressive Flow
        :param log_p_z: Give value of log_p_z (useful if you have a ground truth decoder)
        :param learn_prior_scale: Bool: Should a scalar scaling the prior covariance be learned

        """
        super().__init__()
        self.log_p_z_fixed = log_p_z
        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        self.z_full_cov = full_cov
        self.z_autoregressive = autoregresssive
        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            full_cov=full_cov,
            autoregressive=autoregresssive,
        )
        self.n_input = n_input
        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(
            n_input,
            1,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            prevent_saturation=True,
        )
        # decoder goes from n_latent-dimensional space to n_input-d data
        self.decoder = None
        self.log_variational = log_variational

        if learn_prior_scale:
            self.prior_scale = nn.Parameter(torch.FloatTensor([4.0]))
        else:
            self.prior_scale = 1.0

    def forward(self, *input):
        pass

    def log_p_z(self, z: torch.Tensor):
        if self.log_p_z_fixed is not None:
            return self.log_p_z_fixed(z)
        else:
            z_prior_m, z_prior_v = self.get_prior_params(device=z.device)
            return self.z_encoder.distrib(z_prior_m, z_prior_v).log_prob(z)

    def ratio_loss(self, x, local_l_mean, local_l_var, batch_index, y, return_mean):
        pass

    def iwelbo(
        self,
        x,
        local_l_mean,
        local_l_var,
        batch_index=None,
        y=None,
        k=3,
        single_backward=False,
    ):
        n_batch = len(x)
        log_ratios = torch.zeros(k, n_batch, device="cuda", dtype=torch.float)
        for it in range(k):
            log_ratios[it, :] = self.ratio_loss(
                x,
                local_l_mean,
                local_l_var,
                batch_index=batch_index,
                y=y,
                return_mean=False,
            )

        normalizers, _ = log_ratios.max(dim=0)
        # w_tilde = torch.softmax(log_ratios - normalizers, dim=0).detach()
        w_tilde = (log_ratios - torch.logsumexp(log_ratios, dim=0)).exp().detach()
        if not single_backward:
            loss = -(w_tilde * log_ratios).sum(dim=0)
        else:
            selected_k = torch.distributions.Categorical(
                probs=w_tilde.transpose(-1, -2)
            ).sample()
            assert len(selected_k) == n_batch

            loss = -log_ratios[selected_k, torch.arange(n_batch)]
            # selected_k = selected_k.view(1, -1)
            # mask = torch.zeros_like(log_ratios).scatter(0, selected_k, 1.0).type(torch.ByteTensor)
            # # loss = - (mask * log_ratios).sum(dim=0)
            # loss = - log_ratios[mask]
        # dummy = loss.mean(dim=0)
        # if torch.isnan(dummy):
        #     print('TOTOTOT')
        return loss.mean(dim=0)

    @property
    def encoder_params(self):
        """
        :return: List of learnable encoder parameters (to feed to torch.optim object
        for instance
        """
        return self.get_list_params(
            self.z_encoder.parameters(), self.l_encoder.parameters()
        )

    @property
    def decoder_params(self):
        """
        :return: List of learnable decoder parameters (to feed to torch.optim object
        for instance
        """
        return self.get_list_params(self.decoder.parameters()) + [self.px_r]

    def get_latents(self, x, y=None):
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

    def sample_from_posterior_l(self, x):
        r""" samples the tensor of library sizes from the posterior
        #doesn't really sample, returns the tensor of the means of the posterior distribution

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :return: tensor of shape ``(batch_size, 1)``
        :rtype: :py:class:`torch.Tensor`
        """
        if self.log_variational:
            x = torch.log(1 + x)
        ql_m, ql_v, library = self.l_encoder(x)
        return library

    def get_prior_params(self, device):
        mean = torch.zeros((self.n_latent,), device=device)
        if self.z_full_cov or self.z_autoregressive:
            scale = self.prior_scale * torch.eye(self.n_latent, device=device)
        else:
            scale = self.prior_scale * torch.ones((self.n_latent,), device=device)
        return mean, scale

    @staticmethod
    def get_list_params(*params):
        res = []
        for param_li in params:
            res += list(filter(lambda p: p.requires_grad, param_li))
        return res


# VAE model
class VAE(NormalEncoderVAE):
    r"""Variational auto-encoder model.

    :param n_input: Number of input genes
    :param n_batch: Number of batches
    :param n_labels: Number of labels
    :param n_hidden: Number of nodes per hidden layer
    :param n_latent: Dimensionality of the latent space
    :param n_layers: Number of hidden layers used for encoder and decoder NNs
    :param dropout_rate: Dropout rate for neural networks
    :param dispersion: One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell

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
        dispersion: str = "gene",
        log_variational: bool = True,
        reconstruction_loss: str = "zinb",
        full_cov: bool = False,
        autoregresssive: bool = False,
        log_p_z=None,
        n_blocks=0,
        decoder_do_last_skip=False,
    ):
        super().__init__(
            n_input=n_input,
            n_hidden=n_hidden,
            n_latent=n_latent,
            dropout_rate=dropout_rate,
            log_variational=log_variational,
            full_cov=full_cov,
            autoregresssive=autoregresssive,
            log_p_z=log_p_z,
        )
        self.decoder = DecoderSCVI(
            n_latent,
            n_input,
            n_cat_list=[n_batch],
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_blocks=n_blocks,
            do_last_skip=decoder_do_last_skip,
        )
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.reconstruction_loss = reconstruction_loss
        # Automatically deactivate if useless
        self.n_batch = n_batch
        self.n_labels = n_labels

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input,))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        else:  # gene-cell
            pass

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

    def get_log_ratio(self, x, batch_index=None, y=None, n_samples=1):
        r"""Returns the tensor of log_pz + log_px_z - log_qz_x

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param n_samples: number of samples
        :return: tensor of predicted frequencies of expression with shape ``(batch_size, n_input)``
        :rtype: :py:class:`torch.Tensor`
        """
        (
            px_scale,
            px_r,
            px_rate,
            px_dropout,
            qz_m,
            qz_v,
            z,
            ql_m,
            ql_v,
            library,
        ) = self.inference(x, batch_index=batch_index, y=y, n_samples=n_samples)

        log_px_z = self.get_reconstruction_loss(x, px_rate, px_r, px_dropout)
        log_pz = (
            Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v))
            .log_prob(z)
            .sum(dim=-1)
        )
        log_qz_x = Normal(qz_m, qz_v.sqrt()).log_prob(z).sum(dim=-1)

        return log_pz + log_px_z - log_qz_x

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

    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout):
        # Reconstruction Loss
        if self.reconstruction_loss == "zinb":
            reconst_loss = -log_zinb_positive(x, px_rate, px_r, px_dropout)
        elif self.reconstruction_loss == "nb":
            reconst_loss = -log_nb_positive(x, px_rate, px_r)
        return reconst_loss

    def scale_from_z(self, sample_batch, fixed_batch):
        if self.log_variational:
            sample_batch = torch.log(1 + sample_batch)
        qz_m, qz_v, z = self.z_encoder(sample_batch)
        batch_index = fixed_batch * torch.ones_like(sample_batch[:, [0]])
        library = 4.0 * torch.ones_like(sample_batch[:, [0]])
        px_scale, _, _, _ = self.decoder("gene", z, library, batch_index)
        return px_scale

    def inference(self, x, batch_index=None, y=None, n_samples=1, train_library=True):
        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        # Sampling
        qz_m, qz_v, z = self.z_encoder(x_, y)
        ql_m, ql_v, library = self.l_encoder(x_)

        if n_samples > 1:
            assert not self.z_full_cov
            # TODO: Check no issues when full cov
            qz_m = qz_m.unsqueeze(0).expand([n_samples] + list(qz_m.size()))
            qz_v = qz_v.unsqueeze(0).expand([n_samples] + list(qz_v.size()))
            ql_m = ql_m.unsqueeze(0).expand([n_samples] + list(ql_m.size()))
            ql_v = ql_v.unsqueeze(0).expand([n_samples] + list(ql_v.size()))
            z = self.z_encoder.sample(qz_m, qz_v)
            library = self.l_encoder.sample(ql_m, ql_v)

        # library = torch.clamp(library, max=14)
        # if (library >= 14).any():
        #     print('TOTOTATA')

        if not train_library:
            library = x.sum(1, keepdim=True).log()
        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion, z, library, batch_index, y
        )
        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r)

        return dict(
            px_scale=px_scale,
            px_r=px_r,
            px_rate=px_rate,
            px_dropout=px_dropout,
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            ql_m=ql_m,
            ql_v=ql_v,
            library=library,
        )

    def forward(
        self, x, local_l_mean, local_l_var, batch_index=None, y=None, train_library=True
    ):
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
        # Parameters for z latent distribution
        outputs = self.inference(x, batch_index, y)
        qz_m = outputs["qz_m"]
        qz_v = outputs["qz_v"]
        ql_m = outputs["ql_m"]
        ql_v = outputs["ql_v"]
        px_rate = outputs["px_rate"]
        px_r = outputs["px_r"]
        px_dropout = outputs["px_dropout"]

        # KL Divergence
        mean, scale = self.get_prior_params(device=qz_m.device)

        kl_divergence_z = kl(
            self.z_encoder.distrib(qz_m, qz_v), self.z_encoder.distrib(mean, scale)
        )
        if len(kl_divergence_z.size()) == 2:
            kl_divergence_z = kl_divergence_z.sum(dim=1)
        kl_divergence = kl_divergence_z
        reconst_loss = self.get_reconstruction_loss(x, px_rate, px_r, px_dropout)

        if train_library:
            kl_divergence_l = kl(
                Normal(ql_m, torch.sqrt(ql_v)),
                Normal(local_l_mean, torch.sqrt(local_l_var)),
            ).sum(dim=1)
            reconst_loss += kl_divergence_l
        nan_issues = (
            torch.isnan(reconst_loss).any()
            or torch.isinf(reconst_loss).any()
            or torch.isnan(kl_divergence).any()
            or torch.isinf(kl_divergence).any()
        )
        if nan_issues:
            for key in outputs:
                vals = outputs[key]
                print("{}: ({}, {})".format(key, vals.min().item(), vals.max().item()))
            raise ValueError
        return reconst_loss, kl_divergence

    def ratio_loss(
        self,
        x,
        local_l_mean,
        local_l_var,
        batch_index=None,
        y=None,
        return_mean=True,
        train_library=True,
        outputs=None,
    ):
        if outputs is None:
            outputs = self.inference(x, batch_index, y, train_library=train_library)

        px_r = outputs["px_r"]
        px_rate = outputs["px_rate"]
        px_dropout = outputs["px_dropout"]
        qz_m = outputs["qz_m"]
        qz_v = outputs["qz_v"]
        z = outputs["z"]
        ql_m = outputs["ql_m"]
        ql_v = outputs["ql_v"]
        library = outputs["library"]

        # KL Divergence
        z_prior_m, z_prior_v = self.get_prior_params(device=qz_m.device)

        log_px_zl = -self.get_reconstruction_loss(x, px_rate, px_r, px_dropout)
        log_pl = (
            Normal(local_l_mean, torch.sqrt(local_l_var)).log_prob(library).sum(dim=-1)
        )

        log_pz = self.z_encoder.distrib(z_prior_m, z_prior_v).log_prob(z).sum(dim=-1)
        log_qz_x = self.z_encoder.distrib(qz_m, qz_v).log_prob(z).sum(dim=-1)
        # if log_pz.dim() == 2 and log_qz_x.dim() == 2:
        #     log_pz = log_pz.sum(dim=1)
        #     log_qz_x = log_qz_x.sum(dim=1)

        log_ql_x = Normal(ql_m, torch.sqrt(ql_v)).log_prob(library).sum(dim=-1)
        if train_library:
            assert (
                log_px_zl.shape
                == log_pl.shape
                == log_pz.shape
                == log_qz_x.shape
                == log_ql_x.shape
            )
            log_ratio = (log_px_zl + log_pz + log_pl) - (log_qz_x + log_ql_x)
        else:
            assert log_px_zl.shape == log_pz.shape == log_qz_x.shape
            log_ratio = (log_px_zl + log_pz) - log_qz_x

        # if torch.isnan(log_ratio).any():
        #     print('FUFUFUFUF')

        if not return_mean:
            return log_ratio
        elbo = log_ratio.mean(dim=0)

        if torch.isnan(elbo).any() or torch.isinf(elbo).any():
            for key in outputs:
                vals = outputs[key]
                print("{}: ({}, {})".format(key, vals.min().item(), vals.max().item()))
            raise ValueError
        return -elbo


class LDVAE(VAE):
    r"""Linear-decoded Variational auto-encoder model.

    This model uses a linear decoder, directly mapping the latent representation
    to gene expression levels. It still uses a deep neural network to encode
    the latent representation.

    Compared to standard VAE, this model is less powerful, but can be used to
    inspect which genes contribute to variation in the dataset.

    :param n_input: Number of input genes
    :param n_batch: Number of batches
    :param n_labels: Number of labels
    :param n_hidden: Number of nodes per hidden layer (for encoder)
    :param n_latent: Dimensionality of the latent space
    :param n_layers: Number of hidden layers used for encoder NNs
    :param dropout_rate: Dropout rate for neural networks
    :param dispersion: One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell

    :param log_variational: Log(data+1) prior to encoding for numerical stability. Not normalization.
    :param reconstruction_loss:  One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
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
        dispersion: str = "gene",
        log_variational: bool = True,
        reconstruction_loss: str = "zinb",
    ):
        super().__init__(
            n_input,
            n_batch,
            n_labels,
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
            dispersion,
            log_variational,
            reconstruction_loss,
        )

        self.decoder = LinearDecoderSCVI(
            n_latent,
            n_input,
            n_cat_list=[n_batch],
            n_layers=n_layers,
            n_hidden=n_hidden,
        )

    def get_loadings(self):
        """ Extract per-gene weights (for each Z) in the linear decoder.
        """
        return self.decoder.factor_regressor.parameters()
