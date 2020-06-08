# -*- coding: utf-8 -*-
"""Main module."""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence as kl

from scvi.models.log_likelihood import log_zinb_positive, log_nb_positive
from scvi.models.modules import Encoder, DecoderSCVI, LinearDecoderSCVI
from scvi.models.iaf_encoder import EncoderIAF
from scvi.models.utils import one_hot

logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True


# VAE model
class VAE(nn.Module):
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
        decoder_dropout_rate: float = 0.0,
        dispersion: str = "gene",
        use_batch_norm: bool = False,
        use_weight_norm: bool = False,
        use_layer_norm: bool = False,
        iaf_t: int = 0,
        scale_normalize: bool = True,
        log_variational: bool = True,
        reconstruction_loss: str = "zinb",
        autoregresssive: bool = False,
        log_p_z=None,
        n_blocks=0,
        decoder_do_last_skip=False,
        prevent_library_saturation: bool = False,
    ):
        super().__init__()

        if iaf_t >= 1:
            self.z_encoder = EncoderIAF(
                n_in=n_input,
                n_latent=n_latent,
                n_cat_list=None,
                n_hidden=n_hidden,
                n_layers=n_layers,
                t=iaf_t,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
                do_h=True,
            )
        else:
            self.z_encoder = Encoder(
                n_input,
                n_latent,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                use_weight_norm=use_weight_norm,
            )
        self.n_input = n_input
        # l encoder goes from n_input-dimensional data to 1-d library size
        # n_cat_list = [n_batch] if n_batch is not None else None
        # logger.info("Defining l encoder with cats {}".format(n_cat_list))
        self.l_encoder = Encoder(
            n_input,
            1,
            # n_cat_list=n_cat_list,
            n_layers=1,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm,
            use_weight_norm=use_weight_norm,
            dropout_rate=dropout_rate,
            prevent_saturation=prevent_library_saturation,
        )
        # decoder goes from n_latent-dimensional space to n_input-d data
        self.decoder = None
        self.log_variational = log_variational

        self.decoder = DecoderSCVI(
            n_latent,
            n_input,
            n_cat_list=[n_batch],
            scale_normalize=scale_normalize,
            n_layers=n_layers,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm,
            use_weight_norm=use_weight_norm,
            n_hidden=n_hidden,
            n_blocks=n_blocks,
            dropout_rate=decoder_dropout_rate,
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

    def log_p_z(self, z: torch.Tensor):
        if self.log_p_z_fixed is not None:
            return self.log_p_z_fixed(z)
        else:
            z_prior_m, z_prior_v = self.get_prior_params(device=z.device)
            return Normal(z_prior_m, z_prior_v.sqrt()).log_prob(z).sum(-1)
            # return self.z_encoder.distrib(z_prior_m, z_prior_v).log_prob(z)

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
        z_post = self.z_encoder(sample_batch)
        # qz_m = z_post["q_m"]
        # qz_v = z_post["q_v"]
        z = z_post["latent"]
        batch_index = fixed_batch * torch.ones_like(sample_batch[:, [0]])
        library = 4.0 * torch.ones_like(sample_batch[:, [0]])
        px_scale, _, _, _ = self.decoder("gene", z, library, batch_index)
        return px_scale

    def inference(
        self, x, batch_index=None, y=None, reparam=True, n_samples=1, train_library=True
    ):
        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        # Sampling
        library_post = self.l_encoder(x_, n_samples=n_samples, reparam=reparam)
        library_variables = dict(
            ql_m=library_post["q_m"],
            ql_v=library_post["q_v"],
            library=library_post["latent"],
            log_ql_x=library_post["posterior_density"],
        )

        z_post = self.z_encoder(x_, y, n_samples=n_samples, reparam=reparam)
        z_variables = dict(
            qz_m=z_post["q_m"],
            qz_v=z_post["q_v"],
            z=z_post["latent"],
            log_qz_x=z_post["posterior_density"],
        )

        z = z_variables["z"]
        if not train_library:
            library = x.sum(1, keepdim=True).log()
        else:
            library = library_variables["library"]
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

        decoder_variables = dict(
            px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout
        )

        return {
            **library_variables,
            **z_variables,
            **decoder_variables,
        }

    def forward(
        self,
        x,
        local_l_mean,
        local_l_var,
        batch_index=None,
        y=None,
        loss=None,
        n_samples=1,
        train_library=True,
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
        outputs = self.inference(
            x,
            batch_index,
            y,
            n_samples=n_samples,
            train_library=train_library,
            reparam=True,
        )
        # qz_m = outputs["qz_m"]
        # qz_v = outputs["qz_v"]
        # ql_m = outputs["ql_m"]
        # ql_v = outputs["ql_v"]
        px_rate = outputs["px_rate"]
        px_r = outputs["px_r"]
        px_dropout = outputs["px_dropout"]
        library = outputs["library"]
        z = outputs["z"]

        z_prior_m, z_prior_v = self.get_prior_params(device=x.device)
        log_px_zl = -self.get_reconstruction_loss(x, px_rate, px_r, px_dropout)
        log_pl = (
            Normal(local_l_mean, torch.sqrt(local_l_var)).log_prob(library).sum(dim=-1)
        )

        log_pz = Normal(z_prior_m, z_prior_v.sqrt()).log_prob(z).sum(dim=-1)
        # log_qz_x = self.z_encoder.distrib(qz_m, qz_v).log_prob(z).sum(dim=-1)
        log_qz_x = outputs["log_qz_x"]
        # if log_pz.dim() == 2 and log_qz_x.dim() == 2:
        #     log_pz = log_pz.sum(dim=1)
        #     log_qz_x = log_qz_x.sum(dim=1)

        # log_ql_x = Normal(ql_m, torch.sqrt(ql_v)).log_prob(library).sum(dim=-1)
        log_ql_x = outputs["log_ql_x"]
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

        if loss is None:
            return {
                "log_ratio": log_ratio,
                **outputs,
            }
        elif loss == "ELBO":
            obj = -log_ratio.mean(0)
        elif loss == "IWELBO":
            obj = -(torch.softmax(log_ratio, dim=0).detach() * log_ratio).sum(dim=0)
            # print("log_ratio : ", log_ratio.shape)
            # print("log_ratio : ", log_ratio.mean())
            # print("log_px_zl :", log_px_zl.min(), log_px_zl.max())
            # print("log_pl :", log_pl.min(), log_pl.max())
            # print("log_pz :", log_pz.min(), log_pz.max())
            # print("log_qz_x :", log_qz_x.min(), log_qz_x.max())
            # print("log_ql_x :", log_ql_x.min(), log_ql_x.max())
        return obj

    def log_ratios(
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
        pass

    # def ratio_loss(
    #     self,
    #     x,
    #     local_l_mean,
    #     local_l_var,
    #     batch_index=None,
    #     y=None,
    #     return_mean=True,
    #     train_library=True,
    #     outputs=None,
    # ):
    #     if outputs is None:
    #         outputs = self.inference(x, batch_index, y, train_library=train_library)

    #     px_r = outputs["px_r"]
    #     px_rate = outputs["px_rate"]
    #     px_dropout = outputs["px_dropout"]
    #     qz_m = outputs["qz_m"]
    #     qz_v = outputs["qz_v"]
    #     z = outputs["z"]
    #     ql_m = outputs["ql_m"]
    #     ql_v = outputs["ql_v"]
    #     library = outputs["library"]

    #     # KL Divergence
    #     z_prior_m, z_prior_v = self.get_prior_params(device=qz_m.device)

    #     log_px_zl = -self.get_reconstruction_loss(x, px_rate, px_r, px_dropout)
    #     log_pl = (
    #         Normal(local_l_mean, torch.sqrt(local_l_var)).log_prob(library).sum(dim=-1)
    #     )

    #     log_pz = self.z_encoder.distrib(z_prior_m, z_prior_v).log_prob(z).sum(dim=-1)
    #     log_qz_x = self.z_encoder.distrib(qz_m, qz_v).log_prob(z).sum(dim=-1)
    #     # if log_pz.dim() == 2 and log_qz_x.dim() == 2:
    #     #     log_pz = log_pz.sum(dim=1)
    #     #     log_qz_x = log_qz_x.sum(dim=1)

    #     log_ql_x = Normal(ql_m, torch.sqrt(ql_v)).log_prob(library).sum(dim=-1)
    #     if train_library:
    #         assert (
    #             log_px_zl.shape
    #             == log_pl.shape
    #             == log_pz.shape
    #             == log_qz_x.shape
    #             == log_ql_x.shape
    #         )
    #         log_ratio = (log_px_zl + log_pz + log_pl) - (log_qz_x + log_ql_x)
    #     else:
    #         assert log_px_zl.shape == log_pz.shape == log_qz_x.shape
    #         log_ratio = (log_px_zl + log_pz) - log_qz_x

    #     # if torch.isnan(log_ratio).any():
    #     #     print('FUFUFUFUF')

    #     if not return_mean:
    #         return log_ratio
    #     elbo = log_ratio.mean(dim=0)

    #     if torch.isnan(elbo).any() or torch.isinf(elbo).any():
    #         print("train_library ?", train_library)
    #         for key in outputs:
    #             vals = outputs[key]
    #             print("{}: ({}, {})".format(key, vals.min().item(), vals.max().item()))
    #         raise ValueError
    #     return -elbo

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
        l_dist = self.l_encoder(x)
        library = l_dist["latent"]
        return library

    def get_prior_params(self, device):
        mean = torch.zeros((self.n_latent,), device=device)
        # if self.z_full_cov or self.z_autoregressive:
        #     scale = self.prior_scale * torch.eye(self.n_latent, device=device)
        # else:
        scale = torch.ones((self.n_latent,), device=device)
        return mean, scale

    @staticmethod
    def get_list_params(*params):
        res = []
        for param_li in params:
            res += list(filter(lambda p: p.requires_grad, param_li))
        return res


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
