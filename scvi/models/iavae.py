import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F

from .log_likelihood import log_zinb_positive, log_nb_positive
from .modules import DecoderSCVI, Encoder
from .iaf_encoder import EncoderIAF
from .utils import one_hot


class IAVAE(nn.Module):
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
        t: int = 3,
        dropout_rate: float = 5e-2,
        dispersion: str = "gene",
        log_variational: bool = True,
        reconstruction_loss: str = "zinb",
    ):
        super().__init__()
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.reconstruction_loss = reconstruction_loss
        # Automatically deactivate if useless
        self.n_batch = n_batch
        self.n_labels = n_labels

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        else:  # gene-cell
            pass

        # latent space representation
        self.z_encoder = EncoderIAF(
            n_in=n_input,
            n_latent=n_latent,
            n_cat_list=None,
            n_layers=n_layers,
            t=t,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(
            n_input, 1, n_layers=1, n_hidden=n_hidden, dropout_rate=dropout_rate
        )
        # decoder goes from n_latent-dimensional space to n_input-d data
        self.decoder = DecoderSCVI(
            n_latent,
            n_input,
            n_cat_list=[n_batch],
            n_layers=n_layers,
            n_hidden=n_hidden,
        )

    def ratio_loss(self, x, local_l_mean, local_l_var, batch_index=None, y=None):
        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        # variationnal probas computation
        z, log_qz_x = self.z_encoder(x, batch_index)
        ql_m, ql_v, library = self.l_encoder(x_)
        log_ql_x = dist.Normal(ql_m, torch.sqrt(ql_v)).log_prob(library).sum(dim=-1)

        # priors computation
        log_pz = dist.Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z).sum(dim=-1)
        log_pl = dist.Normal(local_l_mean, torch.sqrt(local_l_var)).log_prob(library).sum(dim=-1)

        # reconstruction proba computation
        px_scale, px_r, px_rate, px_dropout = self.decoder(self.dispersion, z, library, batch_index, y)
        if self.dispersion == "gene-label":
            px_r = F.linear(one_hot(y, self.n_labels), self.px_r)  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r)

        log_px_zl = -self.get_reconstruction_loss(x, px_rate, px_r, px_dropout)

        elbo = (
            log_px_zl + log_pz + log_pl
            - log_qz_x - log_ql_x
        ).mean(dim=0)

        return -elbo

    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout):
        # Reconstruction Loss
        if self.reconstruction_loss == 'zinb':
            reconst_loss = -log_zinb_positive(x, px_rate, px_r, px_dropout)
        elif self.reconstruction_loss == 'nb':
            reconst_loss = -log_nb_positive(x, px_rate, px_r)
        else:
            raise NotImplementedError
        return reconst_loss

