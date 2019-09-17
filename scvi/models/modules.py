import collections
from typing import Iterable, List
import logging

import torch
from torch import nn as nn
from torch.distributions import Normal, MultivariateNormal
from torch.distributions import Normal
from torch.nn import ModuleList

from scvi.models.utils import one_hot


logger = logging.getLogger(__name__)


def tril_indices(rows, cols, offset=0):
    return torch.ones(rows, cols, dtype=torch.uint8).tril(offset).nonzero()


def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()


def get_one_hot_cat_list(n_cat_list, cat_list):
    one_hot_cat_list = []  # for generality in this list many indices useless.
    assert len(n_cat_list) <= len(
        cat_list
    ), "nb. categorical args provided doesn't match init. params."
    for n_cat, cat in zip(n_cat_list, cat_list):
        assert not (
            n_cat and cat is None
        ), "cat not provided while n_cat != 0 in init. params."
        if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
            if cat.size(1) != n_cat:
                one_hot_cat = one_hot(cat, n_cat)
            else:
                one_hot_cat = cat  # cat has already been one_hot encoded
            one_hot_cat_list += [one_hot_cat]
    return one_hot_cat_list


class FCLayers(nn.Module):
    r"""A helper class to build fully-connected layers for a neural network.

    :param n_in: The dimensionality of the input
    :param n_out: The dimensionality of the output
    :param n_cat_list: A list containing, for each category of interest,
                 the number of categories. Each category will be
                 included using a one-hot encoding.
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    :param use_batch_norm: Whether to have `BatchNorm` layers or not
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        with_activation: bool = True,
    ):
        super().__init__()
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Linear(n_in + sum(self.n_cat_list), n_out),
                            # Below, 0.01 and 0.001 are the default values for `momentum` and `eps` from
                            # the tensorflow implementation of batch norm; we're using those settings
                            # here too so that the results match our old tensorflow code. The default
                            # setting from pytorch would probably be fine too but we haven't tested that.
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                            if use_batch_norm
                            else None,
                            nn.ReLU() if with_activation else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:])
                    )
                ]
            )
        )

    def forward(self, x: torch.Tensor, *cat_list: int, instance_id: int = 0):
        r"""Forward computation on ``x``.

        :param x: tensor of values with shape ``(n_in,)``
        :param cat_list: list of category membership(s) for this sample
        :param instance_id: Use a specific conditional instance normalization (batchnorm)
        :return: tensor of shape ``(n_out,)``
        :rtype: :py:class:`torch.Tensor`
        """
        one_hot_cat_list = get_one_hot_cat_list(self.n_cat_list, cat_list)
        for layers in self.fc_layers:
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat(
                                [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                            )
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear):
                            if x.dim() == 3:
                                one_hot_cat_list = [
                                    o.unsqueeze(0).expand(
                                        (x.size(0), o.size(0), o.size(1))
                                    )
                                    for o in one_hot_cat_list
                                ]
                            x = torch.cat((x, *one_hot_cat_list), dim=-1)
                        x = layer(x)
        return x


class ResNetBlock(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []
        self.layer1 = FCLayers(
            n_in=n_in,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=1,
            n_hidden=n_hidden,  # Should be useless
            with_activation=True,
            dropout_rate=dropout_rate,
        )
        self.layer2 = FCLayers(
            n_in=n_hidden,
            n_out=n_out,
            n_cat_list=n_cat_list,
            n_layers=1,
            n_hidden=n_hidden,  # Should be useless
            with_activation=False,
            dropout_rate=dropout_rate,
        )

        if n_cat_list is None:
            in_features = n_in
        else:
            in_features = n_in + sum(self.n_cat_list)
        if in_features != n_out:
            self.adjust = nn.Linear(in_features=in_features, out_features=n_out)
        else:
            self.adjust = nn.Sequential()

        self.last_bn = nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, *cat_list: int, instance_id: int = 0):
        one_hot_cat_list = get_one_hot_cat_list(self.n_cat_list, cat_list)

        h = self.layer1(x, *cat_list, instance_id=instance_id)
        h = self.layer2(h, *cat_list, instance_id=instance_id)

        # Residual connection adjustments if needed
        if x.dim() == 3:
            one_hot_cat_list = [
                o.unsqueeze(0).expand((x.size(0), o.size(0), o.size(1)))
                for o in one_hot_cat_list
            ]
        x = torch.cat((x, *one_hot_cat_list), dim=-1)
        x_adj = self.adjust(x)

        h = h + x_adj

        # last Batch normalization
        if h.dim() == 3:
            h = torch.cat(
                [(self.last_bn(slice_h)).unsqueeze(0) for slice_h in h], dim=0
            )
        else:
            h = self.last_bn(h)

        # Activation
        h = self.activation(h)
        return h


class DenseResNet(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_blocks: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        # First block
        modules = nn.ModuleList()
        modules.append(
            ResNetBlock(
                n_in=n_in,
                n_out=n_hidden,
                n_cat_list=n_cat_list,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
            )
        )
        # Intermediary blocks
        for block in range(n_blocks - 2):
            modules.append(
                ResNetBlock(
                    n_in=n_hidden,
                    n_out=n_hidden,
                    n_cat_list=n_cat_list,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                )
            )
        # Last Block
        modules.append(
            ResNetBlock(
                n_in=n_hidden,
                n_out=n_out,
                n_cat_list=n_cat_list,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
            )
        )
        self.resnet_layers = modules

    def forward(self, x: torch.Tensor, *cat_list: int, instance_id: int = 0):
        h = x
        for module in self.resnet_layers:
            h = module(h, *cat_list, instance_id=instance_id)
        return h

class LinearExpLayer(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        use_batch_norm=True,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.linear_layer = nn.Sequential(
            nn.Linear(n_in, n_out),
            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001) if use_batch_norm else None,
            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
        )

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""Forward computation on ``x``.

        :param x: tensor of values with shape ``(n_in,)``
        :param cat_list: list of category membership(s) for this sample
        :return: tensor of shape ``(n_out,)``
        :rtype: :py:class:`torch.Tensor`
        """
        for layer in self.linear_layer:
            if layer is not None:
                x = layer(x)
        return torch.clamp(x.exp(), max=1e6)


# Encoder
class Encoder(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        full_cov=False,
        autoregressive=False,
        prevent_saturation: bool = False,
    ):
        super().__init__()
        assert not (full_cov and autoregressive)
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        self.prevent_saturation = prevent_saturation
        logger.info("Preventing saturation: {}".format(prevent_saturation))
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.full_cov = full_cov
        self.autoregressive = autoregressive
        self.n_output = n_output
        if full_cov:
            self.var_encoder = nn.Linear(n_hidden, int((n_output * (n_output + 1)) / 2))
        elif autoregressive:
            self.var_encoder = nn.Linear(n_hidden, n_output)
            self.ltria = nn.Linear(n_hidden, int((n_output * (n_output - 1)) / 2))
        else:
            self.var_encoder = nn.Linear(n_hidden, n_output)

    def reparameterize(self, mu, var, sample_size=torch.Size()):
        return self.distrib(mu, var).rsample(sample_size)

    def sample(self, mu, var, sample_size=torch.Size()):
        return self.distrib(mu, var).sample(sample_size)

    def distrib(self, mu, var):
        if self.full_cov or self.autoregressive:
            return MultivariateNormal(loc=mu, covariance_matrix=var)
        else:
            return Normal(mu, var.sqrt())

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\) (clamped to \\( [-5, 5] \\))
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim N(q_m, \\mathbf{I}q_v) \\)

        :param x: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 3-tuple of :py:class:`torch.Tensor`
        """

        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = self.var_encoder(q)

        if self.prevent_saturation:
            q_m = 14.0 * nn.Tanh()(self.mean_encoder(q))

        q_v = self.get_cov(
            q_v
        )  # (computational stability safeguard)torch.clamp(, -5, 5)
        if self.autoregressive:
            l_vals = self.ltria(q)
            n_batch = q.size(0)
            l_mat = torch.zeros(n_batch, self.n_output, self.n_output, device=q.device)
            indices = tril_indices(self.n_output, self.n_output, offset=-1)
            rg = torch.arange(self.n_output, device=x.device)
            l_mat[:, indices[:, 0], indices[:, 1]] = l_vals
            l_mat[:, rg, rg] = 1.0
            q_m = torch.bmm(l_mat, q_m.view(n_batch, self.n_output, 1)).squeeze()
            last_term = q_v.view((n_batch, self.n_output, 1)) * l_mat.transpose(-1, -2)
            q_v = torch.bmm(l_mat, last_term)

        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent

    def get_cov(self, x):
        if self.full_cov:
            n_batch = x.size(0)
            l_mat = torch.zeros(n_batch, self.n_output, self.n_output, device=x.device)
            lower_idx = tril_indices(self.n_output, self.n_output)
            l_mat[:, lower_idx[:, 0], lower_idx[:, 1]] = x
            rg = torch.arange(self.n_output, device=x.device)
            l_mat[:, rg, rg] = 1e-4 + torch.nn.Softplus()(l_mat[:, rg, rg])

            res = torch.matmul(l_mat, l_mat.transpose(-1, -2))
            return res
        else:
            if self.prevent_saturation:
                x = 5 * nn.Sigmoid()(x)
            else:
                x = torch.exp(x) + 1e-4
            return x


# Decoder
class DecoderSCVI(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        n_blocks: int = 0,
        do_last_skip: bool = False
    ):
        super().__init__()
        self.do_last_skip = do_last_skip
        if n_blocks == 0:
            self.px_decoder = FCLayers(
                n_in=n_input,
                n_out=n_hidden,
                n_cat_list=n_cat_list,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=0,
            )
        else:
            logger.info("Using ResNet structure for the Decoder")
            self.px_decoder = DenseResNet(
                n_in=n_input,
                n_out=n_hidden,
                n_cat_list=n_cat_list,
                n_blocks=n_blocks,
                n_hidden=n_hidden,
                dropout_rate=0.1,
            )

        n_in = n_hidden + n_input if do_last_skip else n_hidden
        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_in, n_output), nn.Softmax(dim=-1)
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_in, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_in, n_output)

    def forward(
        self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int
    ):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        :param dispersion: One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell

        :param z: tensor with shape ``(n_input,)``
        :param library: library size
        :param cat_list: list of category membership(s) for this sample
        :return: parameters for the ZINB distribution of expression
        :rtype: 4-tuple of :py:class:`torch.Tensor`
        """

        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)

        if self.do_last_skip:
            last_input = torch.cat([px, z], dim=1)
        else:
            last_input = px
        px_scale = self.px_scale_decoder(last_input)
        px_dropout = self.px_dropout_decoder(last_input)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r_decoder(last_input) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout


class LinearDecoderSCVI(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
    ):
        super(LinearDecoderSCVI, self).__init__()

        # mean gamma
        self.n_batches = n_cat_list[0]  # Just try a simple case for now
        if self.n_batches > 1:
            self.batch_regressor = nn.Linear(self.n_batches - 1, n_output, bias=False)
        else:
            self.batch_regressor = None

        self.factor_regressor = nn.Linear(n_input, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_input, n_output)

    def forward(
        self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int
    ):
        # The decoder returns values for the parameters of the ZINB distribution
        p1_ = self.factor_regressor(z)
        if self.n_batches > 1:
            one_hot_cat = one_hot(cat_list[0], self.n_batches)[:, :-1]
            p2_ = self.batch_regressor(one_hot_cat)
            raw_px_scale = p1_ + p2_
        else:
            raw_px_scale = p1_

        px_scale = torch.softmax(raw_px_scale, dim=-1)
        px_dropout = self.px_dropout_decoder(z)
        px_rate = torch.exp(library) * px_scale
        px_r = None

        return px_scale, px_r, px_rate, px_dropout


# Decoder
class Decoder(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions to ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.
    Output is the mean and variance of a multivariate Gaussian

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
    ):
        super().__init__()
        self.decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
        )

        self.mean_decoder = nn.Linear(n_hidden, n_output)
        self.var_decoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns tensors for the mean and variance of a multivariate distribution

        :param x: tensor with shape ``(n_input,)``
        :param cat_list: list of category membership(s) for this sample
        :return: Mean and variance tensors of shape ``(n_output,)``
        :rtype: 2-tuple of :py:class:`torch.Tensor`
        """

        # Parameters for latent distribution
        p = self.decoder(x, *cat_list)
        p_m = self.mean_decoder(p)
        p_v = torch.exp(self.var_decoder(p))
        return p_m, p_v


# Decoder
class DecoderPoisson(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
    ):
        super().__init__()
        self.rate_decoder = FCLayers(
            n_in=n_input,
            n_out=n_output,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=5e-2,
        )

        # self.rate_decoder = LinearExpLayer(
        #     n_in=n_input,
        #     n_out=n_output,
        #     n_cat_list=n_cat_list,
        #     dropout_rate=5e-2,
        # )

    def forward(self, z: torch.Tensor, library: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        :param z: tensor with shape ``(n_input,)``
        :param library: library size
        :param cat_list: list of category membership(s) for this sample
        :return: parameters for the ZINB distribution of expression
        :rtype: 4-tuple of :py:class:`torch.Tensor`
        """

        # The decoder returns values for the parameters of the ZINB distribution
        rate = self.rate_decoder(z, *cat_list)
        # px_rate = torch.exp(library) * rate  # torch.clamp( , max=12)
        px_rate = rate.exp()
        px_rate = 1e-6 + torch.clamp(px_rate, max=1e5)
        return px_rate


class MultiEncoder(nn.Module):
    def __init__(
        self,
        n_heads: int,
        n_input_list: List[int],
        n_output: int,
        n_hidden: int = 128,
        n_layers_individual: int = 1,
        n_layers_shared: int = 2,
        n_cat_list: Iterable[int] = None,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.encoders = ModuleList(
            [
                FCLayers(
                    n_in=n_input_list[i],
                    n_out=n_hidden,
                    n_cat_list=n_cat_list,
                    n_layers=n_layers_individual,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    use_batch_norm=True,
                )
                for i in range(n_heads)
            ]
        )

        self.encoder_shared = FCLayers(
            n_in=n_hidden,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers_shared,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, head_id: int, *cat_list: int):
        q = self.encoders[head_id](x, *cat_list)
        q = self.encoder_shared(q, *cat_list)

        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q))
        latent = reparameterize_gaussian(q_m, q_v)

        return q_m, q_v, latent


class MultiDecoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden_conditioned: int = 32,
        n_hidden_shared: int = 128,
        n_layers_conditioned: int = 1,
        n_layers_shared: int = 1,
        n_cat_list: Iterable[int] = None,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        n_out = n_hidden_conditioned if n_layers_shared else n_hidden_shared
        if n_layers_conditioned:
            self.px_decoder_conditioned = FCLayers(
                n_in=n_input,
                n_out=n_out,
                n_cat_list=n_cat_list,
                n_layers=n_layers_conditioned,
                n_hidden=n_hidden_conditioned,
                dropout_rate=dropout_rate,
                use_batch_norm=True,
            )
            n_in = n_out
        else:
            self.px_decoder_conditioned = None
            n_in = n_input

        if n_layers_shared:
            self.px_decoder_final = FCLayers(
                n_in=n_in,
                n_out=n_hidden_shared,
                n_cat_list=[],
                n_layers=n_layers_shared,
                n_hidden=n_hidden_shared,
                dropout_rate=dropout_rate,
                use_batch_norm=True,
            )
            n_in = n_hidden_shared
        else:
            self.px_decoder_final = None

        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_in, n_output), nn.Softmax(dim=-1)
        )
        self.px_r_decoder = nn.Linear(n_in, n_output)
        self.px_dropout_decoder = nn.Linear(n_in, n_output)

    def forward(
        self,
        z: torch.Tensor,
        dataset_id: int,
        library: torch.Tensor,
        dispersion: str,
        *cat_list: int
    ):

        px = z
        if self.px_decoder_conditioned:
            px = self.px_decoder_conditioned(px, *cat_list, instance_id=dataset_id)
        if self.px_decoder_final:
            px = self.px_decoder_final(px, *cat_list)

        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        px_rate = torch.exp(library) * px_scale
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None

        return px_scale, px_r, px_rate, px_dropout
