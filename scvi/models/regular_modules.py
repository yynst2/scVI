import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal
from scipy.stats import genpareto

import logging

logger = logging.getLogger(__name__)


class FCLayersA(nn.Module):
    def __init__(
        self,
        n_input,
        n_output,
        dropout_rate=0.1,
        do_batch_norm=True,
    ):
        super().__init__()
        self.to_hidden = nn.Linear(in_features=n_input, out_features=500)
        self.do_batch_norm = do_batch_norm
        if do_batch_norm:
            self.batch_norm = nn.BatchNorm1d(num_features=500)

        self.to_out = nn.Linear(in_features=500, out_features=n_output)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.activation = nn.SELU()

    def forward(self, x):
        res = self.to_hidden(x)
        if self.do_batch_norm:
            if res.ndim == 4:
                n1, n2, n3, n4 = res.shape
                res = self.batch_norm(res.view(n1*n2*n3, n4))
                res = res.view(n1, n2, n3, n4)
            elif res.ndim == 3:
                n1, n2, n3 = res.shape
                res = self.batch_norm(res.view(n1*n2, n3))
                res = res.view(n1, n2, n3)
            elif res.ndim == 2:
                res = self.batch_norm(res)
            else:
                raise ValueError("{} ndim not handled.".format(res.ndim))
        res = self.activation(res)
        res = self.dropout(res)
        res = self.to_out(res)
        return res


class EncoderA(nn.Module):
    def __init__(
        self,
        n_input,
        n_output,
        n_hidden,
        dropout_rate,
        do_batch_norm,
    ):
        super().__init__()
        self.encoder = FCLayersA(
            n_input=n_input,
            n_output=n_hidden,
            dropout_rate=dropout_rate,
            do_batch_norm=do_batch_norm
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.tanh = nn.Tanh()

    def forward(self, x, n_samples, squeeze=True, reparam=True):
        q = self.encoder(x)
        q_m = self.mean_encoder(q)
        q_v = self.var_encoder(q)
        q_v = 16.0 * self.tanh(q_v)
        q_v = q_v.exp()
        # q_v = 1e-16 + q_v.exp()

        variational_dist = Normal(loc=q_m, scale=q_v.sqrt())

        if n_samples == 1 and squeeze:
            sample_shape = []
        else:
            sample_shape = (n_samples,)
        if reparam:
            latent = variational_dist.rsample(sample_shape=sample_shape)
        else:
            latent = variational_dist.sample(sample_shape=sample_shape)
        return dict(q_m=q_m, q_v=q_v, latent=latent)


class LinearEncoder(nn.Module):
    def __init__(
        self,
        n_input,
        n_output,
    ):
        super().__init__()
        self.mean_encoder = nn.Linear(n_input, n_output)
        self.n_output = n_output

        self.var_vals = nn.Parameter(0.1 * torch.rand(n_output, n_output), requires_grad=True)

    @property
    def l_mat_encoder(self):
        l_mat = torch.tril(self.var_vals)
        range_vals = np.arange(self.n_output)
        l_mat[range_vals, range_vals] = l_mat[range_vals, range_vals].exp()
        return l_mat

    @property
    def var_encoder(self):
        l_mat = self.l_mat_encoder
        return l_mat.matmul(l_mat.T)

    def forward(self, x, n_samples, reparam=True, squeeze=True):
        q_m = self.mean_encoder(x)
        l_mat = self.var_encoder
        q_v = l_mat.matmul(l_mat.T)

        variational_dist = MultivariateNormal(
            loc=q_m,
            scale_tril=l_mat
        )

        if squeeze and n_samples == 1:
            sample_shape = []
        else:
            sample_shape = (n_samples,)
        if reparam:
            latent = variational_dist.rsample(sample_shape=sample_shape)
        else:
            latent = variational_dist.sample(sample_shape=sample_shape)
        return dict(q_m=q_m, q_v=q_v, latent=latent)


# Decoder
class DecoderA(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int,
    ):
        super().__init__()
        self.decoder = FCLayersA(
            n_input=n_input,
            n_output=n_hidden,
            dropout_rate=0.,
        )

        self.mean_decoder = nn.Linear(n_hidden, n_output)
        self.var_decoder = nn.Linear(n_hidden, n_output)

        self.tanh = nn.Tanh()

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
        p_v = self.var_decoder(p)
        p_v = 16. * self.tanh(p_v)
        return p_m, p_v.exp()


class ClassifierA(nn.Module):
    def __init__(
        self,
        n_input,
        n_output,
        dropout_rate=0.,
        do_batch_norm=True,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            FCLayersA(
                n_input,
                n_output,
                dropout_rate=dropout_rate,
                do_batch_norm=do_batch_norm,
            ),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        probas = self.classifier(x)
        probas = probas + 1e-16
        probas = probas / probas.sum(-1, keepdim=True)
        return probas


class BernoulliDecoderA(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        dropout_rate: float = 0.0,
        do_batch_norm: bool = True,
    ):
        super().__init__()
        self.loc = FCLayersA(
            n_input,
            n_output,
            dropout_rate=dropout_rate,
            do_batch_norm=do_batch_norm,
        )

    def forward(self, x):
        means = self.loc(x)
        means = nn.Sigmoid()(means)
        return means


class PSIS:
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.m_largest_samples = int(min(num_samples/5.0, 3*np.sqrt(num_samples)))
        self.shape = []
        self.loc = []
        self.scale = []

    def fit(self, log_ratios: np.ndarray):
        if log_ratios.ndim == 1:
            log_ratios = log_ratios[None, :]

        for log_ratio_ex in log_ratios:
            m_biggest = np.argsort(-log_ratio_ex)[:self.m_largest_samples]
            m_best_log_ratios = log_ratio_ex[m_biggest]

            res = genpareto.fit(m_best_log_ratios)
            if len(res) == 2:
                # self.loc, self.scale = res
                pass
            elif len(res) == 3:
                # self.shape, self.loc, self.scale = res
                self.shape.append(res[0])
            else:
                raise ValueError("Unknown data results")
