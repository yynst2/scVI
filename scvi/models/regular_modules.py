import torch
import torch.nn as nn
from torch.distributions import Normal


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

    def forward(self, x, n_samples, squeeze=True, reparam=True):
        q = self.encoder(x)
        q_m = self.mean_encoder(q)
        q_v = self.var_encoder(q)
        q_v = 1e-16 + q_v.exp()

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
        return p_m, 1e-16 + p_v.exp()
