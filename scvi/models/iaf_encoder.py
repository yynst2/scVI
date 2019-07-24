import numpy as np
import logging
import torch
from torch import nn as nn
import torch.distributions as dist

from .modules import FCLayers

logger = logging.getLogger(__name__)


class EncoderH(nn.Module):
    def __init__(
        self,
        n_in,
        n_out,
        n_cat_list,
        n_layers,
        n_hidden,
        do_h,
        dropout_rate,
        use_batch_norm,
        do_sigmoid=True,
    ):
        """

        :param n_in:
        :param n_out:
        :param n_cat_list:
        :param n_layers:
        :param n_hidden:
        :param do_h:
        :param dropout_rate:
        :param use_batch_norm:
        """
        super().__init__()
        self.do_h = do_h
        self.do_sigmoid = do_sigmoid

        self.encoder = FCLayers(
            n_in=n_in,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
        )
        # with torch.no_grad():
        #     encoder0 =
        self.mu = nn.Linear(n_hidden, n_out)
        self.sigma = nn.Linear(n_hidden, n_out)
        if do_sigmoid:
            self.init_weights(self.sigma, bias_val=1.5)
        else:
            self.init_weights(self.sigma, bias_val=0.0)
        if do_h:
            self.h = nn.Linear(n_hidden, n_out)

        self.activation = nn.Sigmoid()

    def forward(self, x, *cat_list: int):
        """

        :param x:
        :param cat_list:
        :return:
        """
        z = self.encoder(x, *cat_list)
        mu = self.mu(z)
        sigma = self.sigma(z)
        if self.do_sigmoid:
            sigma = self.activation(sigma)
        else:
            sigma = nn.ReLU()(sigma)
        if self.do_h:
            h = self.h(z)
            return mu, sigma, h
        return mu, sigma

    @staticmethod
    def init_weights(m, bias_val=1.5):
        torch.nn.init.normal_(m.weight, mean=0., std=1e-8)
        torch.nn.init.constant_(m.bias, val=bias_val)


class EncoderIAF(nn.Module):
    def __init__(
        self,
        n_in,
        n_latent,
        n_cat_list,
        n_hidden,
        n_layers,
        t,
        dropout_rate=0.05,
        use_batch_norm=True,
        do_h=True,
    ):
        """
        Encoder using h representation as described in IAF paper

        :param n_in:
        :param n_latent:
        :param n_cat_list:
        :param n_hidden:
        :param n_layers:
        :param t:
        :param dropout_rate:
        :param use_batch_norm:
        """
        super().__init__()
        self.do_h = do_h
        msg = '' if do_h else 'Not '
        logger.info(msg='{}Using Hidden State'.format(msg))
        self.n_latent = n_latent
        self.encoders = torch.nn.ModuleList()
        self.encoders.append(
            EncoderH(
                n_in=n_in,
                n_out=n_latent,
                n_cat_list=n_cat_list,
                n_layers=n_layers,
                n_hidden=n_hidden,
                do_h=do_h,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
                do_sigmoid=True
            )
        )

        n_in = 2*n_latent if do_h else n_latent
        for _ in range(t - 1):
            self.encoders.append(
                EncoderH(
                    n_in=n_in,
                    n_out=n_latent,
                    n_cat_list=None,
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    do_h=False,
                    dropout_rate=dropout_rate,
                    use_batch_norm=use_batch_norm,
                    do_sigmoid=True
                )
            )

        self.dist0 = dist.Normal(
            loc=torch.zeros(n_latent, device="cuda"),
            scale=torch.ones(n_latent, device="cuda"),
        )

    def forward(self, x, *cat_list: int):
        """

        :param x:
        :param cat_list:
        :return:
        """
        if self.do_h:
            mu, sigma, h = self.encoders[0](x, *cat_list)
        else:
            mu, sigma = self.encoders[0](x, *cat_list)
            h = None

        # Big issue when x is 3d !!!
        # Should stay 2d!!
        eps = self.dist0.rsample((len(x),))
        assert eps.shape == (len(x), self.n_latent)

        z = mu + eps * sigma
        qz_x = sigma.log() + 0.5 * (eps ** 2) + 0.5 * np.log(2.0 * np.pi)
        qz_x = -qz_x.sum(dim=-1)

        # z shape (n_samples, n_batch, n_latent)
        for ar_nn in self.encoders[1:]:
            inp = torch.cat([z, h], dim=-1) if self.do_h else z
            mu, sigma = ar_nn(inp)
            z = sigma * z + (1.0 - sigma) * mu
            new_term = sigma.log()
            qz_x -= new_term.sum(dim=-1)
        return z, qz_x
