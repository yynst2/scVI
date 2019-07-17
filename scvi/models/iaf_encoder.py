import numpy as np

import torch
from torch import nn as nn
import torch.distributions as dist

from .modules import FCLayers


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
        if do_h:
            self.h = nn.Linear(n_hidden, n_out)

        self.sigm = nn.Sigmoid()

    def forward(self, x, *cat_list: int):
        """

        :param x:
        :param cat_list:
        :return:
        """
        z = self.encoder(x, *cat_list)
        mu = self.mu(z)
        sigma = self.sigm(self.sigma(z))
        if self.do_h:
            h = self.h(z)
            return mu, sigma, h
        return mu, sigma


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
    ):
        """

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
        self.n_latent = n_latent
        self.encoders = torch.nn.ModuleList()
        self.encoders.append(
            EncoderH(
                n_in=n_in,
                n_out=n_latent,
                n_cat_list=n_cat_list,
                n_layers=n_layers,
                n_hidden=n_hidden,
                do_h=True,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
            )
        )
        for _ in range(t - 2):
            self.encoders.append(
                EncoderH(
                    n_in=2*n_latent,
                    n_out=n_latent,
                    n_cat_list=None,
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    do_h=False,
                    dropout_rate=dropout_rate,
                    use_batch_norm=use_batch_norm,
                )
            )

        self.dist0 = dist.Normal(
            loc=torch.zeros(n_latent, device="cuda"),
            scale=torch.ones(n_latent, device="cuda"),
        )

    def forward(self, x, *cat_list: int, n_samples: int = 1):
        """

        :param x:
        :param cat_list:
        :param n_samples:
        :return:
        """
        mu, sigma, h = self.encoders[0](x, *cat_list)
        eps = self.dist0.rsample((n_samples, len(x),))
        assert eps.shape == (n_samples, len(x), self.n_latent)

        z = mu + eps * sigma
        qz_x = sigma.log() + 0.5 * (eps ** 2) + 0.5 * np.log(2.0 * np.pi)
        qz_x = -qz_x.sum(dim=-1)

        h = h.unsqueeze(0)  # shape (1, n_batch, n_latent)
        h = h.expand(n_samples, -1, -1) # shape (n_samples, n_batch, n_latent) same as z
        # z shape (n_samples, n_batch, n_latent)
        for ar_nn in self.encoders[1:]:
            mu, sigma = ar_nn(torch.cat([z, h], dim=-1))
            z = sigma * z + (1.0 - sigma) * mu
            new_term = sigma.log()
            qz_x -= new_term.sum(dim=-1)
        if n_samples == 1:
            z = z.squeeze()
            qz_x = qz_x.squeeze()
        return z, qz_x
