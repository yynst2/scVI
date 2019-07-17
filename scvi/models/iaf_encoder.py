import numpy as np

import torch
from torch import nn as nn
import torch.distributions as dist

from .modules import FCLayers


class EncoderH(nn.Module):
    def __init__(
        self, n_in, n_out, n_cat_list, n_layers, n_hidden, do_h, dropout_rate, use_batch_norm
    ):
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
                    n_in=n_latent,
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

    def forward(self, x, *cat_list: int):
        mu, sigma, h = self.encoders[0](x, *cat_list)
        eps = self.dist0.rsample((len(x),))
        assert eps.shape == (len(x), self.n_latent)

        z = mu + eps * sigma
        qz_x = sigma.log() + 0.5 * (eps ** 2) + 0.5 * np.log(2.0 * np.pi)
        qz_x = -qz_x.sum(dim=-1)

        for ar_nn in self.encoders[1:]:
            mu, sigma = ar_nn(z)
            z = sigma * z + (1.0 - sigma) * mu
            new_term = sigma.log()
            qz_x -= new_term.sum(dim=-1)
        return z, qz_x
