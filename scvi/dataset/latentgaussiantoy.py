import os

import numpy as np
import pyro
import torch
from pyro.infer import config_enumerate
from pyro.infer.mcmc import NUTS, MCMC

from .logpoisson import save_figs

dist = pyro.distributions


class LatentGaussianToy:
    def __init__(self, n_genes: int, n_latent: int):
        self.n_genes = n_genes
        self.n_latent = n_latent

        self.b = torch.randn(size=(n_genes,)).float()
        self.w = torch.randn(size=(n_genes, n_latent)).float()

        self.prior_mean = torch.zeros(size=(n_latent,)).float()
        self.sigma = 1e0
        self.prior_std = (self.sigma ** 2 * torch.ones_like(self.prior_mean)).float()

    @config_enumerate(default="sequential")
    def pyro_mdl(self, data: torch.Tensor):
        with pyro.plate("data", len(data)):
            z = pyro.sample(
                "z", dist.Normal(self.prior_mean, self.prior_std).to_event(1)
            )
            mean = (
                (self.w @ z.reshape(-1, self.n_latent, 1)) + self.b.reshape(-1, 1)
            ).squeeze()
            std = (self.sigma ** 2) * torch.ones_like(self.b).reshape(1, self.n_genes)
            # print(mean.shape, std.shape)
            pyro.sample("x", dist.Normal(mean, std).independent(1), obs=data)

    def gt_posterior(self, x: torch.Tensor):
        """

        :param x: Shape (n_batch, n_genes)
        :return:
        """
        x = x.reshape(shape=(-1, self.n_genes, 1))
        sigma_coef = 1.0 / (self.sigma ** 2)

        q_v = (sigma_coef * self.w.transpose(-1, -2) @ self.w) + torch.eye(
            self.n_latent
        )
        q_v = torch.inverse(q_v)
        diff = x - self.b.reshape(shape=(-1, 1))
        q_m = sigma_coef * q_v @ (self.w.transpose(-1, -2) @ diff)
        # had shape n_batch, n_latent, 1
        q_m = q_m.squeeze()
        # had shape (n_latent, n_latent)
        n_batch = q_m.shape[0]
        q_v = q_v.reshape((1, self.n_latent, self.n_latent)).expand(
            (n_batch, self.n_latent, self.n_latent)
        )
        assert q_m.shape == (n_batch, self.n_latent) and q_v.shape == (
            n_batch,
            self.n_latent,
            self.n_latent,
        )
        return q_m, q_v

    def compute_posteriors(self, x_obs: torch.Tensor, mcmc_kwargs: dict = None):
        """

        :param x_obs:
        :param mcmc_kwargs: By default:
        {num_samples=1000, warmup_steps=1000, num_chains=4)
        :return:
        """
        if mcmc_kwargs is None:
            mcmc_kwargs = {"num_samples": 1000, "warmup_steps": 1000, "num_chains": 4}
        kernel = NUTS(
            self.pyro_mdl,
            adapt_step_size=True,
            max_plate_nesting=1,
            jit_compile=True,
            target_accept_prob=0.6,
        )
        mcmc_run = MCMC(kernel, **mcmc_kwargs).run(data=x_obs)
        marginals = mcmc_run.marginal(sites=["z"])
        marginals_supp = marginals.support()
        z_x = marginals_supp["z"]
        return z_x, marginals

    def local_bayes(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        save_dir: str = None,
        mcmc_kwargs: dict = None,
    ):
        """

        :param x_a:
        :param x_b:
        :param save_dir:
        :param mcmc_kwargs: By default:
        {num_samples=1000, warmup_steps=1000, num_chains=4)
        :return:
        """
        z_a, pi_a, marginals_a = self.compute_posteriors(x_a, mcmc_kwargs=mcmc_kwargs)
        z_b, pi_b, marginals_b = self.compute_posteriors(x_b, mcmc_kwargs=mcmc_kwargs)

        if save_dir is not None:
            np.save(os.path.join(save_dir, "xa.npy"), x_a.numpy())
            np.save(os.path.join(save_dir, "xb.npy"), x_b.numpy())
            np.save(os.path.join(save_dir, "z_a.npy"), z_a.numpy())
            np.save(os.path.join(save_dir, "z_b.npy"), z_b.numpy())
            np.save(os.path.join(save_dir, "pi_a.npy"), pi_a.numpy())
            np.save(os.path.join(save_dir, "pi_b.npy"), pi_b.numpy())

            mcmc_stats_a = marginals_a.diagnostics()
            mcmc_stats_b = marginals_b.diagnostics()

            try:
                save_figs(mcmc_stats_a, savepath=os.path.join(save_dir, "stats_a.png"))
                save_figs(mcmc_stats_b, savepath=os.path.join(save_dir, "stats_b.png"))

                n_eff_a = mcmc_stats_a["z"]["n_eff"]
                n_eff_b = mcmc_stats_b["z"]["n_eff"]
                np.save(os.path.join(save_dir, "n_eff_a.npy"), n_eff_a.numpy())
                np.save(os.path.join(save_dir, "n_eff_b.npy"), n_eff_b.numpy())

            except ValueError:
                raise Warning(
                    "Invalid values encountered in MCMC diagnostic."
                    "Please rerun experiment"
                )
