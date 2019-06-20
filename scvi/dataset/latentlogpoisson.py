import os

import numpy as np
import pyro
from pyro.infer import config_enumerate
from pyro.infer.mcmc import NUTS, MCMC
import torch
from torch import distributions, nn

from .dataset import GeneExpressionDataset
from .logpoisson import save_figs
dist = pyro.distributions


def compute_rate(a_mat: torch.Tensor, z: torch.Tensor):
    n_latent = a_mat.shape[1]
    rate = (a_mat @ z.reshape(-1, n_latent, 1)).squeeze()
    rate = torch.clamp(rate.exp(), max=1e5)
    return rate


@torch.no_grad()
class LatentLogPoissonModel(nn.Module):
    def __init__(self, a_mat, mus, sigmas, logprobas):
        """
        Object to be used as Decoder for scvi.VAE class.
        Serves as non trainable, perfect decoder

        :param a_mat:
        :param mus:
        :param sigmas:
        :param logprobas:
        """
        super().__init__()
        self.a_mat = nn.Parameter(a_mat)
        self.mus = nn.Parameter(mus)
        self.sigmas = nn.Parameter(sigmas)
        self.logprobas = nn.Parameter(logprobas)

    def forward(self, z: torch.Tensor, library: torch.Tensor, *cat_list: int):
        """

        :param z:
        :param library:
        :param cat_list:
        :return:
        """

        return compute_rate(self.a_mat, z)

    def log_p_z(self, z: torch.Tensor):
        dist0 = distributions.MultivariateNormal(
            loc=self.mus[0], covariance_matrix=self.sigmas[0]
        )
        dist1 = distributions.MultivariateNormal(
            loc=self.mus[1], covariance_matrix=self.sigmas[1]
        )
        logprobas = self.logprobas.view(2, -1)

        assert z.shape[-1] == self.mus.shape[-1], 'Latent representations dimensions must match'
        log_p_z_0 = dist0.log_prob(z)
        log_p_z_1 = dist1.log_prob(z)
        assert log_p_z_0.shape == log_p_z_1.shape
        assert log_p_z_1.dim() == 1
        logits = torch.stack([log_p_z_0, log_p_z_1])
        # Shape (2, N_batch)
        assert logits.shape[0] == 2
        logits += logprobas
        res = torch.logsumexp(logits, dim=0)
        return res


class LatentLogPoissonDataset(GeneExpressionDataset):
    def __init__(
        self,
        n_genes: int,
        n_latent: int,
        n_cells: int,
        diff_coef: float = 1.0,
        z_center: float = 1.0,
        scale_coef: float = 0.1,
        mu_init: torch.Tensor = None,
        requires_grad: bool = False
    ):
        """
        Latent Log Poisson Model:
            z ~ Mixture of 2 Gaussians
            h = log(z)
            x ~ Poisson(h)

        :param n_genes:
        :param n_latent:
        :param n_cells:
        :param diff_coef:
        :param z_center:
        :param scale_coef:
        :param mu_init:
        :param requires_grad:
        """
        self.n_genes = n_genes
        self.n_latent = n_latent
        self.n_cells = n_cells
        self.cat = torch.tensor([0.5])
        self.probas = torch.tensor([1.0 - self.cat[0], self.cat[0]], requires_grad=False)
        self.logprobas = self.probas.log()
        np.random.seed(42)
        torch.manual_seed(42)

        if mu_init is None:
            mu_init = torch.randn(size=(n_latent,), requires_grad=False)
        mu0 = diff_coef * mu_init
        mu1 = -mu0

        mu0 += z_center
        mu1 += z_center

        self.mus = torch.stack([mu0, mu1]).float()
        sigma0 = scale_coef * torch.eye(n_latent)
        sigma1 = scale_coef * torch.eye(n_latent)
        self.sigmas = torch.stack([sigma0, sigma1]).float()

        a_mat = torch.rand(size=(n_genes, n_latent), requires_grad=requires_grad)
        self.a_mat = a_mat

        self.dist0 = distributions.MultivariateNormal(
            loc=self.mus[0], covariance_matrix=self.sigmas[0]
        )
        self.dist1 = distributions.MultivariateNormal(
            loc=self.mus[1], covariance_matrix=self.sigmas[1]
        )
        self.dist_x = distributions.Poisson

        self.z = None
        self.h = None
        self.generate_data()

        self.nn_model = LatentLogPoissonModel(a_mat=a_mat, mus=self.mus, sigmas=self.sigmas, logprobas=self.logprobas)

    def generate_data(self):
        cell_type = distributions.Bernoulli(probs=torch.tensor(self.cat)).sample(
            (self.n_cells,)
        )
        zero_mask = (cell_type == 0).squeeze()
        one_mask = ~zero_mask  # (cell_type == 1).squeeze()

        z = torch.zeros((self.n_cells, self.n_latent)).float()
        z[zero_mask, :] = self.dist0.sample((zero_mask.sum(),))
        z[one_mask, :] = self.dist1.sample((one_mask.sum(),))
        self.z = z
        rate = compute_rate(self.a_mat, z)
        self.h = rate

        gene_expressions = np.expand_dims(
            distributions.Poisson(rate=rate).sample(), axis=0
        )
        labels = np.expand_dims(cell_type, axis=0)
        gene_names = np.arange(self.n_genes).astype(str)

        super(LatentLogPoissonDataset, self).__init__(
            *GeneExpressionDataset.get_attributes_from_list(
                gene_expressions, list_labels=labels
            ),
            gene_names=gene_names
        )

    @config_enumerate(default="sequential")
    def pyro_mdl(self, data: torch.Tensor):
        with pyro.plate("data", len(data)):
            cell_type = pyro.sample("cell_type", dist.Categorical(self.probas))
            z = pyro.sample(
                "z",
                dist.MultivariateNormal(
                    self.mus[cell_type], covariance_matrix=self.sigmas[cell_type]
                ),
            )
            rate = compute_rate(self.a_mat, z)
            pyro.sample("x", dist.Poisson(rate=rate).to_event(1), obs=data)

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
        marginals = mcmc_run.marginal(sites=["z", "cell_type"])
        marginals_supp = marginals.support()
        z_x, pi_x = marginals_supp["z"], marginals_supp["cell_type"]
        return z_x, pi_x, marginals

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
                save_figs(
                    mcmc_stats_a, savepath=os.path.join(save_dir, "stats_a.png")
                )
                save_figs(
                    mcmc_stats_b, savepath=os.path.join(save_dir, "stats_b.png")
                )

                n_eff_a = mcmc_stats_a["z"]["n_eff"]
                n_eff_b = mcmc_stats_b["z"]["n_eff"]
                np.save(os.path.join(save_dir, "n_eff_a.npy"), n_eff_a.numpy())
                np.save(os.path.join(save_dir, "n_eff_b.npy"), n_eff_b.numpy())

            except ValueError:
                raise Warning(
                    "Invalid values encountered in MCMC diagnostic."
                    "Please rerun experiment"
                )

    def cuda(self, device=None):
        self.nn_model.cuda(device=device)

    def compute_rate(self, z: torch.Tensor):
        return compute_rate(self.a_mat, z)
