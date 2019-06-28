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


def compute_rate(a_mat: torch.Tensor, b: torch.Tensor, z: torch.Tensor):
    n_latent = a_mat.shape[1]
    rate = (a_mat @ z.reshape(-1, n_latent, 1) + b.reshape(-1, 1)).squeeze()
    rate = torch.clamp(rate.exp(), max=1e5)
    return rate


@torch.no_grad()
class LatentLogPoissonModel(nn.Module):
    def __init__(
        self,
        a_mat: torch.Tensor,
        bias: torch.Tensor,
        mus: torch.Tensor,
        sigmas: torch.Tensor,
        logprobas: torch.Tensor,
        learn_prior_scale: bool,
    ):
        """
        Object to be used as Decoder for scvi.VAE class.
        Serves as non trainable, perfect decoder

        :param a_mat:
        :param bias:
        :param mus: shape (n_comps, n_latent)
        :param sigmas: shape (n_comps, n_latent, n_latent)
        :param logprobas: shape(n_comps,)
        """
        super().__init__()
        self.a_mat = nn.Parameter(a_mat)
        self.b = nn.Parameter(bias)

        assert mus.dim() == 2
        assert mus.shape[0] == sigmas.shape[0], 'Number of mixtures must match'
        self.n_comps = mus.shape[0]

        self.mus = nn.Parameter(mus)
        self.sigmas = nn.Parameter(sigmas)
        self.logprobas = nn.Parameter(logprobas)
        if learn_prior_scale:
            assert self.n_comps == 1
            self.prior_scale = nn.Parameter(torch.FloatTensor([2.5]))
        else:
            self.prior_scale = 1.0

    def forward(self, z: torch.Tensor, library: torch.Tensor, *cat_list: int):
        """

        :param z:
        :param library:
        :param cat_list:
        :return:
        """

        return compute_rate(self.a_mat, self.b, z)

    def log_p_z(self, z: torch.Tensor):
        # Case where there are 2 mixtures
        dist0 = distributions.MultivariateNormal(
            loc=self.mus[0], covariance_matrix=self.prior_scale * self.sigmas[0]
        )
        if self.n_comps == 2:
            # TODO: Generalize if useful
            assert z.dim() == 2  # Case where z has a weird shape (3 or more) not taken into account yet
            dist1 = distributions.MultivariateNormal(
                loc=self.mus[1], covariance_matrix=self.sigmas[1]
            )
            logprobas = self.logprobas.view(2, -1)

            assert (
                z.shape[-1] == self.mus.shape[-1]
            ), "Latent representations dimensions must match"
            log_p_z_0 = dist0.log_prob(z)
            log_p_z_1 = dist1.log_prob(z)
            assert log_p_z_0.shape == log_p_z_1.shape
            assert log_p_z_1.dim() == 1
            logits = torch.stack([log_p_z_0, log_p_z_1])
            # Shape (2, N_batch)
            assert logits.shape[0] == 2
            logits += logprobas
            res = torch.logsumexp(logits, dim=0)

        # Easier case when only 1 mixture
        elif self.n_comps == 1:
            res = dist0.log_prob(z)
        else:
            raise NotImplementedError
        return res


class LatentLogPoissonDataset(GeneExpressionDataset):
    def __init__(
        self,
        n_genes: int,
        n_latent: int,
        n_cells: int,
        n_comps: int = 2,
        diff_coef: float = 1.0,
        z_center: float = 1.0,
        scale_coef: float = 0.1,
        a_mat: torch.Tensor = None,
        b: torch.Tensor = None,
        mu_init: torch.Tensor = None,
        learn_prior_scale: bool = False,
        requires_grad: bool = False,
    ):
        """
        Latent Log Poisson Model:
            z ~ Mixture of 2 Gaussians
            h = log(z)
            x ~ Poisson(h)

        :param n_genes:
        :param n_latent:
        :param n_cells:
        :param n_comps: Number of mixtures in p(z)
        :param diff_coef:
        :param z_center:
        :param scale_coef:
        :param mu_init:
        :param requires_grad:
        """
        np.random.seed(42)
        torch.manual_seed(42)

        self.n_genes = n_genes
        self.n_latent = n_latent
        self.n_cells = n_cells
        self.n_comps = n_comps
        assert n_comps in [1, 2], 'More than 2 mixtures not implemented'

        self.cat = torch.tensor([0.5])
        self.probas = torch.tensor(
            [1.0 - self.cat[0], self.cat[0]], requires_grad=False
        )
        self.logprobas = self.probas.log()

        if self.n_comps == 2:
            if mu_init is None:
                mu_init = torch.randn(size=(n_latent,), requires_grad=False)
            mu0 = diff_coef * mu_init
            mu1 = -mu0
            mu0 += z_center
            mu1 += z_center

            sigma0 = scale_coef * torch.eye(n_latent)
            sigma1 = scale_coef * torch.eye(n_latent)

            dist0 = distributions.MultivariateNormal(
                loc=mu0, covariance_matrix=sigma0
            )
            dist1 = distributions.MultivariateNormal(
                loc=mu1, covariance_matrix=sigma1
            )

            self.mus = torch.stack([mu0, mu1]).float()
            self.sigmas = torch.stack([sigma0, sigma1]).float()
            self.dists = [dist0, dist1]
        elif self.n_comps == 1:
            self.mus = torch.zeros(1, n_latent)
            self.sigmas = torch.eye(n_latent).reshape(1, n_latent, n_latent)
            self.dists = [
                distributions.MultivariateNormal(loc=self.mus[0], covariance_matrix=self.sigmas[0])
            ]
        else:
            raise NotImplementedError

        if a_mat is None and b is None:
            self.a_mat = (torch.rand(size=(n_genes, n_latent), requires_grad=False) >= 0.5).float()
            self.b = 1.5 * torch.ones(size=(n_genes,), requires_grad=False)
        else:
            self.a_mat = a_mat
            self.b = b

        self.dist_x = distributions.Poisson

        self.z = None
        self.h = None
        self.generate_data()

        self.nn_model = LatentLogPoissonModel(
            a_mat=self.a_mat,
            bias=self.b,
            mus=self.mus,
            sigmas=self.sigmas,
            logprobas=self.logprobas,
            learn_prior_scale=learn_prior_scale
        )

        assert self.nn_model.n_comps == self.n_comps

    def generate_data(self):
        if self.n_comps == 2:
            cell_type = distributions.Bernoulli(probs=torch.tensor(self.cat)).sample(
                (self.n_cells,)
            )
        else:
            cell_type = torch.zeros(self.n_cells)
        z = torch.zeros((self.n_cells, self.n_latent)).float()
        for idx in range(z.shape[0]):
            z[idx, :] = self.dists[int(cell_type[idx])].sample()
        self.z = z
        rate = compute_rate(self.a_mat, self.b, z)
        self.h = rate

        gene_expressions = np.expand_dims(
            distributions.Poisson(rate=rate).sample(), axis=0
        )
        labels = np.expand_dims(cell_type, axis=0) if self.n_comps == 2 else None
        gene_names = np.arange(self.n_genes).astype(str)

        super(LatentLogPoissonDataset, self).__init__(
            *GeneExpressionDataset.get_attributes_from_list(
                gene_expressions,
                list_labels=labels
            ),
            gene_names=gene_names
        )

    @config_enumerate(default="sequential")
    def pyro_mdl(self, data: torch.Tensor):
        n_batch = data.shape[0]
        with pyro.plate("data", n_batch):
            # if self.n_comps == 2:
            #     cell_type = pyro.sample("cell_type", dist.Categorical(self.probas))
            # else:
            cell_type = 0
            z = pyro.sample(
                "z",
                dist.MultivariateNormal(
                    self.mus[cell_type], covariance_matrix=self.sigmas[cell_type]
                ),
            )
            rate = compute_rate(self.a_mat, self.b, z)
            pyro.sample("x", dist.Poisson(rate=rate).to_event(1), obs=data)

    def compute_posteriors(
        self,
        x_obs: torch.Tensor,
        mcmc_kwargs: dict = None,
        target_p: float = 0.6
    ):
        """

        :param x_obs:
        :param mcmc_kwargs: By default:
        {num_samples=1000, warmup_steps=1000, num_chains=4)
        :param target_p: Target probability for NUTS MCMC Sampler
        :return:
        """
        if mcmc_kwargs is None:
            mcmc_kwargs = {"num_samples": 1000, "warmup_steps": 1000, "num_chains": 4}
        kernel = NUTS(
            self.pyro_mdl,
            adapt_step_size=True,
            max_plate_nesting=1,
            jit_compile=True,
            target_accept_prob=target_p,
        )
        mcmc_run = MCMC(kernel, **mcmc_kwargs).run(data=x_obs)
        marginals = mcmc_run.marginal(sites=["z"])
        marginals_supp = marginals.support()
        z_x= marginals_supp["z"]
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

    def cuda(self, device=None):
        self.nn_model.cuda(device=device)

    def compute_rate(self, z: torch.Tensor):
        return compute_rate(self.a_mat, self.b, z)
