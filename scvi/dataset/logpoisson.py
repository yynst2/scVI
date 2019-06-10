import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
from pyro.infer import config_enumerate
from pyro.infer.mcmc import NUTS, MCMC
import torch
from torch import distributions, nn
from tqdm import tqdm

from .dataset import GeneExpressionDataset

dist = pyro.distributions


def save_figs(stats, savepath):
    fig, axes = plt.subplots(ncols=2, figsize=(10, 6))
    plt.sca(axes[0])
    plt.title("R Hat")
    plt.hist(stats["z"]["r_hat"].view(-1), bins=20)
    plt.sca(axes[1])
    plt.title("N effective")
    plt.hist(stats["z"]["n_eff"].view(-1), bins=20)
    plt.savefig(savepath)
    plt.close()


class LogPoissonDataset(GeneExpressionDataset):
    def __init__(
        self,
        pi=[0.7],
        n_cells=100,
        mu0_path="mu_0.npy",
        mu1_path="mu_2.npy",
        sig0_path="sigma_0.npy",
        sig1_path="sigma_2.npy",
        seed=42,
        n_genes=None,
        change_means=False,
        cuda_mcmc=False,
    ):
        torch.manual_seed(seed)
        assert len(pi) == 1
        self.probas = torch.tensor([1.0 - pi[0], pi[0]])
        self.logprobas = np.log(self.probas)
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.mu_0 = self.load_array(os.path.join(current_dir, mu0_path), n_genes)
        self.mu_1 = self.load_array(os.path.join(current_dir, mu1_path), n_genes)

        n_genes = len(self.mu_0)
        if change_means:
            self.mu_0[: n_genes // 4] = self.mu_0[: n_genes // 4] / 1.5
            self.mu_0[n_genes // 4 : n_genes // 2] = (
                self.mu_0[n_genes // 4 : n_genes // 2] / 0.5
            )

        self.sigma_0 = self.load_array(os.path.join(current_dir, sig0_path), n_genes)
        self.sigma_1 = self.load_array(os.path.join(current_dir, sig1_path), n_genes)

        d1, d2 = self.sigma_1.shape
        assert d1 == d2
        self.sigma_0 = self.sigma_0 + 2e-6 * torch.eye(d2, d2, dtype=self.sigma_0.dtype)
        self.sigma_1 = self.sigma_1 + 2e-6 * torch.eye(d2, d2, dtype=self.sigma_1.dtype)

        self.mus = torch.stack([self.mu_0, self.mu_1]).float()
        self.sigmas = torch.stack([self.sigma_0, self.sigma_1]).float()
        if cuda_mcmc:
            self.mus.cuda()
            self.sigmas.cuda()
            self.probas.cuda()
            self.logprobas.cuda()
        self.dist0 = distributions.MultivariateNormal(
            loc=self.mu_0, covariance_matrix=self.sigma_0
        )
        self.dist1 = distributions.MultivariateNormal(
            loc=self.mu_1, covariance_matrix=self.sigma_1
        )
        self.dist_x = distributions.Poisson

        cell_type = distributions.Bernoulli(probs=torch.tensor(pi)).sample((n_cells,))
        zero_mask = (cell_type == 0).squeeze()
        one_mask = ~zero_mask  # (cell_type == 1).squeeze()

        z = torch.zeros((n_cells, n_genes)).double()
        z[zero_mask] = self.dist0.sample((zero_mask.sum(),))
        z[one_mask] = self.dist1.sample((one_mask.sum(),))
        print(z.min(), z.max())
        rate = torch.clamp(z.exp(), max=1e5)
        gene_expressions = np.expand_dims(
            distributions.Poisson(rate=rate).sample(), axis=0
        )
        labels = np.expand_dims(cell_type, axis=0)
        gene_names = np.arange(n_genes).astype(str)

        print("Dataset shape: ", gene_expressions.shape)
        print(
            "Gene expressions bounds: ", gene_expressions.min(), gene_expressions.max()
        )
        super().__init__(
            *GeneExpressionDataset.get_attributes_from_list(
                gene_expressions, list_labels=labels
            ),
            gene_names=gene_names
        )

    def compute_bayes_factors(self, n_sim=10000, on="z"):
        """
        Computed numerically to gain some time
        :return:
        """
        # TODO: Derive exact value
        assert on in ["z", "exp_z"]
        res = torch.zeros(self.nb_genes, dtype=torch.int)
        for _ in tqdm(range(n_sim)):
            obs0 = self.dist0.sample()
            obs1 = self.dist1.sample()
            if on == "exp_z":
                obs0 = obs0.exp()
                obs1 = obs1.exp()
            hypothesis = (obs0 >= obs1).int()
            res += hypothesis
        p_h0 = res.double() / n_sim
        res = np.log(p_h0 + 1e-8) - np.log(1.00 - p_h0 + 1e-8)
        return pd.Series(data=res, index=self.gene_names)

    def logproba_z_fn(self, x):
        def logproba_z(z):
            if type(z) != torch.Tensor:
                z = torch.tensor(z, requires_grad=True)
            if z.grad is not None:
                z.grad.zero_()
            exp_z = z.exp()
            p_x_z = self.dist_x(rate=exp_z).log_prob(x)
            assert p_x_z.shape == x.shape
            p_x_z = p_x_z.sum()
            p_z_0 = self.logprobas[0] + self.dist0.log_prob(z)
            p_z_1 = self.logprobas[1] + self.dist1.log_prob(z)
            p_z = torch.tensor([p_z_0, p_z_1])
            p_z = torch.logsumexp(p_z, dim=0)
            res = p_x_z + p_z
            res.backward()
            grad = z.grad
            # print(grad)
            return res.item(), grad.numpy()

        return logproba_z

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
            exp_z = z.exp()
            pyro.sample("x", dist.Poisson(rate=exp_z).to_event(1), obs=data)

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

    @staticmethod
    def load_array(path, n_genes=None):
        arr = torch.tensor(np.load(path))
        if n_genes is not None:
            if arr.dim() == 1:
                arr = arr[:n_genes]
            else:
                arr = arr[:n_genes, :n_genes]
        return arr


class LatentLogPoissonDataset(GeneExpressionDataset, nn.Module):
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

        self.n_latent = n_latent
        self.cat = torch.tensor([0.5])
        self.probas = torch.tensor([1.0 - self.cat[0], self.cat[0]], requires_grad=False)
        self.logprobas = self.probas.log()
        np.random.seed(42)
        torch.manual_seed(42)

        # possible_values = torch.tensor(np.geomspace(1, 300, num=n_values))
        # mu0 = possible_values[torch.randint(high=n_values, size=(n_genes,))]
        if mu_init is None:
            mu_init = torch.randn(size=(n_latent,), requires_grad=False)
        mu0 = diff_coef * mu_init
        mu1 = -mu0

        mu0 += z_center
        mu1 += z_center
        #  *torch.randn(size=(n_latent,))

        print(mu0.shape)
        print(mu1.shape)

        # mu1 = torch.ones_like(mu0)
        self.mus = torch.stack([mu0, mu1]).float()
        sigma0 = scale_coef * torch.eye(n_latent)
        sigma1 = scale_coef * torch.eye(n_latent)
        self.sigmas = torch.stack([sigma0, sigma1]).float()

        # a_mat = torch.eye(n_genes)
        # indices = tril_indices(n_genes, n_genes, offset=-1)
        # a_mat[indices[:, 0], indices[:, 1]] = torch.rand(size=(int(n_genes*(n_genes-1)/2),))
        a_mat = torch.rand(size=(n_genes, n_latent), requires_grad=requires_grad)
        self.a_mat = a_mat

        self.dist0 = distributions.MultivariateNormal(
            loc=self.mus[0], covariance_matrix=self.sigmas[0]
        )
        self.dist1 = distributions.MultivariateNormal(
            loc=self.mus[1], covariance_matrix=self.sigmas[1]
        )
        self.dist_x = distributions.Poisson

        cell_type = distributions.Bernoulli(probs=torch.tensor(self.cat)).sample(
            (n_cells,)
        )
        zero_mask = (cell_type == 0).squeeze()
        one_mask = ~zero_mask  # (cell_type == 1).squeeze()

        z = torch.zeros((n_cells, n_latent)).float()
        z[zero_mask, :] = self.dist0.sample((zero_mask.sum(),))
        z[one_mask, :] = self.dist1.sample((one_mask.sum(),))
        self.z = z
        rate = self.compute_rate(z)
        self.h = rate

        gene_expressions = np.expand_dims(
            distributions.Poisson(rate=rate).sample(), axis=0
        )
        labels = np.expand_dims(cell_type, axis=0)
        gene_names = np.arange(n_genes).astype(str)

        super(LatentLogPoissonDataset, self).__init__(
            *GeneExpressionDataset.get_attributes_from_list(
                gene_expressions, list_labels=labels
            ),
            gene_names=gene_names
        )

    def forward(self, z: torch.Tensor, library: torch.Tensor, *cat_list: int):
        """

        :param z:
        :param library:
        :param cat_list:
        :return:
        """

        return self.compute_rate(z)

    def compute_rate(self, z: torch.Tensor):
        rate = (self.a_mat @ z.reshape(-1, self.n_latent, 1)).squeeze()
        rate = torch.clamp(rate.exp(), max=1e5)
        return rate

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
            rate = self.compute_rate(z)
            pyro.sample("x", dist.Poisson(rate=rate).to_event(1), obs=data)

    def log_p_z(self, z: torch.Tensor):
        dist0 = distributions.MultivariateNormal(
            loc=self.mus[0].to(z.device), covariance_matrix=self.sigmas[0].to(z.device)
        )
        dist1 = distributions.MultivariateNormal(
            loc=self.mus[1].to(z.device), covariance_matrix=self.sigmas[1].to(z.device)
        )
        logprobas = self.logprobas.to(z.device).view(2, -1)

        assert z.shape[-1] == self.mus.shape[-1], 'Latent representations dimensions must match'
        log_p_z_0 = dist0.log_prob(z)
        log_p_z_1 = dist1.log_prob(z)
        assert log_p_z_0.shape == log_p_z_1.shape
        assert log_p_z_1.dim() == 1
        logits = torch.stack([log_p_z_0, log_p_z_0])
        # Shape (2, N_batch)
        assert logits.shape[0] == 2

        # print('logits logprobas shapes: ', logits.shape, logprobas.shape)
        logits += logprobas
        res = torch.logsumexp(logits, dim=0)
        return res

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
