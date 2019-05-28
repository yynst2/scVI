import os
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
from torch import distributions
from .dataset import GeneExpressionDataset


class LogPoissonDataset(GeneExpressionDataset):
    def __init__(
        self,
        pi=[0.7],
        n_cells=100,
        mu0_path="mu_0.npy",
        mu1_path="mu_2.npy",
        sig0_path="sigma_0.npy",
        sig1_path="sigma_2.npy",
    ):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        mu_0 = self.load_array(os.path.join(current_dir, mu0_path))
        mu_1 = self.load_array(os.path.join(current_dir, mu1_path))
        sigma_0 = self.load_array(os.path.join(current_dir, sig0_path))
        sigma_1 = self.load_array(os.path.join(current_dir, sig1_path))
        d1, d2 = sigma_1.shape
        assert d1 == d2
        sigma_0 = sigma_0 + 2e-6 * torch.eye(d2, d2, dtype=sigma_0.dtype)
        sigma_1 = sigma_1 + 2e-6 * torch.eye(d2, d2, dtype=sigma_1.dtype)
        n_genes = len(mu_0)

        self.dist0 = distributions.MultivariateNormal(
            loc=mu_0, covariance_matrix=sigma_0
        )
        self.dist1 = distributions.MultivariateNormal(
            loc=mu_1, covariance_matrix=sigma_1
        )

        cell_type = distributions.Bernoulli(probs=torch.tensor(pi)).sample((n_cells,))
        zero_mask = (cell_type == 0).squeeze()
        one_mask = (cell_type == 1).squeeze()

        z = torch.zeros((n_cells, n_genes)).double()

        z[zero_mask, :] = self.dist0.sample((zero_mask.sum(),))
        z[one_mask, :] = self.dist1.sample((one_mask.sum(),))
        print(z.min(), z.max())
        rate = torch.clamp(z.exp(), max=1e5)
        gene_expressions = np.expand_dims(
            distributions.Poisson(rate=rate).sample(), axis=0
        )
        labels = np.expand_dims(cell_type, axis=0)
        gene_names = np.arange(n_genes).astype(str)

        print(gene_expressions.shape)
        print(gene_expressions.min(), gene_expressions.max())
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

    @staticmethod
    def load_array(path):
        return torch.tensor(np.load(path))
