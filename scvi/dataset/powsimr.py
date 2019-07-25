from . import GeneExpressionDataset
import numpy as np
import os
from scipy.interpolate import interp1d
import pandas as pd
import torch.distributions as distributions


batch_lfc = distributions.Normal(loc=0.0, scale=0.25)


class SignedGamma:
    def __init__(self, dim, proba_pos=0.75, shape=2, rate=4):
        self.proba_pos = proba_pos
        self.shape = shape
        self.rate = rate
        self.dim = dim

    def sample(self, size):
        if type(size) == int:
            sample_size = (size, self.dim)
        else:
            sample_size = list(size) + [self.dim]
        signs = 2.0 * distributions.Bernoulli(probs=0.75).sample(sample_size) - 1.0
        gammas = distributions.Gamma(concentration=self.shape, rate=self.rate).sample(
            sample_size
        )
        return signs * gammas


class PowSimSynthetic(GeneExpressionDataset):
    def __init__(
        self,
        cluster_to_samples=[20, 100, 30, 25, 500],
        n_genes=10000,
        real_data_path=None,
        de_p=0.1,
        de_lfc=None,
        batch_p=0.0,
        batch_lfc=None,
        batch_pattern=None,
        marker_p=0.0,
        marker_lfc=0.0,
        do_spike_in=False,
        do_downsample=False,
        geneset=False,
        cst_mu=None,
        mode="NB",
        seed=42,
    ):
        """

        :param cluster_to_samples:
        :param n_genes:
        :param de_p:
        :param de_lfc:
        :param batch_p:
        :param batch_lfc:
        :param marker_p:
        :param marker_lfc:
        :param do_spike_in:
        :param do_downsample:
        :param geneset:
        :param mode:
        :param seed:
        """
        super().__init__()
        np.random.seed(seed)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        real_data_path = os.path.join(dir_path, "kolodziejczk_param.csv")
        self.real_data_df = pd.read_csv(real_data_path)
        if de_lfc is None:
            de_lfc = SignedGamma(dim=len(cluster_to_samples))

        n_cells_total = sum(cluster_to_samples)
        self.n_clusters = len(cluster_to_samples)
        self.cluster_to_samples = cluster_to_samples

        self.phenotypes = np.concatenate(
            [
                self._get_one_hot(idx=idx, n_idx_total=self.n_clusters, size=val)
                for (idx, val) in enumerate(self.cluster_to_samples)
            ]
        )
        labels = np.array([v.argmax() for v in self.phenotypes])
        assert len(labels) == len(self.phenotypes)

        self.mode = mode
        assert self.mode in ["NB", "ZINB"]

        self.geneset = geneset
        assert not self.geneset
        self.do_downsample = do_downsample
        assert not do_downsample
        assert not do_spike_in

        # Gene expression parameters
        n_genes_de = int(n_genes * de_p)
        n_genes_batch = int(batch_p * n_genes)
        n_genes_marker = int(marker_p * n_genes)
        assert n_genes_marker == 0
        assert n_genes_batch == 0
        assert n_genes_de > 0, "No genes differentially expressed"

        # Diff exp genes
        self.de_lfc = np.zeros((n_genes, self.n_clusters))
        self.de_genes_idx = np.random.choice(a=n_genes, size=n_genes_de, replace=False)
        self.de_lfc[self.de_genes_idx] = self.unvectorize(
            de_lfc.sample((len(self.de_genes_idx),))
        )

        # Batch affected genes
        if n_genes_batch != 0:
            batch_genes_id = np.random.choice(
                a=n_genes, size=n_genes_batch, replace=False
            )
            self.batch_lfc = np.zeros(n_genes, self.n_clusters)
            self.batch_lfc[batch_genes_id] = self.unvectorize(
                batch_lfc.sample((len(batch_genes_id),))
            )
            assert batch_pattern in ["uncorrelated", "orthogonal", "correlated"]
            self.batch_pattern = batch_pattern
        else:
            self.batch_lfc = None
            self.batch_pattern = None

        # Marker genes
        if n_genes_marker != 0:
            pass
        else:
            self.marker_lfc = None
            self.ids = self.de_genes_idx
            self.lfc = self.de_lfc

        self._log_mu_mat = None
        self._mu_mat = None
        self._sizes = None

        self.cst_mu = cst_mu

        sim_data = self.generate_data(n_cells_total=n_cells_total, n_genes=n_genes)
        # sim_data[sim_data >= 1000] = 1000
        assert sim_data.shape == (n_cells_total, n_genes)

        sim_data = np.expand_dims(sim_data, axis=0)
        labels = np.expand_dims(labels, axis=0)

        gene_names = np.arange(n_genes).astype(str)

        self.populate_from_per_batch_list(
            sim_data,
            labels_per_batch=labels,
            gene_names=gene_names,
        )

        gene_data = {
            "lfc{}".format(idx): arr for (idx, arr) in enumerate(self.de_lfc.T)
        }
        self.gene_properties = pd.DataFrame(data=gene_data, index=gene_names)

    def generate_data(self, n_cells_total, n_genes):
        if self.batch_lfc is None:
            model_matrix = self.phenotypes
            coeffs = self.lfc
            batch = None
        else:
            if self.batch_pattern == "uncorrelated":
                raise NotImplementedError
            elif self.batch_pattern == "orthogonal":
                raise NotImplementedError
            else:
                raise NotImplementedError

        # Generating data based on those parameters
        if self.mode == "NB":
            new_data = self.generate_nb(model_matrix, coeffs, n_cells_total, n_genes)
        elif self.mode == "ZINB":
            new_data = self.generate_zinb(model_matrix, coeffs, n_cells_total, n_genes)
        return new_data

    def generate_nb(self, model_matrix, coeffs, n_cells_total, nb_genes):
        """

        DIFFERENCE WITH ORIGINAL IMPLEMENTATION
        HERE WE WORK WITH N_CELLS, N_GENES REPRESENTATIONS

        :param model_matrix: Mask Matrice (n_cells, n_clusters)
        :param coeffs: LFC Coefficients (n_genes, n_clusters)
        :return:
        """

        if self.cst_mu is not None:
            true_means = self.cst_mu * np.ones(nb_genes)
        else:
            mu = self.real_data_df["means"]
            true_means = np.random.choice(a=mu, size=nb_genes, replace=True)
            true_means = 1.0 / 5.0 * true_means
            true_means[true_means >= 300] = 300
            print(true_means.min(), true_means.mean(), true_means.max())

        log_mu = np.log2(1.0 + true_means)
        print(log_mu.min(), log_mu.mean(), log_mu.max())
        # log_mu = np.minimum(log_mu, 5.0)
        # NN interpolation
        interpolator_mean = interp1d(
            self.real_data_df.x,
            self.real_data_df.y,
            kind="nearest",
            bounds_error=False,
            fill_value="extrapolate",
        )
        size_mean = interpolator_mean(log_mu)
        interpolator_std = interp1d(
            self.real_data_df.x,
            self.real_data_df.sd,
            kind="nearest",
            bounds_error=False,
            fill_value="extrapolate",
        )
        size_std = interpolator_std(log_mu)

        # TODO: Verify Size
        log_sizes = np.random.normal(loc=size_mean, scale=size_std)
        assert log_sizes.shape == size_std.shape

        # all_facs = np.ones(n_samples)
        # effective_means = np.repeat(true_means, repeats=n_samples, axis=0)
        # assert effective_means.shape == (self.nb_genes, n_samples)
        effective_means = true_means  # when no size factor

        log_effective = np.log2(effective_means + 1.0)
        _perturbations = model_matrix.dot(coeffs.T)

        log_mu_mat = np.log2(effective_means + 1.0) + model_matrix.dot(coeffs.T)
        log_mu_mat[log_mu_mat < 0] = log_effective.min()

        mu_mat = 2 ** log_mu_mat
        sizes = 2 ** log_sizes

        self._log_mu_mat = log_mu_mat
        self._mu_mat = mu_mat
        self._sizes = sizes

        nb_proba = sizes / (sizes + mu_mat)
        # TODO: Verify no mistakes here
        sim_data = np.random.negative_binomial(
            n=sizes, p=nb_proba, size=(n_cells_total, nb_genes)
        )
        return sim_data

    def generate_zinb(self, model_matrix, coeffs, n_cells_total, n_genes):
        raise NotImplementedError

    @staticmethod
    def _get_one_hot(idx, n_idx_total, size):
        res = np.zeros((size, n_idx_total))
        res[:, idx] = 1
        return res

    @staticmethod
    def unvectorize(vec):
        if len(vec.shape) == 1:
            return vec.reshape((-1, 1))
        return vec
