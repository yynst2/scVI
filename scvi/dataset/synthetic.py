import pickle
import os
import numpy as np
import torch.distributions as distributions

from . import GeneExpressionDataset


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


class SyntheticDataset(GeneExpressionDataset):
    def __init__(self, batch_size=200, nb_genes=100, n_batches=2, n_labels=3):
        # Generating samples according to a ZINB process
        data = np.random.negative_binomial(
            5, 0.3, size=(n_batches, batch_size, nb_genes)
        )
        mask = np.random.binomial(n=1, p=0.7, size=(n_batches, batch_size, nb_genes))
        newdata = data * mask  # We put the batch index first
        labels = np.random.randint(0, n_labels, size=(n_batches, batch_size, 1))
        super().__init__(
            *GeneExpressionDataset.get_attributes_from_list(
                newdata, list_labels=labels
            ),
            gene_names=np.arange(nb_genes).astype(np.str)
        )


class SyntheticRandomDataset(
    GeneExpressionDataset
):  # The exact random parameters taken by romain with simlr labels
    def __init__(self, mu=4.0, theta=2.0, dropout=0.7, save_path="data/"):
        np.random.seed(0)  # simlr attributes computed with this seed.
        p = mu / (mu + theta)
        r = theta
        shape_train = (2000, 10)
        l_train = np.random.gamma(r, p / (1 - p), size=shape_train)
        X_train = np.random.poisson(l_train)
        X_train *= np.random.binomial(1, 1 - dropout, size=shape_train)
        indices_to_keep = (X_train.sum(axis=1) > 0).ravel()
        X_train = X_train[indices_to_keep]
        print(
            "mu "
            + str(mu)
            + " theta "
            + str(theta)
            + " r "
            + str(r)
            + " p "
            + str(p)
            + " dropout "
            + str(dropout)
        )

        self.save_path = save_path
        self.url = (
            "https://github.com/YosefLab/scVI-data/raw/master/random_metadata.pickle"
        )
        self.download_name = "random_metadata.pickle"
        self.download()

        self.simlr_metadata = pickle.load(
            open(os.path.join(self.save_path, "random_metadata.pickle"), "rb")
        )
        labels_simlr = self.simlr_metadata["clusters"]

        super().__init__(
            *GeneExpressionDataset.get_attributes_from_matrix(
                X_train, labels=labels_simlr
            )
        )
