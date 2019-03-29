import copy

import matplotlib.pyplot as plt
import torch
from . import Posterior
from . import Trainer

plt.switch_backend('agg')


class GaussianTrainer(Trainer):
    r"""UnsupervisedTrainer but for Gaussian datasets.

    Args:
        :model: A model instance from class ``VAE``, ``VAEC``, ``SCANVI``
        :dataset: A gaussian_dataset instance``
        :train_size: The train size, either a float between 0 and 1 or and integer for the number of training samples
         to use Default: ``0.8``.
        :\*\*kwargs: Other keywords arguments from the general Trainer class.

    Examples:
        >>> gene_dataset = CortexDataset()
        >>> vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=gene_dataset.n_labels)

        >>> infer = VariationalInference(gene_dataset, vae, train_size=0.5)
        >>> infer.train(n_epochs=20, lr=1e-3)
    """
    default_metrics_to_monitor = ['ll']

    def __init__(self, model, dataset, train_size=0.8, test_size=None, **kwargs):
        super().__init__(model, dataset, **kwargs)
        if type(self) is GaussianTrainer:
            self.train_set, self.test_set = self.train_test(model, dataset, train_size, test_size, type_class=GaussianPosterior)
            #self.train_set.to_monitor = ['ll']
            #self.test_set.to_monitor = ['ll']

    @property
    def posteriors_loop(self):
        return ['train_set']

    def loss(self, tensors):
        data_tensor = torch.stack(tensors, 0)
        loss = torch.mean(self.model(data_tensor))
        return loss


class GaussianPosterior(Posterior):

    def elbo(self, verbose=False):
        # Iterate once over the posterior and computes the total log_likelihood
        log_lkl = 0
        for i_batch, tensors in enumerate(self):
            data_tensor = torch.stack(tensors, 0)
            loss = self.model(data_tensor)
            log_lkl += torch.sum(loss).item()
        n_samples = len(self.indices)
        ll = log_lkl / n_samples
        if verbose:
            print("LL : %.4f" % ll)
        return ll
