import copy

import matplotlib.pyplot as plt
import torch
from . import Posterior
from . import Trainer
from scipy.stats import multivariate_normal

plt.switch_backend('agg')


class GaussianTrainer(Trainer):
    r"""UnsupervisedTrainer but for Gaussian datasets.

    Args:
        :model: A model instance from class ``VAE``, ``VAEC``, ``SCANVI``
        :dataset: A gaussian_dataset instance``
        :train_size: The train size, either a float between 0 and 1 or and integer for the number of training samples
         to use Default: ``0.8``.
        :\*\*kwargs: Other keywords arguments from the general Trainer class.
    """
    default_metrics_to_monitor = ['ll']

    def __init__(self, model, dataset, wake_loss, train_size=0.8, test_size=None, **kwargs):
        super().__init__(model, dataset, **kwargs)
        if type(self) is GaussianTrainer:
            self.train_set, self.test_set = self.train_test(model, dataset, train_size, test_size, type_class=GaussianPosterior)
            #self.train_set.to_monitor = ['ll']
            #self.test_set.to_monitor = ['ll']
            self.wake_loss = wake_loss

    @property
    def posteriors_loop(self):
        return ['train_set']

    def loss(self, tensors):
        data_tensor = torch.stack(tensors, 0)
        loss = torch.mean(self.model(data_tensor, self.wake_loss))
        return loss


class GaussianPosterior(Posterior):

    @torch.no_grad()
    def elbo(self, verbose=False):
        # Iterate once over the posterior and computes the total log_likelihood
        log_lkl = 0
        for i_batch, tensors in enumerate(self):
            data_tensor = torch.stack(tensors, 0)
            loss = self.model.neg_elbo(data_tensor)
            log_lkl += torch.sum(loss).item()
        n_samples = len(self.indices)
        ll = - log_lkl / n_samples
        if verbose:
            print("ELBO : %.4f" % ll)
        return ll

    @torch.no_grad()
    def exact_log_likelihood(self, verbose=False):
        mean = self.gene_dataset.px_mean
        cov = self.gene_dataset.px_var
        data = self.gene_dataset.X
        ll = multivariate_normal.logpdf(data, mean=mean, cov=cov).mean()
        if verbose:
            print("log p(x) : %.4f" % ll)
        return ll

    @torch.no_grad()
    def iwelbo(self, n_samples_mc, verbose=False):
        # Iterate once over the posterior and computes the total log_likelihood
        log_lkl = 0
        for i_batch, tensors in enumerate(self):
            data_tensor = torch.stack(tensors, 0)
            loss = self.model.neg_iwelbo(data_tensor, n_samples_mc)
            log_lkl += torch.sum(loss).item()
        n_samples = len(self.indices)
        ll = - log_lkl / n_samples
        if verbose:
            print("IWELBO", n_samples_mc, " : %.4f" % ll)
        return ll

    @torch.no_grad()
    def cubo(self, n_samples_mc, verbose=False):
        # Iterate once over the posterior and computes the total log_likelihood
        log_lkl = 0
        for i_batch, tensors in enumerate(self):
            data_tensor = torch.stack(tensors, 0)
            loss = self.model.cubo(data_tensor, n_samples_mc)
            log_lkl += torch.sum(loss).item()
        n_samples = len(self.indices)
        ll = log_lkl / n_samples
        if verbose:
            print("CUBO", n_samples_mc, " : %.4f" % ll)
        return ll

    @torch.no_grad()
    def VR_max(self, n_samples_mc, verbose=False):
        # Iterate once over the posterior and computes the total log_likelihood
        log_lkl = 0
        for i_batch, tensors in enumerate(self):
            data_tensor = torch.stack(tensors, 0)
            loss = self.model.VR_max(data_tensor, n_samples_mc)
            log_lkl += torch.sum(loss).item()
        n_samples = len(self.indices)
        ll = log_lkl / n_samples
        if verbose:
            print("VR_max", n_samples_mc, " : %.4f" % ll)
        return ll
