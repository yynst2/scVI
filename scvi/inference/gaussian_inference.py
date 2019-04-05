import copy

import matplotlib.pyplot as plt
import torch
import time
from . import Posterior
from . import Trainer
from scipy.stats import multivariate_normal
from tqdm import trange
import sys
import numpy as np

plt.switch_backend('agg')


class GaussianTrainer(Trainer):
    r"""UnsupervisedTrainer but for Gaussian datasets. Also implements the wake-sleep methods

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
            self.train_set, self.test_set = self.train_test(model, dataset, train_size,
                                                            test_size, type_class=GaussianPosterior)
            self.wake_loss = wake_loss
            self.train_set.to_monitor = ['elbo']
            self.test_set.to_monitor = ['elbo']

    def train(self, n_epochs=20, lr=1e-3, eps=0.01, params_gen=None, params_var=None):
        begin = time.time()
        self.model.train()

        if params_gen is None:
            print("NO GENERATIVE MODEL")
        else:
            optimizer_gen = torch.optim.Adam(params_gen, lr=lr, eps=eps)

        optimizer_var = torch.optim.Adam(params_var, lr=lr, eps=eps)

        self.compute_metrics_time = 0
        self.n_epochs = n_epochs
        self.compute_metrics()

        with trange(n_epochs, desc="training", file=sys.stdout, disable=self.verbose) as pbar:
            # We have to use tqdm this way so it works in Jupyter notebook.
            # See https://stackoverflow.com/questions/42212810/tqdm-in-jupyter-notebook
            for self.epoch in pbar:
                self.on_epoch_begin()
                pbar.update(1)

                if params_gen is not None:
                    # WAKE PHASE for generative model
                    for tensors_list in self.data_loaders_loop():
                        loss = self.loss(*tensors_list, "ELBO")
                        optimizer_gen.zero_grad()
                        loss.backward()
                        optimizer_gen.step()

                if params_var is not None:
                    # WAKE PHASE for variational model
                    for tensors_list in self.data_loaders_loop():
                        loss = self.loss(*tensors_list, self.wake_loss)
                        optimizer_var.zero_grad()
                        loss.backward()
                        optimizer_var.step()

                if not self.on_epoch_end():
                    break

        if self.early_stopping.save_best_state_metric is not None:
            self.model.load_state_dict(self.best_state_dict)
            self.compute_metrics()

        self.model.eval()
        self.training_time += (time.time() - begin) - self.compute_metrics_time
        if self.verbose and self.frequency:
            print("\nTraining time:  %i s. / %i epochs" % (int(self.training_time), self.n_epochs))


    @property
    def posteriors_loop(self):
        return ['train_set']

    def loss(self, tensors, type):
        data_tensor = torch.stack(tensors, 0)
        loss = torch.mean(self.model(data_tensor, type))
        return loss


class GaussianPosterior(Posterior):

    @torch.no_grad()
    def elbo(self, verbose=False):
        # Iterate once over the posterior and computes the total log_likelihood
        log_lkl = 0
        for i_batch, tensors in enumerate(self):
            data_tensor = torch.stack(tensors, 0)
            loss = self.model.neg_iwelbo(data_tensor, 2)
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
            print("log p*(x) : %.4f" % ll)
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
    def vr_max(self, n_samples_mc, verbose=False):
        # Iterate once over the posterior and computes the total log_likelihood
        log_lkl = 0
        for i_batch, tensors in enumerate(self):
            data_tensor = torch.stack(tensors, 0)
            loss = self.model.vr_max(data_tensor, n_samples_mc)
            log_lkl += torch.sum(loss).item()
        n_samples = len(self.indices)
        ll = log_lkl / n_samples
        if verbose:
            print("VR_max", n_samples_mc, " : %.4f" % ll)
        return ll

    @torch.no_grad()
    def posterior_var(self):
        # Iterate once over the posterior and get the marginal variance
        ave_var = np.zeros(self.model.n_latent)
        for i_batch, tensors in enumerate(self):
            data_tensor = torch.stack(tensors, 0)
            _, _, _, qz_v, _ = self.model.inference(data_tensor)
            ave_var += torch.sum(qz_v, dim=0).cpu().detach().numpy()
        n_samples = len(self.indices)
        return ave_var / n_samples

    @torch.no_grad()
    def prob_eval(self, n_samples_mc):
        # Iterate once over the posterior and get the marginal variance
        prob = []
        qz_m = []
        qz_v = []
        ess = []
        for i_batch, tensors in enumerate(self):
            data_tensor = torch.stack(tensors, 0)
            x, y, z, t = self.model.prob_event(data_tensor, n_samples_mc)
            qz_m += [x]
            qz_v += [y]
            prob += [z]
            ess += [t]
        return np.array(torch.cat(qz_m)), np.array(torch.cat(qz_v)), \
               np.array(torch.cat(prob)), np.array(torch.cat(ess))
