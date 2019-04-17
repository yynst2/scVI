import copy

import matplotlib.pyplot as plt
import torch
import time
from tqdm import trange
import sys
from . import Trainer

plt.switch_backend('agg')


class UnsupervisedTrainer(Trainer):
    r"""The VariationalInference class for the unsupervised training of an autoencoder.

    Args:
        :model: A model instance from class ``VAE``, ``VAEC``, ``SCANVI``
        :gene_dataset: A gene_dataset instance like ``CortexDataset()``
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

    def __init__(self, model, gene_dataset, train_size=0.8, test_size=None, kl=None, **kwargs):
        super().__init__(model, gene_dataset, **kwargs)
        self.kl = kl
        if type(self) is UnsupervisedTrainer:
            self.train_set, self.test_set = self.train_test(model, gene_dataset, train_size, test_size)
            self.train_set.to_monitor = ['ll']
            self.test_set.to_monitor = ['ll']

    def train(self, n_epochs=20, lr=1e-3, eps=0.01, wake_psi="ELBO"):
        begin = time.time()
        self.model.train()

        params_gen = list(filter(lambda p: p.requires_grad, self.model.decoder.parameters())) + [self.model.px_r]
        optimizer_gen = torch.optim.Adam(params_gen, lr=lr, eps=eps)

        params_var = filter(lambda p: p.requires_grad, list(self.model.l_encoder.parameters())
                            + list(self.model.z_encoder.parameters()))
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

                # for all minibatches, update phi and psi alternately
                for tensors_list in self.data_loaders_loop():
                    sample_batch, local_l_mean, local_l_var, batch_index, _ = tensors_list[0]

                    # wake theta update
                    elbo = self.model(sample_batch, local_l_mean, local_l_var, batch_index)
                    loss = torch.mean(elbo)
                    optimizer_gen.zero_grad()
                    loss.backward()
                    optimizer_gen.step()

                    # wake theta update
                    if wake_psi == "ELBO":
                        loss = self.model(sample_batch, local_l_mean, local_l_var, batch_index)
                    if wake_psi == "CUBO":
                        loss = self.model.cubo(sample_batch, local_l_mean, local_l_var, batch_index)
                    if wake_psi == "KL":
                        loss = self.model.kl(sample_batch, local_l_mean, local_l_var, batch_index)
                    loss = torch.mean(loss)
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


class AdapterTrainer(UnsupervisedTrainer):
    def __init__(self, model, gene_dataset, posterior_test, frequency=5):
        super().__init__(model, gene_dataset, frequency=frequency)
        self.test_set = posterior_test
        self.test_set.to_monitor = ['ll']
        self.params = list(self.model.z_encoder.parameters()) + list(self.model.l_encoder.parameters())
        self.z_encoder_state = copy.deepcopy(model.z_encoder.state_dict())
        self.l_encoder_state = copy.deepcopy(model.l_encoder.state_dict())

    @property
    def posteriors_loop(self):
        return ['test_set']

    def train(self, n_path=10, n_epochs=50, **kwargs):
        for i in range(n_path):
            # Re-initialize to create new path
            self.model.z_encoder.load_state_dict(self.z_encoder_state)
            self.model.l_encoder.load_state_dict(self.l_encoder_state)
            super().train(n_epochs, params=self.params, **kwargs)

        return min(self.history["ll_test_set"])
