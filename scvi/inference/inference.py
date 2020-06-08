import copy

import matplotlib.pyplot as plt
import torch
import logging
from scvi.inference import Trainer

plt.switch_backend("agg")
logger = logging.getLogger(__name__)


class UnsupervisedTrainer(Trainer):
    r"""The VariationalInference class for the unsupervised training of an autoencoder.

    Args:
        :model: A model instance from class ``VAE``, ``VAEC``, ``SCANVI``
        :gene_dataset: A gene_dataset instance like ``CortexDataset()``
        :train_size: The train size, either a float between 0 and 1 or and integer for the number of training samples
         to use Default: ``0.8``.
        :n_epochs_kl_warmup: Number of epochs for linear warmup of KL(q(z|x)||p(z)) term. After `n_epochs_kl_warmup`,
            the training objective is the ELBO. This might be used to prevent inactivity of latent units, and/or to
            improve clustering of latent space, as a long warmup turns the model into something more of an autoencoder.
        :\*\*kwargs: Other keywords arguments from the general Trainer class.

    Examples:
        >>> gene_dataset = CortexDataset()
        >>> vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=gene_dataset.n_labels)

        >>> infer = VariationalInference(gene_dataset, vae, train_size=0.5)
        >>> infer.train(n_epochs=20, lr=1e-3)
    """
    # default_metrics_to_monitor = ['elbo']

    def __init__(
        self,
        model,
        gene_dataset,
        train_size=0.8,
        test_size=None,
        n_epochs_kl_warmup=400,
        train_library=True,
        k: int = 1,
        loss_type="ELBO",
        test_indices=None,
        metrics: list = [],
        **kwargs
    ):
        super().__init__(model, gene_dataset, **kwargs)
        self.n_epochs_kl_warmup = n_epochs_kl_warmup
        self.loss_type = loss_type
        self.k = k
        self.train_library = train_library

        if type(self) is UnsupervisedTrainer:
            self.train_set, self.test_set = self.train_test(
                model,
                gene_dataset,
                train_size=train_size,
                test_size=test_size,
                test_indices=test_indices,
            )
            self.train_set.to_monitor = metrics
            self.test_set.to_monitor = metrics

    @property
    def posteriors_loop(self):
        return ["train_set"]

    def loss(self, tensors):
        sample_batch, local_l_mean, local_l_var, batch_index, _ = tensors
        # print(batch_index)
        loss = self.model(
            x=sample_batch,
            local_l_mean=local_l_mean,
            local_l_var=local_l_var,
            batch_index=batch_index,
            loss=self.loss_type,
            n_samples=self.k,
            train_library=self.train_library,
        )
        return loss.mean()

    def on_epoch_begin(self):
        if self.n_epochs_kl_warmup is not None:
            self.kl_weight = min(1, self.epoch / self.n_epochs_kl_warmup)
        else:
            self.kl_weight = 1.0

    # TODO: Train Wake Sleep Procedure when CUBO and everything implemented


class AdapterTrainer(UnsupervisedTrainer):
    def __init__(self, model, gene_dataset, posterior_test, frequency=5):
        super().__init__(model, gene_dataset, frequency=frequency)
        self.test_set = posterior_test
        self.test_set.to_monitor = ["elbo"]
        self.params = list(self.model.z_encoder.parameters()) + list(
            self.model.l_encoder.parameters()
        )
        self.z_encoder_state = copy.deepcopy(model.z_encoder.state_dict())
        self.l_encoder_state = copy.deepcopy(model.l_encoder.state_dict())

    @property
    def posteriors_loop(self):
        return ["test_set"]

    def train(self, n_path=10, n_epochs=50, **kwargs):
        for i in range(n_path):
            # Re-initialize to create new path
            self.model.z_encoder.load_state_dict(self.z_encoder_state)
            self.model.l_encoder.load_state_dict(self.l_encoder_state)
            super().train(n_epochs, params=self.params, **kwargs)

        return min(self.history["elbo_test_set"])
