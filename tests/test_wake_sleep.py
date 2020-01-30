import torch
import logging

from scvi.dataset import CortexDataset, MnistDataset, GeneExpressionDataset
from scvi.inference import MnistTrainer
from scvi.models import VAE, SemiSupervisedVAE
from scvi.inference import UnsupervisedTrainer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__file__)


def test_procedure(save_path):
    dataset = CortexDataset(save_path=save_path)
    mdl = VAE(n_input=dataset.nb_genes, n_batch=dataset.n_batches)
    trainer = UnsupervisedTrainer(model=mdl, gene_dataset=dataset, frequency=1)
    trainer.train_aevb(n_epochs=20, lr=1e-3, eps=0.01)
    trainer.test_set.marginal_ll(5)
    outputs = trainer.test_set.get_posterior(keys=["x", "px_scale"], n_samples=5)

    mdl = VAE(n_input=dataset.nb_genes, n_batch=dataset.n_batches)
    trainer = UnsupervisedTrainer(model=mdl, gene_dataset=dataset, frequency=1)
    trainer.train(
        n_epochs=20,
        lr=1e-3,
        eps=0.01,
        wake_theta="IWELBO",
        wake_psi="CUBO",
        n_samples=5,
    )
    trainer.train_set.get_latent()
    logger.info(trainer.history)

    for loss in ["KL", "ELBO", "CUBO"]:
        mdl = VAE(n_input=dataset.nb_genes, n_batch=dataset.n_batches)
        trainer = UnsupervisedTrainer(model=mdl, gene_dataset=dataset, frequency=1)
        trainer.train(n_epochs=20, lr=1e-3, eps=0.01, wake_psi=loss, n_samples=5)
        trainer.train_set.get_latent()
        logger.info(trainer.history)

    for loss in ["KL", "ELBO", "CUBO"]:
        mdl = VAE(n_input=dataset.nb_genes, n_batch=dataset.n_batches, iaf_t=3)
        trainer = UnsupervisedTrainer(model=mdl, gene_dataset=dataset, frequency=1)
        trainer.train(n_epochs=20, lr=1e-3, eps=0.01, wake_psi=loss, n_samples=5)
        trainer.train_set.get_latent()
        logger.info(trainer.history)


def test_full_dataset():
    full_dataset = CortexDataset()
    mdl = VAE(
        n_input=full_dataset.nb_genes,
        n_batch=full_dataset.n_batches,
        iaf_t=3,
        prevent_library_saturation=True,
    )
    trainer = UnsupervisedTrainer(model=mdl, gene_dataset=full_dataset, frequency=1)
    trainer.train(n_epochs=20, lr=1e-3, eps=0.01, wake_psi="KL", n_samples=5)
    logger.debug(trainer.history)


def test_generate_joint(save_path):
    dataset = CortexDataset(save_path=save_path)
    mdl = VAE(n_input=dataset.nb_genes, n_batch=dataset.n_batches)
    trainer = UnsupervisedTrainer(model=mdl, gene_dataset=dataset)
    trainer.train(n_epochs=20, lr=1e-3, eps=0.01, wake_psi="KL")

    for tensors_list in trainer.data_loaders_loop():
        sample_batch, local_l_mean, local_l_var, batch_index, _ = tensors_list[0]
        mdl.generate_joint(
            sample_batch, local_l_mean, local_l_var, batch_index,
        )


def test_mnist():
    n_labels = 5
    n_input = 20
    n_batch = 100
    device = "cuda"
    mdl = SemiSupervisedVAE(
        n_input=n_input,
        n_labels=n_labels,
        n_latent=5,
        n_hidden=64,
        prevent_saturation=False,
    )
    mdl = mdl.cuda()

    x = torch.rand(n_batch, n_input, device=device)
    labels = torch.randint(high=n_labels - 1, size=(n_batch,), device=device)
    variables = mdl.inference(x, y=None, n_samples=1, reparam=True)

    n_samples = 2
    variables = mdl.inference(x, y=labels, n_samples=2, reparam=True)
    everything_ok = (
        (variables["log_qc_z1"].shape == (n_samples, n_batch))
        & (variables["log_qz2_z1"].shape == (n_samples, n_batch))
        & (variables["log_pz2"].shape == (n_samples, n_batch))
        & (variables["log_pc"].shape == (n_batch,))  # for consistency
        & (variables["log_pz1_z2"].shape == (n_samples, n_batch))
        & (variables["log_px_z"].shape == (n_samples, n_batch))
    )
    assert everything_ok
    ##
    n_input = 28 * 28
    n_labels = 10
    dataset = MnistDataset(
        labelled_fraction=0.05,
        labelled_proportions=[0.1] * 10,
        root="/home/pierre/scVI/tests/mnist",
        download=True,
        do_1d=True,
        test_size=0.95,
    )

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    mdl = SemiSupervisedVAE(
        n_input=n_input,
        n_labels=n_labels,
        n_latent=5,
        n_hidden=64,
        prevent_saturation=False,
    )
    mdl = mdl.cuda()
    trainer = MnistTrainer(dataset=dataset, model=mdl, use_cuda=True)
    trainer.train(
        n_epochs=5,
        lr=1e-4,
        n_samples=5,
        overall_loss=None,
        wake_theta="ELBO",
        wake_psi="CUBO",
    )

    trainer.train(
        n_epochs=5,
        lr=1e-4,
        n_samples=3,
        overall_loss=None,
        wake_theta="ELBO",
        wake_psi="REVKL",
    )


def test_mnist_defensive():
    n_labels = 5
    n_input = 20
    n_batch = 100
    device = "cuda"
    mdl = SemiSupervisedVAE(
        n_input=n_input,
        n_labels=n_labels,
        n_latent=5,
        n_hidden=25,
        prevent_saturation=False,
        multi_encoder_keys=["CUBO", "EUBO"],
    )
    mdl = mdl.cuda()

    x = torch.rand(n_batch, n_input, device=device)
    variables = mdl.inference_defensive_sampling(
        x, y=None, counts=torch.tensor([2, 3, 3])
    )

    dataset = MnistDataset(
        labelled_fraction=0.05,
        labelled_proportions=[0.1] * 10,
        root="/home/pierre/scVI/tests/mnist",
        download=True,
        do_1d=True,
        test_size=0.95,
    )

    n_input = 28 * 28
    n_labels = 10
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    mdl = SemiSupervisedVAE(
        n_input=n_input,
        n_labels=10,
        n_latent=5,
        n_hidden=64,
        prevent_saturation=False,
        multi_encoder_keys=["CUBO", "EUBO"],
    )
    mdl = mdl.cuda()
    trainer = MnistTrainer(dataset=dataset, model=mdl, use_cuda=True)
    trainer.train_defensive(
        n_epochs=1,
        lr=1e-4,
        n_samples_theta=15,
        n_samples_phi=10,
        wake_theta="ELBO",
        cubo_wake_psi="CUBO",
        counts=torch.tensor([1, 1, 1]),
    )

    trainer.train_defensive(
        n_epochs=1,
        lr=1e-4,
        n_samples_theta=15,
        n_samples_phi=10,
        wake_theta="ELBO",
        cubo_wake_psi="CUBO",
        counts=torch.tensor([1, 0, 1]),
    )

    trainer.train_defensive(
        n_epochs=1,
        lr=1e-4,
        n_samples_theta=15,
        n_samples_phi=10,
        wake_theta="ELBO",
        cubo_wake_psi="CUBO",
        counts=torch.tensor([1, 1, 0]),
    )


def test_defensive(save_path):
    # dataset = CortexDataset(save_path=save_path)

    X_obs = torch.randint(high=1000, size=(700, 25))
    dataset = GeneExpressionDataset(
        *GeneExpressionDataset.get_attributes_from_matrix(X=X_obs.numpy())
    )

    mdl = VAE(
        n_input=dataset.nb_genes,
        n_batch=dataset.n_batches,
        multi_encoder_keys=["CUBO", "EUBO"],
    )
    trainer = UnsupervisedTrainer(model=mdl, gene_dataset=dataset, batch_size=128)
    trainer.train_defensive(
        n_epochs=20,
        lr_theta=1e-3,
        lr_phi=1e-3,
        wake_theta="IWELBO",
        wake_psi="defensive",
        n_samples_theta=15,
        n_samples_phi=10,
        counts=torch.tensor([2, 10, 0]),
    )
