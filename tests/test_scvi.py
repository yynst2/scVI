import numpy as np
import torch

from scvi.benchmark import all_benchmarks, benchmark_fish_scrna, ldvae_benchmark
from scvi.dataset import CortexDataset, SyntheticDataset, SmfishDataset
from scvi.dataset import LogPoissonDataset, PowSimSynthetic, LatentLogPoissonDataset
from scvi.inference import (
    JointSemiSupervisedTrainer,
    AlternateSemiSupervisedTrainer,
    ClassifierTrainer,
    UnsupervisedTrainer,
    AdapterTrainer,
)
from scvi.inference.annotation import compute_accuracy_rf, compute_accuracy_svc
from scvi.models import VAE, SCANVI, VAEC, LogNormalPoissonVAE
from scvi.models.classifier import Classifier
from scvi.models.modules import LinearExpLayer

use_cuda = True


def test_cortex(save_path):
    cortex_dataset = CortexDataset(save_path=save_path)
    vae = VAE(cortex_dataset.nb_genes, cortex_dataset.n_batches)
    trainer_cortex_vae = UnsupervisedTrainer(
        vae, cortex_dataset, train_size=0.5, use_cuda=use_cuda
    )
    trainer_cortex_vae.train(n_epochs=1)
    trainer_cortex_vae.train_set.reconstruction_error()
    trainer_cortex_vae.train_set.differential_expression_stats()

    trainer_cortex_vae.corrupt_posteriors(corruption="binomial")
    trainer_cortex_vae.corrupt_posteriors()
    trainer_cortex_vae.train(n_epochs=1)
    trainer_cortex_vae.uncorrupt_posteriors()

    trainer_cortex_vae.train_set.imputation_benchmark(
        n_samples=1, show_plot=False, title_plot="imputation", save_path=save_path
    )
    full = trainer_cortex_vae.create_posterior(
        vae,
        cortex_dataset,
        indices=np.arange(len(cortex_dataset))
    )
    x_new, x_old = full.generate(n_samples=10)
    assert x_new.shape == (cortex_dataset.nb_cells, cortex_dataset.nb_genes, 10)
    assert x_old.shape == (cortex_dataset.nb_cells, cortex_dataset.nb_genes)

    trainer_cortex_vae.train_set.imputation_benchmark(n_samples=1, show_plot=False,
                                                      title_plot='imputation', save_path=save_path)

    svaec = SCANVI(
        cortex_dataset.nb_genes, cortex_dataset.n_batches, cortex_dataset.n_labels
    )
    trainer_cortex_svaec = JointSemiSupervisedTrainer(
        svaec, cortex_dataset, n_labelled_samples_per_class=3, use_cuda=use_cuda
    )
    trainer_cortex_svaec.train(n_epochs=1)
    trainer_cortex_svaec.labelled_set.accuracy()
    trainer_cortex_svaec.full_dataset.reconstruction_error()

    svaec = SCANVI(
        cortex_dataset.nb_genes, cortex_dataset.n_batches, cortex_dataset.n_labels
    )
    trainer_cortex_svaec = AlternateSemiSupervisedTrainer(
        svaec, cortex_dataset, n_labelled_samples_per_class=3, use_cuda=use_cuda
    )
    trainer_cortex_svaec.train(n_epochs=1, lr=1e-2)
    trainer_cortex_svaec.unlabelled_set.accuracy()
    data_train, labels_train = trainer_cortex_svaec.labelled_set.raw_data()
    data_test, labels_test = trainer_cortex_svaec.unlabelled_set.raw_data()
    compute_accuracy_svc(
        data_train,
        labels_train,
        data_test,
        labels_test,
        param_grid=[{"C": [1], "kernel": ["linear"]}],
    )
    compute_accuracy_rf(
        data_train,
        labels_train,
        data_test,
        labels_test,
        param_grid=[{"max_depth": [3], "n_estimators": [10]}],
    )

    cls = Classifier(cortex_dataset.nb_genes, n_labels=cortex_dataset.n_labels)
    cls_trainer = ClassifierTrainer(cls, cortex_dataset)
    cls_trainer.train(n_epochs=1)
    cls_trainer.train_set.accuracy()


def test_synthetic_1():
    synthetic_dataset = SyntheticDataset()
    synthetic_dataset.cell_types = np.array(["A", "B", "C"])
    svaec = SCANVI(
        synthetic_dataset.nb_genes,
        synthetic_dataset.n_batches,
        synthetic_dataset.n_labels,
    )
    trainer_synthetic_svaec = JointSemiSupervisedTrainer(
        svaec, synthetic_dataset, use_cuda=use_cuda
    )
    trainer_synthetic_svaec.train(n_epochs=1)
    trainer_synthetic_svaec.labelled_set.entropy_batch_mixing()
    trainer_synthetic_svaec.full_dataset.knn_purity()
    trainer_synthetic_svaec.labelled_set.show_t_sne(n_samples=5)
    trainer_synthetic_svaec.unlabelled_set.show_t_sne(n_samples=5, color_by="labels")
    trainer_synthetic_svaec.labelled_set.show_t_sne(
        n_samples=5, color_by="batches and labels"
    )
    trainer_synthetic_svaec.labelled_set.clustering_scores()
    trainer_synthetic_svaec.labelled_set.clustering_scores(prediction_algorithm="gmm")
    trainer_synthetic_svaec.unlabelled_set.unsupervised_classification_accuracy()
    trainer_synthetic_svaec.unlabelled_set.differential_expression_score(
        synthetic_dataset.labels.ravel() == 1,
        synthetic_dataset.labels.ravel() == 2,
        genes=["2", "4"],
        n_samples=2,
        M_permutation=10,
    )
    trainer_synthetic_svaec.unlabelled_set.one_vs_all_degenes(
        n_samples=2, M_permutation=10
    )


def test_synthetic_2():
    synthetic_dataset = SyntheticDataset()
    vaec = VAEC(
        synthetic_dataset.nb_genes,
        synthetic_dataset.n_batches,
        synthetic_dataset.n_labels,
    )
    trainer_synthetic_vaec = JointSemiSupervisedTrainer(
        vaec,
        synthetic_dataset,
        use_cuda=use_cuda,
        frequency=1,
        early_stopping_kwargs={
            "early_stopping_metric": "reconstruction_error",
            "on": "labelled_set",
            "save_best_state_metric": "reconstruction_error",
        },
    )
    trainer_synthetic_vaec.train(n_epochs=2)


def test_fish_rna(save_path):
    gene_dataset_fish = SmfishDataset(save_path)
    gene_dataset_seq = CortexDataset(save_path=save_path, genes_to_keep=gene_dataset_fish.gene_names,
                                     total_genes=gene_dataset_fish.nb_genes + 50)
    benchmark_fish_scrna(gene_dataset_seq, gene_dataset_fish)


def base_benchmark(gene_dataset):
    vae = VAE(gene_dataset.nb_genes, gene_dataset.n_batches, gene_dataset.n_labels)
    trainer = UnsupervisedTrainer(vae, gene_dataset, train_size=0.5, use_cuda=use_cuda)
    trainer.train(n_epochs=1)
    return trainer


def test_all_benchmarks(save_path):
    all_benchmarks(n_epochs=1, save_path=save_path, show_plot=False)


def test_synthetic_3():
    gene_dataset = SyntheticDataset()
    trainer = base_benchmark(gene_dataset)
    adapter_trainer = AdapterTrainer(
        trainer.model, gene_dataset, trainer.train_set, frequency=1
    )
    adapter_trainer.train(n_path=1, n_epochs=1)


def test_nb_not_zinb():
    synthetic_dataset = SyntheticDataset()
    svaec = SCANVI(
        synthetic_dataset.nb_genes,
        synthetic_dataset.n_batches,
        synthetic_dataset.n_labels,
        labels_groups=[0, 0, 1],
        reconstruction_loss="nb",
    )
    trainer_synthetic_svaec = JointSemiSupervisedTrainer(
        svaec, synthetic_dataset, use_cuda=use_cuda
    )
    trainer_synthetic_svaec.train(n_epochs=1)


def test_classifier_accuracy(save_path):
    cortex_dataset = CortexDataset(save_path=save_path)
    cls = Classifier(cortex_dataset.nb_genes, n_labels=cortex_dataset.n_labels)
    cls_trainer = ClassifierTrainer(
        cls,
        cortex_dataset,
        metrics_to_monitor=["accuracy"],
        frequency=1,
        early_stopping_kwargs={
            "early_stopping_metric": "accuracy",
            "save_best_state_metric": "accuracy",
        },
    )
    cls_trainer.train(n_epochs=2)
    cls_trainer.train_set.accuracy()


def test_LDVAE(save_path):
    synthetic_datset_one_batch = SyntheticDataset(n_batches=1)
    ldvae_benchmark(synthetic_datset_one_batch, n_epochs=1, use_cuda=False)
    synthetic_datset_two_batches = SyntheticDataset(n_batches=2)
    ldvae_benchmark(synthetic_datset_two_batches, n_epochs=1, use_cuda=False)


def test_sampling_zl(save_path):
    cortex_dataset = CortexDataset(save_path=save_path)
    cortex_vae = VAE(cortex_dataset.nb_genes, cortex_dataset.n_batches)
    trainer_cortex_vae = UnsupervisedTrainer(
        cortex_vae, cortex_dataset, train_size=0.5, use_cuda=use_cuda
    )
    trainer_cortex_vae.train(n_epochs=2)

    cortex_cls = Classifier((cortex_vae.n_latent + 1), n_labels=cortex_dataset.n_labels)
    trainer_cortex_cls = ClassifierTrainer(
        cortex_cls, cortex_dataset, sampling_model=cortex_vae, sampling_zl=True
    )
    trainer_cortex_cls.train(n_epochs=2)
    trainer_cortex_cls.test_set.accuracy()


def test_gamma_de():
    cortex_dataset = CortexDataset()
    cortex_vae = VAE(cortex_dataset.nb_genes, cortex_dataset.n_batches)
    trainer_cortex_vae = UnsupervisedTrainer(cortex_vae, cortex_dataset, train_size=0.5,
                                             use_cuda=use_cuda)
    trainer_cortex_vae.train(n_epochs=2)

    full = trainer_cortex_vae.create_posterior(trainer_cortex_vae.model,
                                               cortex_dataset, indices=np.arange(len(cortex_dataset)))

    n_samples = 10
    M_permutation = 100
    cell_idx1 = cortex_dataset.labels.ravel() == 0
    cell_idx2 = cortex_dataset.labels.ravel() == 1

    full.differential_expression_score(cell_idx1, cell_idx2, n_samples=n_samples,
                                       M_permutation=M_permutation)
    full.differential_expression_gamma(cell_idx1, cell_idx2, n_samples=n_samples,
                                       M_permutation=M_permutation)


def test_full_cov():
    dataset = CortexDataset()
    mdl = VAE(n_input=dataset.nb_genes, n_batch=dataset.n_batches,
              reconstruction_loss='zinb', n_latent=2, full_cov=True)
    trainer = UnsupervisedTrainer(
        model=mdl,
        gene_dataset=dataset,
        use_cuda=True,
        train_size=0.7,
        frequency=1,
        early_stopping_kwargs={
            'early_stopping_metric': 'elbo',
            'save_best_state_metric': 'elbo',
            'patience': 15,
            'threshold': 3
        })
    trainer.train(n_epochs=20, lr=1e-3)
    assert not np.isnan(trainer.history['ll_test_set']).any()


def test_powsimr():
    data = PowSimSynthetic()
    assert data.X.shape == (675, 10000)

    lfc_coefs = data.lfc
    # Assert that all genes that are supposed to be differentially expressed
    # Really are
    lfc_coefs_de = lfc_coefs[data.de_genes_idx]
    is_genes_de = (lfc_coefs_de != lfc_coefs_de[:, 0].reshape((-1, 1)))[:, 1:].all(axis=1)
    assert is_genes_de.all()


def test_logpoisson():
    mu_skeletton = 'mu_{}_200genes_pbmc_diag.npy'
    sgm_skeletton = 'sigma_{}full_200genes_pbmc_diag.npy'
    dataset = LogPoissonDataset(
        mu0_path=mu_skeletton.format(0),
        mu1_path=mu_skeletton.format(1),
        sig0_path=sgm_skeletton.format(0),
        sig1_path=sgm_skeletton.format(1),
        pi=[0.5], n_cells=50
    )
    # res = dataset.compute_bayes_factors(n_sim=30)
    kwargs = {
        'early_stopping_metric': 'elbo',
        'save_best_state_metric': 'elbo',
        'patience': 15,
        'threshold': 3
    }
    VAE = LogNormalPoissonVAE(dataset.nb_genes, dataset.n_batches)
    trainer = UnsupervisedTrainer(
        model=VAE,
        gene_dataset=dataset,
        use_cuda=True,
        train_size=0.7,
        frequency=1,
        n_epochs_kl_warmup=2,
        early_stopping_kwargs=kwargs
    )
    trainer.train(n_epochs=5, lr=1e-3)
    train = trainer.train_set.sequential()
    zs, _, _ = train.get_latent()
    assert not np.isnan(zs).any()

    VAE = LogNormalPoissonVAE(
        dataset.nb_genes,
        dataset.n_batches,
        autoregressive=True,
        n_latent=5
    )
    trainer = UnsupervisedTrainer(
        model=VAE,
        gene_dataset=dataset,
        use_cuda=True,
        train_size=0.7,
        frequency=1,
        n_epochs_kl_warmup=2,
        early_stopping_kwargs=kwargs
    )
    torch.autograd.set_detect_anomaly(mode=True)

    trainer.train(n_epochs=5, lr=1e-3)
    train = trainer.train_set.sequential()
    trainer.train_set.show_t_sne(n_samples=1000, color_by='label')
    zs, _, _ = train.get_latent()
    print(zs)
    assert not np.isnan(zs).any()

    print(trainer.history)


def test_vae_ratio_loss(save_path):
    cortex_dataset = CortexDataset(save_path=save_path)
    cortex_vae = VAE(cortex_dataset.nb_genes, cortex_dataset.n_batches)
    trainer_cortex_vae = UnsupervisedTrainer(
        cortex_vae,
        cortex_dataset,
        train_size=0.5,
        use_cuda=use_cuda,
        ratio_loss=True
    )
    trainer_cortex_vae.train(n_epochs=2)

    dataset = LatentLogPoissonDataset(n_genes=5, n_latent=2, n_cells=300, n_comps=1)
    vae = LogNormalPoissonVAE(
        dataset.nb_genes,
        dataset.n_batches,
        full_cov=True
    )
    trainer_vae = UnsupervisedTrainer(
        vae,
        dataset,
        train_size=0.5,
        use_cuda=use_cuda,
        ratio_loss=True
    )
    trainer_vae.train(n_epochs=2)


def test_encoder_only():
    # torch.autograd.set_detect_anomaly(mode=True)
    dataset = LatentLogPoissonDataset(n_genes=5, n_latent=2, n_cells=300, n_comps=1)
    dataset = LatentLogPoissonDataset(n_genes=3, n_latent=2, n_cells=15, n_comps=2)
    dataset = LatentLogPoissonDataset(n_genes=5, n_latent=2, n_cells=150, n_comps=1,
                                      learn_prior_scale=True)

    # _, _, marginals = dataset.compute_posteriors(
    #     x_obs=torch.randint(0, 150, size=(1, 5), dtype=torch.float),
    #     mcmc_kwargs={"num_samples": 20, "warmup_steps": 20, "num_chains": 1}
    # )
    # stats = marginals.diagnostics()
    # print(stats)
    dataset.cuda()

    vae_mdl = LogNormalPoissonVAE(
        dataset.nb_genes,
        dataset.n_batches,
        autoregressive=False,
        full_cov=True,
        n_latent=2,
        gt_decoder=dataset.nn_model,
    )
    params = vae_mdl.encoder_params
    trainer = UnsupervisedTrainer(
        model=vae_mdl,
        gene_dataset=dataset,
        use_cuda=True,
        train_size=0.7,
        n_epochs_kl_warmup=1,
        ratio_loss=True,
        verbose=False
    )
    trainer.train(
        n_epochs=2,
        lr=1e-3,
        params=params,
    )

    full = trainer.create_posterior(trainer.model, dataset, indices=np.arange(len(dataset)))
    lkl_estimate = vae_mdl.marginal_ll(full, n_samples_mc=50)


def test_linear_exp_layer():
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    import torch.nn as nn
    torch.manual_seed(42)
    n_samples = 1000
    x_dim = 2
    a_val = [2, 1]
    x = 2.0*torch.rand(n_samples, x_dim).float() - 1.0
    a = torch.tensor(a_val).reshape(1, x_dim).float()
    b = torch.tensor([0.5]).reshape(1).float()
    y = a @ x.reshape(n_samples, x_dim, 1)
    y = y.squeeze()
    y += b
    y = y.exp()

    y += 0.001 * torch.randn_like(y)
    print(x.shape, y.shape)

    mdl = LinearExpLayer(n_in=2, n_out=1, use_batch_norm=False, dropout_rate=0.0)
    loss = nn.MSELoss()
    # loss = nn.L1Loss()
    optimizer = torch.optim.Adam(mdl.parameters(), lr=1e-2)

    losses = []
    for _ in tqdm(range(3000)):
        optimizer.zero_grad()

        preds = mdl(x).squeeze()
        loss_epoch = loss(preds, y)
        loss_epoch.backward()
        losses.append(loss_epoch)
        optimizer.step()
    plt.plot(losses)
    plt.show()
    print('MSE on weight : ', loss(a, mdl.linear_layer[0].weight))
    print('MSE on bias : ', loss(b, mdl.linear_layer[0].bias))

    print('weight : ', a, mdl.linear_layer[0].weight)
    print('bias : ', b, mdl.linear_layer[0].bias)
