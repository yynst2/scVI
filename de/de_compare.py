import argparse
from utils import name_to_dataset
from de_models import ScVIClassic


def parse_args():
    parser = argparse.ArgumentParser(description='Compare methods for a given dataset')
    parser.add_argument('--dataset', type=str, help='Name of considered dataset')
    parser.add_argument('--nb_genes', type=int, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset_name = args.dataset
    nb_genes = args.nb_genes
    dataset = name_to_dataset[dataset_name]()
    if nb_genes is not None:
        dataset.subsample_genes(new_n_genes=nb_genes)

    models = [
        ('scVI_classic', ScVIClassic(dataset=dataset, reconstruction_loss='zinb', n_latent=5,
                                     full_cov=False, do_mean_variance=False)),
        ('scVI_mean_variance', ScVIClassic(dataset=dataset, reconstruction_loss='zinb', n_latent=5,
                                           full_cov=False, do_mean_variance=True)),
        ('scVI_full_covariance', ScVIClassic(dataset=dataset, reconstruction_loss='zinb',
                                             n_latent=5, full_cov=True, do_mean_variance=False))
    ]

    results = {}
    for model_name, model in models:
        model.full_init()
        model.train(lr=1e-3, n_epochs=150)

        model_perfs = model.predict_de()
        results[model_name] = model_perfs


    # mdl = VAE(n_input=dataset.nb_genes, n_batch=dataset.n_batches,
    #           reconstruction_loss='zinb', n_latent=5)
    # trainer = UnsupervisedTrainer(model=mdl, gene_dataset=dataset, use_cuda=True, train_size=1,
    #                               frequency=1, kl=1,
    #                               # early_stopping_kwargs={'early_stopping_metric': 'll',
    #                               #                        'save_best_state_metric': 'll',
    #                               #                        'patience': 15, 'threshold': 3}
    #                               )
    # trainer.train(n_epochs=150, lr=1e-3)
