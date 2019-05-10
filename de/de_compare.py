import argparse
import os

from utils import name_to_dataset
from de_models import ScVIClassic


def parse_args():
    parser = argparse.ArgumentParser(description='Compare methods for a given dataset')
    parser.add_argument('--dataset', type=str, help='Name of considered dataset')
    parser.add_argument('--nb_genes', type=int, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    # args = parse_args()
    # dataset_name = args.dataset
    # nb_genes = args.nb_genes

    dataset_name = "powsimr"
    nb_genes = 1200
    save_dir = '.'

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
        model_perfs.columns = [model_name+col for col in model_perfs.columns]
        results[model_name] = model_perfs

    assert dataset_name == 'powsimr'
    res_df = dataset.gene_properties

    for key in results:
        res_df = res_df.join(results[key])

    res_df.to_csv(os.path.join(save_dir, '{}_de.tsv'.format(dataset_name)), sep='\t')
