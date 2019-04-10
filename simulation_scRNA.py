from scvi.dataset import GeneExpressionDataset
from scvi.models import VAE
from scvi.inference import UnsupervisedTrainer
import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

save_path = "/home/romain/data_chenling"

# read csv files
count_matrix = pd.read_csv(os.path.join(save_path, "obs_counts.csv"),
                           sep=",", index_col=0).T
label_array = pd.read_csv(os.path.join(save_path, "cellmeta.csv"),
                          sep=",", index_col=0)["pop"].values
gene_names = np.array(count_matrix.columns, dtype=str)

gene_dataset = GeneExpressionDataset(*GeneExpressionDataset.get_attributes_from_matrix(
    count_matrix.values, labels=label_array,
    batch_indices=0),
                                     gene_names=gene_names, cell_types=np.unique(label_array))

theoretical_FC = pd.read_csv(os.path.join(save_path, "theoreticalFC.csv"),
                             sep=",", index_col=0, header=0)
for key in theoretical_FC.columns:
    log_FC = theoretical_FC[key]
    plt.hist(log_FC)
    detected_genes = np.sum(np.abs(log_FC) >= 0.8)
    plt.title(key + ": " + str(detected_genes) + " genes / " + str(log_FC.shape[0]))
    plt.axvline(x=-0.8)
    plt.axvline(x=0.8)
    plt.savefig("figures/simulations_scRNA/" + key + ".png")
    plt.clf()

# now train scVI with all the possible parameters
vae = VAE(gene_dataset.nb_genes, dropout_rate=0.2, reconstruction_loss="zinb", n_latent=10)
trainer = UnsupervisedTrainer(vae,
                              gene_dataset,
                              train_size=0.75,
                              use_cuda=True,
                              frequency=5)

file_name = '%s/vae.pkl' % save_path
if os.path.isfile(file_name):
    print("loaded model from: " + file_name)
    trainer.model.load_state_dict(torch.load(file_name))
    trainer.model.eval()
else:
    # train & save
    n_epochs = 100
    trainer.train(n_epochs=n_epochs, lr=0.001)
    torch.save(trainer.model.state_dict(), file_name)

    # write training info
    ll_train_set = trainer.history["ll_train_set"][1:]
    ll_test_set = trainer.history["ll_test_set"][1:]
    x = np.linspace(1, n_epochs, (len(ll_train_set)))
    plt.plot(x, ll_train_set)
    plt.plot(x, ll_test_set)
    plt.title("training ll")
    plt.savefig("figures/simulations_scRNA/loss_training.png")
    plt.clf()

# get latent space
full = trainer.create_posterior(trainer.model, gene_dataset, indices=np.arange(len(gene_dataset)))
latent, batch_indices, labels = full.sequential().get_latent()
# n_samples_tsne = 4000
# full.show_t_sne(n_samples=n_samples_tsne, color_by='labels', save_name="figures/simulations_scRNA/tSNE.png")

# prepare for differential expression
cell_types = gene_dataset.cell_types
print(gene_dataset.cell_types)
couple_celltypes = (0, 1)

print("\nDifferential Expression A/B for cell types\nA: %s\nB: %s\n" %
      tuple((cell_types[couple_celltypes[i]] for i in [0, 1])))

cell_idx1 = gene_dataset.labels.ravel() == couple_celltypes[0]
cell_idx2 = gene_dataset.labels.ravel() == couple_celltypes[1]

n_samples = 100
M_permutation = 100000

de_res = full.differential_expression_score(cell_idx1, cell_idx2, M_sampling=n_samples,
                                            M_permutation=M_permutation)
print(de_res)
exit(0)
