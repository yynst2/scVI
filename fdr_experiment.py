import torch
import pandas as pd
import numpy as np
from scvi.dataset import SignedGamma, GeneExpressionDataset
from scvi.models import VAE
from scvi.inference import UnsupervisedTrainer
from sklearn.metrics import precision_score
from arviz.stats import psislw


def fdr_score(y_true, y_pred):
    return 1.0 - precision_score(y_true, y_pred)


def true_fdr(y_true, y_pred):
    """ 
        Computes GT FDR
    """
    n_genes = len(y_true)
    probas_sorted = np.argsort(-y_pred)
    true_fdr_arr = np.zeros(n_genes)
    for idx in range(1, len(probas_sorted) + 1):
        y_pred_tresh = np.zeros(n_genes, dtype=bool)
        where_pos = probas_sorted[:idx]
        y_pred_tresh[where_pos] = True
        # print(y_pred_tresh)
        true_fdr_arr[idx - 1] = fdr_score(y_true, y_pred_tresh)
    return true_fdr_arr


def posterior_expected_fdr(y_pred):
    """ 
        Computes posterior expected FDR 
    """
    sorted_genes = np.argsort(-y_pred)
    sorted_pgs = y_pred[sorted_genes]
    cumulative_fdr = (1.0 - sorted_pgs).cumsum() / (1.0 + np.arange(len(sorted_pgs)))
    return cumulative_fdr


N_EXPERIMENTS = 5
n_input = 28 * 28
n_labels = 9

NUMS = 10

CLASSIFICATION_RATIO = 50.0
N_EVAL_SAMPLES = 20
N_EPOCHS = 100
LR = 3e-4
BATCH_SIZE = 512
TRAIN_SIZE = 0.8

n_hidden = 128
n_latent = 10
n_layers = 1

n_cells_samples = 100
n_picks = 5
n_genes = 500
DF_PATH = "./scvi/dataset/kolodziejczk_param.csv"
# 1. Dataset generation
n0 = 500
n1 = 5000
n2 = 4500

selected = pd.read_csv(DF_PATH).sample(n_genes)
means = selected["means"].values

means[means >= 1000] = 1000

lfc_sampler = SignedGamma(dim=3, proba_pos=0.5)
lfcs = lfc_sampler.sample(n_genes).numpy()
non_de_genes = np.random.choice(n_genes, size=300)
lfcs[non_de_genes, :] = 0.0
lfcs = 2.0 * lfcs

# Constructing sigma and mus
log2_mu0 = lfcs[:, 0] + np.log2(means)
log2_mu1 = lfcs[:, 1] + np.log2(means)
log2_mu2 = lfcs[:, 2] + np.log2(means)

loge_mu0 = log2_mu0 / np.log2(np.e)
loge_mu1 = log2_mu1 / np.log2(np.e)
loge_mu2 = log2_mu2 / np.log2(np.e)

a = (2.0 * np.random.random(size=(n_genes, 1)) - 1).astype(float)
sigma = 2.0 * a.dot(a.T) + 0.5 * (
    1.0 + 0.5 * (2.0 * np.random.random(n_genes) - 1.0)
) * np.eye(n_genes)

sigma0 = 0.5 * sigma
sigma1 = sigma0
sigma2 = sigma0

# Poisson rates
h0 = torch.distributions.MultivariateNormal(
    loc=torch.tensor(loge_mu0), covariance_matrix=torch.tensor(sigma0)
).sample((n0,))
h1 = torch.distributions.MultivariateNormal(
    loc=torch.tensor(loge_mu1), covariance_matrix=torch.tensor(sigma1)
).sample((n1,))
h2 = torch.distributions.MultivariateNormal(
    loc=torch.tensor(loge_mu2), covariance_matrix=torch.tensor(sigma2)
).sample((n2,))
h = torch.cat([h0, h1, h2])

# Data sampling
x_obs = torch.distributions.Poisson(rate=h.exp()).sample()

# Zero inflation
is_zi = torch.rand_like(x_obs) <= 0.03
# print("Added zeros: ", is_zi.mean())
x_obs = x_obs * (1.0 - is_zi.double())

labels = torch.zeros((len(x_obs), 1))
labels[n0 : n0 + n1] = 1
labels[n0 + n1 :] = 2

not_null_cell = x_obs.sum(1) != 0
x_obs = x_obs[not_null_cell]
labels = labels[not_null_cell]
y = labels.numpy().squeeze()

dataset = GeneExpressionDataset(
    *GeneExpressionDataset.get_attributes_from_matrix(
        x_obs.numpy(), labels=labels.numpy().squeeze(),
    )
)


# 2. Experiment
scenarios = [  # WAKE updates
    # ( overall_training, loss_gen, loss_wvar, loss_svar, n_samples_train, n_samples_wtheta, n_samples_wphi,)
    (None, "ELBO", "CUBO", None, None, 1, 15),
    ("ELBO", "ELBO", "ELBO", None, 1, None, None),
    (None, "ELBO", "REVKL", None, None, 1, 15),
]

for scenario in scenarios:
    (
        overall_training,
        loss_gen,
        loss_wvar,
        loss_svar,
        n_samples_train,
        n_samples_wtheta,
        n_samples_wphi,
    ) = scenario

    cubo_arr = []
    iwelbo_arr = []
    khat_arr = []
    fdr_l2_err = []
    fdr_l1_err = []

    for num in range(NUMS):
        mdl = VAE(
            n_input=n_genes,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            prevent_library_saturation=True,
        )
        trainer = UnsupervisedTrainer(
            model=mdl,
            gene_daset=dataset,
            train_size=TRAIN_SIZE,
            wake_theta=loss_gen,
            wake_psi=loss_wvar,
            n_samples=n_samples_train,
        )
        try:
            trainer.train(
                n_epochs=N_EPOCHS,
                lr=LR,
                wake_theta=loss_gen,
                wake_psi=loss_wvar,
                n_samples=n_samples_train,
            )
            post = trainer.test_set.sequential()
            # *** CUBO
            cubo_loss = (
                post.getter(keys=["CUBO"], n_samples=100)["CUBO"]
                .cpu().numpy()
            )
            cubo_arr.append(cubo_loss.mean())

            # *** IWELBO
            iwelbo_loss = (
                post.getter(keys=["IWELBO"], n_samples=100)["IWELBO"]
                .cpu().numpy()
            )
            iwelbo_arr.append(iwelbo_loss.mean())

            # *** KHAT
            log_ratios = (
                post.getter(keys=["log_ratio"], n_samples=100)["log_ratio"].T
                .cpu().numpy()
            )
            assert log_ratios.shapes[0] == len(post.indices)
            _, khat_vals = psislw(log_ratios)
            khat_arr.append(khat_vals)

            # *** FDR
            post_vals = post.getter(keys=["px_scale"], n_samples=300)

            # Ground Truth
            test_indices = post.indices
            y_test = y[test_indices]
            h_test = h[test_indices]

            samples_a = np.random.choice(np.where(y_test == 0)[0], size=n_cells_samples)
            samples_b = np.random.choice(np.where(y_test == 1)[0], size=n_cells_samples)

            l2_errs = np.zeros(n_picks)
            l1_errs = np.zeros(n_picks)
            for ipick in range(n_picks):
                lfc_gt = (
                    1.0 * (np.abs(h_test[samples_a] - h_test[samples_b]) >= 0.5)
                ).mean(0)
                (lfc_gt >= 0.5).sum()

                is_de_gt = lfc_gt >= 0.5
                y_pred = (
                    (
                        (
                            post_vals["px_scale"][:, samples_a].log2()
                            - post_vals["px_scale"][:, samples_b].log2()
                        ).abs()
                        >= 0.5
                    )
                    .float()
                    .mean([0, 1])
                )
                true_fdr_arr = true_fdr(y_true=is_de_gt, y_pred=y_pred)
                pe_fdr_arr = posterior_expected_fdr(y_pred=y_pred)

                l2_err = np.linalg.norm(true_fdr_arr - pe_fdr_arr, ord=2)
                l2_errs[ipick] = l2_err
                l1_err = np.linalg.norm(true_fdr_arr - pe_fdr_arr, ord=1)
                l1_errs[ipick] = l1_err
            
            fdr_l1_err.append(l1_errs)
            fdr_l2_err.append(l2_errs)


        except Exception as e:
            print(e)
            pass
    res = {
        "wake_theta": loss_gen,
        "wake_psi": loss_wvar,
        "n_samples": n_samples_train,
        "cubo": None,
        "iwelbo": None,
        "khat": None,
        "fdr_l1": np.array(fdr_l1_err),
        "fdr_l2": np.array(fdr_l2_err),
    }
