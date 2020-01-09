import pandas as pd
from scvi.dataset import SyntheticGaussianDataset, SyntheticMixtureGaussianDataset
from scvi.models import LinearGaussian
from scvi.models import PSIS
from scvi.inference import GaussianTrainer
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import norm
import torch
from tqdm.auto import tqdm
from arviz.stats import psislw

import sys

FILENAME = "simu_gaussian_res_paper_variable_complexity"
sys.stdout = open("figures/log_gaussian_training.txt", "w")
print("STARTED TRAINING", flush=True)
n_simu = 5

# dim_z = 10
# dim_x = 100
# n_epochs = 50
# n_centers = 5
# dataset = SyntheticMixtureGaussianDataset(n_centers=n_centers, dim_z=dim_z, dim_x=dim_x, n_samples=10000, nu=1)

#########
# OLD Code
########
dim_z = 6
dim_x = 10
n_epochs = 100
dataset = SyntheticGaussianDataset(dim_z=dim_z, dim_x=dim_x, n_samples=1000, nu=1)
plt.imshow(dataset.pz_condx_var)
plt.colorbar()
plt.savefig("figures/post_covariance.png")
plt.clf()

LINEAR_ENCODER = False

# learn_var, wake only, sleep only, wake-sleep, linear_encoder for each loss
scenarios = [  # WAKE updates
    (False, None, "ELBO", None, LINEAR_ENCODER),
    #     (False, None, "IWELBO", None),
    (False, None, "REVKL", None, LINEAR_ENCODER),
    (False, None, "CUBO", None, LINEAR_ENCODER),
    # (False, None, "VRMAX", None),
    # # SLEEP updates
    # (False, None, None, "SLEEPKL"),
    # SAME THING BUT WITH LEARNING GEN MODEL
    # (True, "ELBO", "ELBO", None, True),
    # (True, "ELBO", "IWELBO", None),
    # (True, "ELBO", "REVKL", None, True),
    # (True, "ELBO", "CUBO", None, True),
    # (True, "ELBO", "VRMAX", None),
    # (True, "ELBO", None, "SLEEPKL"),
    # (True, "ELBO", "REVKL", "SLEEPKL"),
    # IWELBO and WAKE updates
    #     (True, "IWELBO", "ELBO", None),
    # (True, "IWELBO", "IWELBO", None),
    #     (True, "IWELBO", "REVKL", None),
    #     (True, "IWELBO", "CUBO", None),
    # (True, "IWELBO", "VRMAX", None),
    # IWELBO and SLEEP updates
    # (True, "IWELBO", None, "SLEEPKL"),
    # wAKE AND SLEEP
    # IWELBO and SLEEP updates
    # (True, "IWELBO", "REVKL", "SLEEPKL"),
]

nus = np.geomspace(1e-4, 1e2, num=20)
n_hidden_ranges = [16, 32, 64, 128, 256, 512]
# n_hidden_ranges = [128]

df = []
for learn_var, loss_gen, loss_wvar, loss_svar, do_linear_encoder in scenarios:
    for n_hidden in tqdm(n_hidden_ranges):
        print(learn_var, loss_gen, loss_wvar, loss_svar)
        iwelbo = []
        cubo = []
        l1_gen_dis = []
        l1_gen_sign = []
        l1_post_dis = []
        l1_post_sign = []
        l1_err_ex_plugin = []
        l1_err_ex_is = []
        l2_ess = []
        l1_errs_is = []
        khat = []
        a_2 = []
        for t in tqdm(range(n_simu)):
            print(t)
            params_gen = None
            params_svar = None
            params_wvar = None
            learn_var = False

            if loss_gen is not None:
                learn_var = True

            model = LinearGaussian(
                dataset.A,
                dataset.pxz_log_det,
                dataset.pxz_inv_sqrt,
                gamma=dataset.gamma,
                n_latent=dim_z,
                n_input=dim_x,
                learn_var=learn_var,
                linear_encoder=do_linear_encoder,
                n_hidden=n_hidden,
            )

            trainer = GaussianTrainer(
                model, dataset, train_size=0.8, use_cuda=True, frequency=5
            )

            if loss_gen is not None:
                params_gen = [model.px_log_diag_var]
            if loss_wvar is not None:
                params_wvar = filter(
                    lambda p: p.requires_grad, model.encoder.parameters()
                )
            if loss_svar is not None:
                params_svar = filter(
                    lambda p: p.requires_grad, model.encoder.parameters()
                )

            losses = loss_gen, loss_wvar, loss_svar
            params = params_gen, params_wvar, params_svar
            trainer.train(params, losses, n_epochs=n_epochs)

            ll_train_set = trainer.history["elbo_train_set"][1:]
            ll_test_set = trainer.history["elbo_test_set"][1:]
            x = np.linspace(0, n_epochs, len(ll_train_set))
            plt.plot(x, ll_train_set)
            plt.plot(x, ll_test_set)
            plt.savefig("figures/training_stats.png")
            plt.clf()

            trainer.test_set.elbo()
            iwelbo += [trainer.test_set.iwelbo(100)]
            trainer.test_set.exact_log_likelihood()
            cubo += [trainer.test_set.cubo(100)]

            gen_loss_var = (
                np.diag(dataset.gamma) - model.get_std().cpu().detach().numpy() ** 2
            )
            if do_linear_encoder:
                post_loss_var = np.diag(
                    dataset.pz_condx_var
                    - model.encoder.var_encoder.detach().cpu().numpy()
                )
            else:
                post_loss_var = (
                    np.diag(dataset.pz_condx_var) - trainer.test_set.posterior_var()
                )

            l1_gen_dis += [np.mean(np.abs(gen_loss_var))]
            l1_gen_sign += [np.mean(np.sign(gen_loss_var))]
            # generator loss variance
            # posterior loss variance

            l1_post_dis += [np.mean(np.abs(post_loss_var))]
            l1_post_sign += [np.mean(np.sign(post_loss_var))]

            # posterior query evaluation: groundtruth
            seq = trainer.test_set.sequential(batch_size=10)
            mean = np.dot(dataset.mz_cond_x_mean, dataset.X[seq.indices, :].T)[0, :]
            std = np.sqrt(dataset.pz_condx_var[0, 0])
            exact_cdf = norm.cdf(0, loc=mean, scale=std)
            # posterior query evaluation: aproposal distribution
            seq_mean, seq_var, is_cdf, ess = seq.prob_eval(1000)
            if not do_linear_encoder:
                plugin_cdf = norm.cdf(
                    0, loc=seq_mean[:, 0], scale=np.sqrt(seq_var[:, 0])
                )
            else:
                plugin_cdf = norm.cdf(
                    0, loc=seq_mean[:, 0], scale=np.sqrt(seq_var[0, 0])
                )  # In this case the encoder learns a constant full

            l1_err_ex_plugin += [np.mean(np.abs(exact_cdf - plugin_cdf))]
            l1_err_ex_is += [np.mean(np.abs(exact_cdf - is_cdf))]
            l2_ess += [ess]

            # IS L1_err comparison
            is_cdf_nus = seq.prob_eval(1000, nu=nus)[2]
            exact_cdfs_nus = np.array(
                [norm.cdf(nu, loc=mean, scale=std) for nu in nus]
            ).T
            l1_errs_is += [np.abs(is_cdf_nus - exact_cdfs_nus).mean(0)]

            # k_hat
            # ratios = (
            #     trainer.test_set.log_ratios(n_samples_mc=100).exp().detach().numpy()
            # )
            # psis = PSIS(num_samples=100)
            # psis.fit(ratios)
            log_ratios = trainer.test_set.log_ratios(n_samples_mc=100).detach().numpy()
            # Input should be n_obs, n_samples
            log_ratios = log_ratios.T
            assert log_ratios.shape[1] == 100
            _, khat_vals = psislw(log_ratios)
            khat.append(khat_vals)

            # a norm
            # model.eval()
            gt_post_var = dataset.pz_condx_var
            sigma_sqrt = sqrtm(gt_post_var)
            if do_linear_encoder:
                var_post_var = model.encoder.var_encoder.detach().cpu().numpy()
                d_inv = np.linalg.inv(var_post_var)
                a = sigma_sqrt @ (d_inv @ sigma_sqrt) - np.eye(dim_z)
                a_2_it = np.linalg.norm(a, ord=2)
            else:
                a_2_it = np.zeros(len(seq_var))
                for it in range(len(seq_var)):
                    seq_var_item = seq_var[it]  # Posterior variance
                    d_inv = np.diag(
                        1.0 / seq_var_item
                    )  # Variationnal posterior precision
                    a = sigma_sqrt @ (d_inv @ sigma_sqrt) - np.eye(dim_z)
                    a_2_it[it] = np.linalg.norm(a, ord=2)
                a_2_it = a_2_it.mean()
            a_2.append(a_2_it)

        res = {
            "CONFIGURATION": (learn_var, loss_gen, loss_wvar, loss_svar),
            "learn_var": learn_var, 
            "loss_gen": loss_gen, 
            "loss_wvar": loss_wvar, 
            "loss_svar": loss_svar,
            "n_hidden": n_hidden,
            "IWELBO": (np.mean(iwelbo), np.std(iwelbo)),
            "CUBO": (np.mean(iwelbo), np.std(iwelbo)),
            "L1 loss gen_variance_dis": (np.mean(l1_gen_dis), np.std(l1_gen_dis)),
            "L1 loss gen_variance_sign": (np.mean(l1_gen_sign), np.std(l1_gen_sign)),
            "L1 loss post_variance_dis": (np.mean(l1_post_dis), np.std(l1_post_dis)),
            "L1 loss post_variance_sign": (np.mean(l1_post_sign), np.std(l1_post_sign)),
            "AVE L1 ERROR EXACT <-> PLUGIN": (
                np.mean(l1_err_ex_plugin),
                np.std(l1_err_ex_plugin),
            ),
            "ESS": (np.mean(l2_ess), np.std(l2_ess)),
            "L1_IS_ERRS": np.array(l1_errs_is),
            "KHAT": np.array(khat),
            "A_2": np.array(a_2),
        }
        df.append(res)

df = pd.DataFrame(df)
df.to_csv("{}.csv".format(FILENAME), sep="\t")
df.to_pickle("{}.pkl".format(FILENAME))
