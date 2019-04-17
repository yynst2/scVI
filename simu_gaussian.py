from scvi.dataset import SyntheticGaussianDataset, SyntheticMixtureGaussianDataset
from scvi.models import LinearGaussian
from scvi.inference import GaussianTrainer
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import torch

import sys
sys.stdout = open("figures/log_gaussian_training.txt", "w")
print("STARTED TRAINING", flush=True)
# dim_z = 10
# dim_x = 100
# n_epochs = 50
# n_centers = 5
# dataset = SyntheticMixtureGaussianDataset(n_centers=n_centers, dim_z=dim_z, dim_x=dim_x, n_samples=10000, nu=1)

#########
# OLD Code
########
dim_z = 12
dim_x = 100
n_epochs = 50
dataset = SyntheticGaussianDataset(dim_z=dim_z, dim_x=dim_x, n_samples=10000, nu=1)
plt.imshow(dataset.pz_condx_var)
plt.colorbar()
plt.savefig("figures/post_covariance.png")
plt.clf()

# learn_var, wake only, sleep only, wake-sleep for each loss
scenarios = [  # WAKE updates
    (False, None, "ELBO", None),
    (False, None, "IWELBO", None),
    (False, None, "REVKL", None),
    (False, None, "CUBO", None),
    (False, None, "VRMAX", None),
    # SLEEP updates
    (False, None, None, "SLEEPKL"),
    # SAME THING BUT WITH LEARNING GEN MODEL
    # ELBO and WAKE updates
    (True, "ELBO", "ELBO", None),
    (True, "ELBO", "IWELBO", None),
    (True, "ELBO", "REVKL", None),
    (True, "ELBO", "CUBO", None),
    (True, "ELBO", "VRMAX", None),
    # IWELBO and WAKE updates
    (True, "IWELBO", "ELBO", None),
    (True, "IWELBO", "IWELBO", None),
    (True, "IWELBO", "REVKL", None),
    (True, "IWELBO", "CUBO", None),
    (True, "IWELBO", "VRMAX", None),
    # ELBO and SLEEP updates
    (True, "ELBO", None, "SLEEPKL"),
    # IWELBO and SLEEP updates
    (True, "IWELBO", None, "SLEEPKL"),
    #wAKE AND SLEEP
    (True, "ELBO", "REVKL", "SLEEPKL"),
    # IWELBO and SLEEP updates
    (True, "IWELBO", "REVKL", "SLEEPKL"),
]

for learn_var, loss_gen, loss_wvar, loss_svar in scenarios:
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
    for t in range(10):
        print(t)
        params_gen = None
        params_svar = None
        params_wvar = None
        learn_var = False

        if loss_gen is not None:
            learn_var = True

        model = LinearGaussian(dataset.A, dataset.pxz_log_det, dataset.pxz_inv_sqrt, gamma=dataset.gamma,
                               n_latent=dim_z, n_input=dim_x, learn_var=learn_var)

        trainer = GaussianTrainer(model, dataset, train_size=0.8, use_cuda=True, frequency=5)

        if loss_gen is not None:
            params_gen = [model.px_log_diag_var]
        if loss_wvar is not None:
            params_wvar = filter(lambda p: p.requires_grad, model.encoder.parameters())
        if loss_svar is not None:
            params_svar = filter(lambda p: p.requires_grad, model.encoder.parameters())

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

        gen_loss_var = np.diag(dataset.gamma) - model.get_std().cpu().detach().numpy() ** 2
        post_loss_var = np.diag(dataset.pz_condx_var) - trainer.test_set.posterior_var()

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
        # posterior query evaluation: proposal distribution
        seq_mean, seq_var, is_cdf, ess = seq.prob_eval(1000)
        plugin_cdf = norm.cdf(0, loc=seq_mean[:, 0], scale=np.sqrt(seq_var[:, 0]))

        l1_err_ex_plugin += [np.mean(np.abs(exact_cdf - plugin_cdf))]
        l1_err_ex_is += [np.mean(np.abs(exact_cdf - is_cdf))]
        l2_ess += [ess]

    print("CONFIGURATION", learn_var, loss_gen, loss_wvar, loss_svar)
    print("IWELBO:", np.mean(iwelbo), np.std(iwelbo))
    print("CUBO:", np.mean(iwelbo), np.std(iwelbo))
    print("L1 loss gen_variance_dis: ", np.mean(l1_gen_dis), np.std(l1_gen_dis))
    print("L1 loss gen_variance_sign: ", np.mean(l1_gen_sign), np.std(l1_gen_sign))
    print("L1 loss post_variance_dis: ", np.mean(l1_post_dis), np.std(l1_post_dis))
    print("L1 loss post_variance_sign: ", np.mean(l1_post_sign), np.std(l1_post_sign))
    print("AVE L1 ERROR EXACT <-> PLUGIN", np.mean(l1_err_ex_plugin), np.std(l1_err_ex_plugin))
    print("AVE L1 ERROR EXACT <-> IS", np.mean(l1_err_ex_is), np.std(l1_err_ex_is))
    print("ESS", np.mean(l2_ess), np.std(l2_ess))
    print("\n", flush=True)
