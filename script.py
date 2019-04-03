from scvi.dataset import SyntheticGaussianDataset
from scvi.models import LinearGaussian
from scvi.inference import GaussianTrainer
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import torch

learn_var = False
dim_z = 10
dim_x = 100
n_epochs = 50
dataset = SyntheticGaussianDataset(dim_z=dim_z, dim_x=dim_x, n_samples=10000, nu=1)
plt.imshow(dataset.pz_condx_var)
plt.colorbar()
plt.savefig("figures/post_covariance.png")
plt.clf()


# wake_loss = "CUBO"  # (ELBO, CUBO, REVKL)
for wake_loss in ("ELBO", "CUBO", "REVKL"):
    iwelbo = []
    cubo = []
    l1_gen_dis = []
    l1_gen_sign = []
    l1_post_dis = []
    l1_post_sign = []
    l1_err_ex_plugin = []
    l1_err_ex_is = []
    l2_ess = []
    for t in range(5):

        model = LinearGaussian(A_param=dataset.A, n_latent=dim_z, px_condz_var=dataset.px_condvz_var,
                               n_input=dim_x, learn_var=learn_var)

        if learn_var:
            params_gen = [model.px_log_diag_var]
        else:
            params_gen = None

        params_var = filter(lambda p: p.requires_grad, model.encoder.parameters())

        trainer = GaussianTrainer(model, dataset, train_size=0.8, use_cuda=True,
                                  wake_loss=wake_loss, frequency=5)

        trainer.train(n_epochs=n_epochs, params_var=params_var, params_gen=params_gen)

        ll_train_set = trainer.history["elbo_train_set"][1:]
        ll_test_set = trainer.history["elbo_test_set"][1:]
        x = np.linspace(0, n_epochs, len(ll_train_set))
        plt.plot(x, ll_train_set)
        plt.plot(x, ll_test_set)
        plt.savefig("figures/training_stats.png")
        plt.clf()

        # trainer.test_set.elbo(verbose=True)
        iwelbo += [trainer.test_set.iwelbo(100)]
        # trainer.test_set.exact_log_likelihood(verbose=True)
        cubo += [trainer.test_set.cubo(100)]

        gen_loss_var = np.diag(dataset.px_condvz_var) - model.get_std().cpu().detach().numpy() ** 2
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

    print("wake_loss", wake_loss)
    print("IWELBO:", np.mean(iwelbo), np.std(iwelbo))
    print("CUBO:", np.mean(iwelbo), np.std(iwelbo))
    print("L1 loss gen_variance_dis: ", np.mean(l1_gen_dis), np.std(l1_gen_dis))
    print("L1 loss gen_variance_sign: ", np.mean(l1_gen_sign), np.std(l1_gen_sign))
    print("L1 loss post_variance_dis: ", np.mean(l1_post_dis), np.std(l1_post_dis))
    print("L1 loss post_variance_sign: ", np.mean(l1_post_sign), np.std(l1_post_sign))
    print("AVE L1 ERROR EXACT <-> PLUGIN", np.mean(l1_err_ex_plugin), np.std(l1_err_ex_plugin))
    print("AVE L1 ERROR EXACT <-> IS", np.mean(l1_err_ex_is), np.std(l1_err_ex_is))
    print("ESS", np.mean(l2_ess), np.std(l2_ess))
