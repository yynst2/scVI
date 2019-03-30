from scvi.dataset import SyntheticGaussianDataset
from scvi.models import LinearGaussian
from scvi.inference import GaussianTrainer


learn_var = False
wake_loss = "CUBO" # (ELBO, CUBO)
dataset = SyntheticGaussianDataset(dim_x=100, n_samples=1000, nu=0.1)

model = LinearGaussian(A_param=dataset.A, px_condz_var=dataset.px_condvz_var, n_input=100, learn_var=learn_var)

if learn_var:
    params_gen = [model.px_log_diag_var]
else:
    params_gen = None

params_var = filter(lambda p: p.requires_grad, model.encoder.parameters())
trainer = GaussianTrainer(model, dataset, train_size=0.8, use_cuda=True,
                          wake_loss="CUBO")

trainer.train(n_epochs=100, params_var=params_var, params_gen=params_gen)


trainer.test_set.elbo(verbose=True)
trainer.test_set.iwelbo(20, verbose=True)
trainer.test_set.iwelbo(100, verbose=True)
trainer.test_set.iwelbo(1000, verbose=True)
trainer.test_set.exact_log_likelihood(verbose=True)
trainer.test_set.cubo(20, verbose=True)
trainer.test_set.cubo(100, verbose=True)
trainer.test_set.cubo(1000, verbose=True)
trainer.test_set.vr_max(1000, verbose=True)

print(model.get_std())

