from scvi.dataset import SyntheticGaussianDataset
dataset = SyntheticGaussianDataset(dim_x=100, n_samples=1000)
from scvi.models import LinearGaussian
model = LinearGaussian(A_param=dataset.A, px_condz_var=dataset.px_condvz_var, n_input=100)
from scvi.inference import GaussianTrainer
trainer = GaussianTrainer(model, dataset, train_size=0.8, use_cuda=True, wake_loss="ELBO")
trainer.train(n_epochs=10)
trainer.test_set.elbo(verbose=True)
trainer.test_set.iwelbo(20, verbose=True)
trainer.test_set.iwelbo(100, verbose=True)
trainer.test_set.iwelbo(1000, verbose=True)
trainer.test_set.exact_log_likelihood(verbose=True)
trainer.test_set.cubo(20, verbose=True)
trainer.test_set.cubo(100, verbose=True)
trainer.test_set.cubo(1000, verbose=True)
trainer.test_set.VR_max(1000, verbose=True)

