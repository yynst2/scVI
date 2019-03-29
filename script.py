from scvi.dataset import SyntheticGaussianDataset
dataset = SyntheticGaussianDataset(dim_x=100, n_samples=1000)
from scvi.models import LinearGaussian
model = LinearGaussian(A_param=dataset.A, px_condz_var=dataset.px_condvz_var, n_input=100)
from scvi.inference import GaussianTrainer
trainer = GaussianTrainer(model, dataset, train_size=0.8, use_cuda=True)
trainer.train(n_epochs=1)
trainer.test_set.elbo()
