import torch

from scvi.dataset import GeneExpressionDataset
from scvi.inference import UnsupervisedTrainer
from scvi.models import VAE

use_cuda = torch.cuda.is_available()

