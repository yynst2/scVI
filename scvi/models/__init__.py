from .classifier import Classifier
from .scanvi import SCANVI
from .vae import VAE
from .vae_fish import VAEF
from .vaec import VAEC
from .gaussian_fixed import LinearGaussian

__all__ = ['SCANVI',
           'VAEC',
           'VAE',
           'VAEF',
           'Classifier',
           'LinearGaussian']
