from .classifier import Classifier
from .scanvi import SCANVI
from .vae import VAE, LDVAE
from .poisson_vae import LogNormalPoissonVAE
from .vaec import VAEC
from .modules import LinearExpLayer
from .jvae import JVAE
from .iavae import IAVAE
from .iaf_encoder import EncoderIAF

__all__ = ['SCANVI',
           'VAEC',
           'VAE',
           'LDVAE',
           'LogNormalPoissonVAE',
           'Classifier',
           'LinearExpLayer',
            "IAVAE",
           "EncoderIAF"
           ]
