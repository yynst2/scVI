from .posterior import Posterior
from .trainer import Trainer
from .inference import (
    UnsupervisedTrainer,
    AdapterTrainer
)
from .annotation import (
    JointSemiSupervisedTrainer,
    SemiSupervisedTrainer,
    AlternateSemiSupervisedTrainer,
    ClassifierTrainer
)
from .fish import TrainerFish
from .gaussian_inference import GaussianTrainer

from .semi_supervised_trainer import MnistTrainer

__all__ = ['Trainer',
           'Posterior',
           'TrainerFish',
           'UnsupervisedTrainer',
           'AdapterTrainer',
           'JointSemiSupervisedTrainer',
           'SemiSupervisedTrainer',
           'AlternateSemiSupervisedTrainer',
           'ClassifierTrainer',
           'GaussianTrainer',
           'MnistTrainer'
           ]
