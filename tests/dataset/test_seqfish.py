from unittest import TestCase

from scvi.dataset import seqfish
from .utils import unsupervised_training_one_epoch


class TestSeqfishDataset(TestCase):
    def test_populate(self):
        dataset = seqfish(save_path="tests/data")
        unsupervised_training_one_epoch(dataset)
