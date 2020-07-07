from unittest import TestCase

from scvi.dataset import retina, prefrontalcortex_starmap, frontalcortex_dropseq
from .utils import unsupervised_training_one_epoch


class TestSubDataset(TestCase):
    def test_retina_load_train_one(self):
        dataset = retina(save_path="tests/data")
        unsupervised_training_one_epoch(dataset)

    def test_pfc_starmap_load_train_one(self):
        gene_dataset = prefrontalcortex_starmap(save_path="tests/data")
        unsupervised_training_one_epoch(gene_dataset)

    def test_fc_dropseq_load_train_one(self):
        gene_dataset = frontalcortex_dropseq(save_path="tests/data")
        unsupervised_training_one_epoch(gene_dataset)
