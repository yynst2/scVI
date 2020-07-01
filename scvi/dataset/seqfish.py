import logging
import pdb
import anndata
import os
import pandas as pd

from scvi.dataset._utils import _download
from scvi.dataset import setup_anndata
from scvi.dataset.dataset import DownloadableDataset

logger = logging.getLogger(__name__)


def seqfish(save_path="data/"):
    save_path = os.path.abspath(save_path)
    url = "https://www.cell.com/cms/attachment/2080562255/2072099886/mmc6.xlsx"
    save_fn = "SeqFISH.xlsx"
    _download(url, save_path, save_fn)
    adata = _load_seqfish_data(os.path.join(save_path, save_fn))
    setup_anndata(adata)
    return adata


def _load_seqfish_data(path_to_file):
    logger.info("Loading seqfish dataset from {}".format(path_to_file))
    xl = pd.ExcelFile(path_to_file)
    counts = xl.parse("Hippocampus Counts")
    X = counts.values[:, 1:].astype(int).T  # transpose because counts is genes X cells
    pdb.set_trace()
    gene_names = counts.values[:, 0].astype(str)
    adata = anndata.AnnData(pd.DataFrame(data=X, columns=gene_names))
    logger.info("Finished loading seqfish dataset")
    return adata


class SeqfishDataset(DownloadableDataset):
    def __init__(self, save_path: str = "data/", delayed_populating: bool = False):
        super().__init__(
            urls="https://www.cell.com/cms/attachment/2080562255/2072099886/mmc6.xlsx",
            filenames="SeqFISH.xlsx",
            save_path=save_path,
            delayed_populating=delayed_populating,
        )

    def populate(self):
        logger.info("Preprocessing dataset")
        xl = pd.ExcelFile(os.path.join(self.save_path, self.filenames[0]))
        ds = xl.parse("Hippocampus Counts")  # They also used cell by genes
        logger.info("Finished preprocessing dataset")
        self.populate_from_data(ds.values[:, 1:].astype(int))
