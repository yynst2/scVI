from .dataset import GeneExpressionDataset
import pandas as pd
import numpy as np
import os


class SymSimDataset(GeneExpressionDataset):
    def __init__(self, save_path):
        count_matrix = pd.read_csv(os.path.join(save_path, "obs_counts.csv"),
                                   sep=",", index_col=0).T
        label_array = pd.read_csv(os.path.join(save_path, "cellmeta.csv"),
                                  sep=",", index_col=0)["pop"].values
        gene_names = np.array(count_matrix.columns, dtype=str)
        super().__init__(*GeneExpressionDataset.get_attributes_from_matrix(
            count_matrix.values, labels=label_array, batch_indices=0), gene_names=gene_names,
            cell_types=np.unique(label_array))

        theoretical_fc = pd.read_csv(os.path.join(save_path, "theoreticalFC.csv"),
                                     sep=",", index_col=0, header=0)
        self.theorical_fc = theoretical_fc
        self.log_fc = theoretical_fc.values
        self.is_de_1 = self.log_fc >= 0.8
        self.is_de_2 = self.log_fc >= 0.6



