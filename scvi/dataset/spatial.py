import logging
from typing import List, Union, Dict

import numpy as np
import scipy.sparse as sp_sparse
from scvi.dataset.dataset import GeneExpressionDataset, CellMeasurement
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class SpatialDataset(GeneExpressionDataset):
    """
    This is a subclass specifically tailored to learn a correlated VAE
    """

    def __init__(self):
        super().__init__()
        self.coords = None

    def populate_from_spatial_data(
        self,
        X: Union[np.ndarray, sp_sparse.csr_matrix],
        coords: np.ndarray,
        Ys: List[CellMeasurement] = None,
        batch_indices: Union[List[int], np.ndarray, sp_sparse.csr_matrix] = None,
        labels: Union[List[int], np.ndarray, sp_sparse.csr_matrix] = None,
        gene_names: Union[List[str], np.ndarray] = None,
        cell_types: Union[List[str], np.ndarray] = None,
        cell_attributes_dict: Dict[str, Union[List, np.ndarray]] = None,
        gene_attributes_dict: Dict[str, Union[List, np.ndarray]] = None,
        remap_attributes: bool = True,
    ):
        self.populate_from_data(
            X=X,
            Ys=Ys,
            batch_indices=batch_indices,
            labels=labels,
            gene_names=gene_names,
            cell_types=cell_types,
            cell_attributes_dict=cell_attributes_dict,
            gene_attributes_dict=gene_attributes_dict,
            remap_attributes=remap_attributes,
        )
        columns = ["x", "y", "z"][: coords.shape[0]]
        self.initialize_cell_measurement(
            CellMeasurement(
                name="coords", data=coords, columns_attr_name="axis", columns=columns
            )
        )

    def populate_from_dataset(
        self, dataset: GeneExpressionDataset, coords: np.ndarray = None
    ):
        """
        Idiotic but useful for compatibility with old code.
        :param dataset: dataset with or without spatial components
        :param coords: additional spatial components (if not included in dataset)
        """
        if not hasattr(GeneExpressionDataset, "coords") and coords is None:
            raise ValueError(
                'If coords is None, dataset should have a "coords" attribute.'
            )
        for attr_name, attr_value in dataset.__dict__.items():
            # duck typing for properties with no setter
            try:
                setattr(self, attr_name, attr_value)
            except AttributeError:
                pass

    def equip(self, k_neighbors: int = 8):
        # create the nearest neighbors tree
        kd_tree = NearestNeighbors().fit(self.coords)
        _, indices = kd_tree.kneighbors(self.coords, n_neighbors=k_neighbors + 1)
        edge_indices = indices[:, 1:]
        self.initialize_cell_measurement(
            CellMeasurement(
                name="edge_indices",
                data=edge_indices,
                columns_attr_name="neighbors",
                columns=["k" + str(i) for i in range(k_neighbors)],
            )
        )

        # fetch NN gene expression
        self.initialize_cell_measurement(
            CellMeasurement(
                name="neighbor_scrna",
                data=self.X.astype(np.float32)[self.edge_indices],
                columns_attr_name="neighbors",
                columns=["k" + str(i) for i in range(k_neighbors)],
            )
        )

        # computing the edge weights
        graph_adjacency = kd_tree.kneighbors_graph(
            self.coords, n_neighbors=k_neighbors + 1
        ).toarray()
        graph_adjacency = graph_adjacency - np.eye(graph_adjacency.shape[0])
        laplacian = np.diag(graph_adjacency.sum(axis=0)) - graph_adjacency
        inv_laplacian = np.linalg.pinv(laplacian)
        edge_weights = np.zeros((edge_indices.shape[0], k_neighbors))
        for i in range(edge_indices.shape[0]):
            for k in range(k_neighbors):
                j = indices[i, k + 1]
                edge_weights[i, k] = (
                    inv_laplacian[i, i]
                    - inv_laplacian[i, j]
                    - inv_laplacian[j, i]
                    + inv_laplacian[j, j]
                )
        self.initialize_cell_measurement(
            CellMeasurement(
                name="edge_weights",
                data=edge_weights,
                columns_attr_name="neighbors",
                columns=["w" + str(i) for i in range(k_neighbors)],
            )
        )

        # we use a vector containing the proportion of each cell type within the cell's neighborhood
        # features = np.zeros(shape=(self.nb_cells, self.n_labels), dtype=np.float32)
        # for cell in range(self.nb_cells):
        #     labels, counts = np.unique(
        #         self.labels[self.indices[cell]], return_counts=True
        #     )
        #     normalizer = 1 if features_type == "cell type counts" else k_neighbors
        #     features[cell, labels] = counts / normalizer
        # features_column_descriptor = "cell_type_proportions"
        # self.initialize_cell_measurement(
        #     CellMeasurement(
        #         name="features",
        #         data=features,
        #         columns_attr_name=features_column_descriptor,
        #         columns=list(range(features.shape[1])),
        #     )
        # )
