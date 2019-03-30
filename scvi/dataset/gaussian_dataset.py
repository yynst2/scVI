# -*- coding: utf-8 -*-

"""Handling datasets.
For the moment, is initialized with a torch Tensor of size (n_cells, nb_genes)"""
import copy
import os
import urllib.request
from collections import defaultdict

import numpy as np
import scipy.sparse as sp_sparse
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class GaussianDataset(Dataset):
    """
    Gaussian dataset
    """

    def __init__(self, X, px_mean, A, px_condvz_var):
        # Args:
        # Xs: a list of numpy tensors with .shape[1] identical (total_size*nb_genes)
        # or a list of scipy CSR sparse matrix,
        # or transposed CSC sparse matrix (the argument sparse must then be set to true)
        self.dense = type(X) is np.ndarray
        self._X = np.ascontiguousarray(X, dtype=np.float32) if self.dense else X
        self.nb_features = self.X.shape[1]
        self.A = A
        self.px_mean = px_mean
        self.px_condvz_var = px_condvz_var

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, X):
        self._X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return idx

    def collate_fn(self, batch):
        indexes = np.array(batch)
        X = self.X[indexes]
        return torch.from_numpy(X)


class SyntheticGaussianDataset(GaussianDataset):
    def __init__(self, dim_z=10, dim_x=100, rank_c=100, nu=0.5, n_samples=1000, seed=0):
        np.random.seed(seed)
        # Generating samples according to a Linear Gaussian system
        # mean of x
        self.px_mean = np.zeros((dim_x,))
        # conditional link
        A = 1 / np.sqrt(dim_z) * np.random.normal(size=(dim_x, dim_z))
        # conditional covar
        sqrt = 1 / np.sqrt(rank_c) * np.random.normal(size=(dim_x, rank_c))
        self.px_condvz_var = nu * np.eye(dim_x) + np.dot(sqrt.T, sqrt)

        self.px_var = self.px_condvz_var + np.dot(A, A.T)

        data = np.random.multivariate_normal(self.px_mean, self.px_var, size=(n_samples,))

        super().__init__(data, self.px_mean, A, self.px_condvz_var)
