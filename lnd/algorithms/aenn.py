from typing import List
import numpy as np
import numba

from sklearn.neighbors import KNeighborsClassifier

from lnd.algorithms import Algorithm
from lnd.data.dataset import DataSet
from lnd.algorithms.enn import predict_knn


class AllKEditedNearestNeighborDetector(Algorithm):
    def __init__(self, max_k: int) -> None:
        self._max_k = max_k
    
    def run(self, data_set: DataSet) -> List[int]:
        X = data_set.features
        y = data_set.labels
        n_points = X.shape[0]
        likely_mislabelled_indices = []
        for k in range(1, self._max_k+1):
            print(f"Running for k={k}")
            for i in range(n_points):
                pred_y = predict_knn(i, k, X, y)
                given_y = y[i]
                if pred_y != given_y:
                    likely_mislabelled_indices.append(i)
        return likely_mislabelled_indices


def create() -> Algorithm:
    return AllKEditedNearestNeighborDetector(max_k=11)
