from typing import List
import numpy as np
import numba

from sklearn.neighbors import KNeighborsClassifier

from lnd.algorithms import Algorithm
from lnd.data.dataset import DataSet


class EditedNearestNeighborDetector:
    def __init__(self, k: int) -> None:
        self._k = k

    def run(self, data_set: DataSet) -> List[int]:
        X = data_set.features
        y = data_set.labels
        n_points = X.shape[0]
        likely_mislabelled_indices = []
        for i in range(n_points):
            knn = KNeighborsClassifier(n_neighbors=self._k)
            knn.fit(X=np.delete(X, i, axis=0), y=np.delete(y, i))
            pred_y = knn.predict(X[[i]])
            given_y = y[i]
            if pred_y != given_y:
                likely_mislabelled_indices.append(i)
        return likely_mislabelled_indices


def create() -> Algorithm:
    return EditedNearestNeighborDetector(k=11)
