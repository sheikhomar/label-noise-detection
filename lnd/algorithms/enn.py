from typing import List
import numpy as np
import numba

from sklearn.neighbors import KNeighborsClassifier

from lnd.algorithms import Algorithm
from lnd.data.dataset import DataSet


@numba.jit(cache=True, nopython=True, parallel=False, fastmath=True, boundscheck=False, nogil=True)
def compute_squared_euclidean_distances(target_point_index, X: np.ndarray) -> np.ndarray:
    n_points, n_dim = X.shape[0], X.shape[1]
    distances = np.zeros(n_points)
    x_target = X[target_point_index].copy()
    for i in numba.prange(n_points):
        if i != target_point_index:
            distance = 0
            for j in range(n_dim):
                distance += (X[i, j] - x_target[j])**2
            distances[i] = distance
    return distances


@numba.jit(cache=True, nopython=True, parallel=False, fastmath=True, boundscheck=False, nogil=True)
def compute_knn(target_point_index, k: int, X: np.ndarray, y: np.ndarray) -> int:
    distances = compute_squared_euclidean_distances(
        target_point_index=target_point_index,
        X=X
    )
    knn_indices = distances.argsort()[1:k+1]
    knn_classes = y[knn_indices].copy()
    
    label_votes = dict()
    selected_label = -1
    selected_label_votes = 0
    for label in knn_classes:
        if label in label_votes:
            label_votes[label] = label_votes.get(label) + 1
        else:
            label_votes[label] = 1
        if label_votes[label] > selected_label_votes:
            selected_label = label
            selected_label_votes = label_votes[label]

    return selected_label, label_votes


class EditedNearestNeighborDetector(Algorithm):
    def __init__(self, k: int) -> None:
        self._k = k

    def run(self, data_set: DataSet) -> List[int]:
        X = data_set.features
        y = data_set.labels
        n_points = X.shape[0]
        likely_mislabelled_indices = []
        for i in range(n_points):
            # knn = KNeighborsClassifier(n_neighbors=self._k)
            # knn.fit(X=np.delete(X, i, axis=0), y=np.delete(y, i))
            # pred_y = knn.predict(X[[i]])
            pred_y, _ = compute_knn(i, self._k, X, y)
            given_y = y[i]
            if pred_y != given_y:
                likely_mislabelled_indices.append(i)
        return likely_mislabelled_indices


def create() -> Algorithm:
    return EditedNearestNeighborDetector(k=11)
