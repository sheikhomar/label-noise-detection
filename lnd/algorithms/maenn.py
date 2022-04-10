from typing import List
import numpy as np
import numba

from sklearn.neighbors import KNeighborsClassifier

from lnd.algorithms import Algorithm
from lnd.data.dataset import DataSet
from lnd.algorithms.enn import compute_knn


class MajorityAllKEditedNearestNeighborDetector(Algorithm):
    def __init__(self, max_k: int) -> None:
        self._max_k = max_k
    
    def run(self, data_set: DataSet) -> List[int]:
        X = data_set.features
        y = data_set.labels
        n_points = X.shape[0]
        likely_mislabelled_indices = []
        for i in range(n_points):
            k_range = list(range(1, self._max_k+1, 2))
            preds_y = np.zeros(len(k_range), dtype=np.int32)
            for j, k in enumerate(k_range):
                preds_y[j] = compute_knn(i, k, X, y)
            
            # The predicted label is a majority voting of
            # all k-NN rules.
            pred_y = np.argmax(np.bincount(preds_y))

            # Point i is likely mislabelled if the majority of
            # k-NN rules have agreed on another label than
            # the given label.
            if pred_y != y[i]:
                likely_mislabelled_indices.append(i)
        return likely_mislabelled_indices


def create() -> Algorithm:
    return MajorityAllKEditedNearestNeighborDetector(max_k=15)
