from typing import List
import numpy as np
import numba

from sklearn.neighbors import KNeighborsClassifier

from lnd.algorithms import Algorithm
from lnd.data.dataset import DataSet
from lnd.algorithms.enn import predict_knn


class ConsensusAllKEditedNearestNeighborDetector(Algorithm):
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
                preds_y[j] = predict_knn(i, k, X, y)
            
            # If all k-NN rules agree on a label, we assume
            # the predict label is correct.
            top_pred_y = preds_y[0]
            if np.all(preds_y == top_pred_y):
                given_y = y[i]
                if top_pred_y != given_y:
                    likely_mislabelled_indices.append(i)
        return likely_mislabelled_indices


def create() -> Algorithm:
    return ConsensusAllKEditedNearestNeighborDetector(max_k=15)
