from cProfile import label
from concurrent.futures import thread
from typing import List
import numpy as np
import numba
import pandas as pd

from sklearn.tree import DecisionTreeClassifier


from lnd.algorithms import Algorithm
from lnd.data.dataset import DataSet


class PartitionFilteringDetector(Algorithm):
    def __init__(self, n_folds: int, threshold_scheme: str="majority", n_repetitions: int=10) -> None:
        self._n_folds = n_folds
        self._n_repetitions = n_repetitions
        if threshold_scheme == "majority":
            # Majority threshold scheme: A data point p is marked as mislabelled
            # if more than half of the classifiers (each trained on different subset)
            # misclassify that data point.
            self._threshold = np.floor(n_folds / 2) + 1
        elif threshold_scheme == "consensus":
            # Consensus threshold scheme: A data point p is marked as mislabelled
            # if all classifiers (training on each subset) misclassify p.
            # Called non-objection scheme in the paper.
            self._threshold = n_folds
        else:
            raise ValueError("Possible threshold methods: 'majority', 'consensus'")
    
    def run(self, data_set: DataSet) -> List[int]:
        all_mislabelled = []

        # Run the algorithm multiple times because
        # the partitioning of data is stochastic.
        for _ in range(self._n_repetitions):
            likely_mislabelled_indices = self._run_partition_filter(data_set=data_set)
            all_mislabelled.append(likely_mislabelled_indices)
        
        # Find the data points which are mislabelled in every repetition
        mislabelled_multiple_iterations = list(set.intersection(*map(set, all_mislabelled)))
        return mislabelled_multiple_iterations

    def _run_partition_filter(self, data_set: DataSet) -> List[int]:
        feature_attrs = data_set.feature_attributes
        label_attr = data_set.label_attribute

        # Local error count vector tracks whether each data point
        # in the data set is misclassified. Since the partitions
        # are disjoint, the vector becomes a binary vector. 
        local_error_count = np.zeros(data_set.size)

        # Global error count tracks the number of out-of-sample 
        # misclassifications for each data point.
        global_error_count = np.zeros(data_set.size)
        
        # Partition data into multiple subsets
        for split in data_set.split_k_fold(n_splits=self._n_folds):
            # Notice that the `test_data` contains the points in the current subset.
            df_fold = split.test_data
            
            # Train a model on the current subset
            X_subset = df_fold[feature_attrs]
            y_subset = df_fold[label_attr]

            model = DecisionTreeClassifier()
            model.fit(X_subset, y_subset)

            # Generate predictions on all data points.
            df_all = pd.concat([split.train_data, split.test_data])
            given_labels = df_all[label_attr]
            pred_labels = model.predict(X=df_all[feature_attrs])

            # Find misclassified data points.
            misclassified_indices = np.where(given_labels != pred_labels)[0]

            # Find data points in the training set which have been misclassified.
            local_misclassified_indices = [idx for idx in misclassified_indices if idx in df_fold.index]

            # Find out-of-sample data points which have been misclassified.
            global_misclassified_indices = [idx for idx in misclassified_indices if idx not in df_fold.index]

            # Increase the respective vectors.
            local_error_count[local_misclassified_indices] += 1
            global_error_count[global_misclassified_indices] += 1

        # Notice that a data point must be misclassified by a classifier
        # before it can be marked as mislabelled. Because a classifier usually
        # has higher accuracy for data points the training set.
        likely_mislabelled_indices = np.where((global_error_count >= self._threshold) & (local_error_count == 1))[0]

        return likely_mislabelled_indices


def create() -> Algorithm:
    return PartitionFilteringDetector(n_folds=10)
