from typing import List
import numpy as np
import numba


from sklearn.neighbors import KNeighborsClassifier

from lnd.algorithms import Algorithm
from lnd.data.dataset import DataSet
from lnd.algorithms.enn import compute_knn, predict_knn


class BlameBasedNoiseDetector(Algorithm):
    def __init__(self, k: int) -> None:
        self._k = k
    
    def run(self, data_set: DataSet) -> List[int]:
        X = data_set.features
        y = data_set.labels
        n_points = X.shape[0]

        # Let `a` be the current point i.
        # Let `b` be one of k-nearest neighbors of i.
        #
        # We say that the data point `b` contributes to the correct
        # classification of the current data point `a`, denoted
        # Classifies(a,b), if all of the following conditions hold:
        #  1) Data point `b` is a k-nearest neighbor of `a`
        #  2) Data point `a` is correctly classified
        #  3) Data point `b` has the same label as `a`
        #
        # We say that the data point `b` contributes to the
        # incorrect classification of the current data point `a`,
        # denoted Misclassifies(a,b), if all of the following hold:
        #  1) Data point `b` is a k-nearest neighbor of `a`
        #  2) Data point `a` is misclassified
        #  3) Data point `b` has a different label than `a`
        
        # The coverage set of a given data point `b` contains
        # all data points for which point `b` contributed to their
        # correct classifications. The covarage set of a point `b`
        # measures the level of "good" caused by point `b`.
        coverage_sets = [set() for _ in range(n_points)]

        # The liability set of a given data point `b` contains
        # all data points for which point `b` contributed to their
        # incorrect classifications. The liability set of a point `b`
        # measures the level of "harm" caused by point `b`.
        liability_sets = [set() for _ in range(n_points)]

        # The dissimilarity set of a given point `b` contains
        # the nearest neighbors of `b` which contribute to
        # the misclassification of point `b`.
        dissimilarity_sets = [set() for _ in range(n_points)]
        
        # Build the different sets
        for i in range(n_points):
            pred_label, votes, knn_indices = compute_knn(i, self._k, X, y)

            for nearest_neighbor_index in knn_indices:
                
                if y[i] == pred_label: # Correctly classified
                    if y[i] == y[nearest_neighbor_index]:
                        coverage_sets[nearest_neighbor_index].add(i)
                
                else: # Misclassified
                    if y[i] != y[nearest_neighbor_index]:
                        liability_sets[nearest_neighbor_index].add(i)


                    if y[i] != y[nearest_neighbor_index]:
                        dissimilarity_sets[i].add(nearest_neighbor_index)

        liability_counts = np.array([len(lset) for lset in liability_sets])

        # The remaining set is the set of points in the original 
        # traning set sorted in descending order by the number
        # of items in the liability set. In the original paper,
        # this set is called TSet.
        remaining_set = np.argsort(liability_counts)[::-1].tolist()

        # Harmful points are data points which caused some level of harm
        # in the correct classification of other points. Basically, these
        # points have a liability set which contains at least one point.
        harmful_points = [i for i in remaining_set if liability_counts[i] > 0]

        c = harmful_points.pop(0)

        while len(liability_sets[c]) > 0:
            # At this point, we know that data point `c` causes some level
            # of harm to other points because its liability set contains at
            # least one data point. The deletion policy of BBNRv1 is 
            # as follows: if the data points in the coverage set of
            # data point `c` can still be classified correctly
            # without point `c`, then the harmful point `c` can be removed.

            # First, remove data point `c`.
            remaining_set.remove(c)

            X_filtered = X[remaining_set]
            y_filtered = y[remaining_set]

            # Check whether each point in the coverage set can
            # correctly be classified without including point `c`
            # the training set.
            is_misclassified = False
            for x in coverage_sets[c]:
                x_given_label = y[x]
                x_pred_label = predict_knn(target_point_index=x, k=self._k, X=X_filtered, y=y_filtered)
                
                if x_given_label != x_pred_label:
                    is_misclassified = True
                    break

            if is_misclassified:
                # Point `c` cannot be removed because otherwise
                # it will cause at least one of the points in its
                # coverage set to be misclassified.
                remaining_set.append(c)
            else:
                # Since the harmful point `c` does not cause any
                # misclassification, we can safely remove it.
                # BBNRv2 rebuilds the model once `c` is removed.
                points_removed += 1
                recompute_liability_sets = False
                for l in liability_sets[c].copy():
                    l_given_label = y[l]
                    l_pred_label = predict_knn(target_point_index=l, k=self._k, X=X_filtered, y=y_filtered)
                    if l_given_label == l_pred_label:
                        for d in dissimilarity_sets[l]:
                            liability_sets[d].remove(l)
                            recompute_liability_sets = True
                
                if recompute_liability_sets:
                    liability_counts = np.array([len(lset) for lset in liability_sets])
                    sorted_by_liabilities = np.argsort(liability_counts)[::-1].tolist()
                    old_remaining_set = [i for i in remaining_set]
                    remaining_set = [i for i in sorted_by_liabilities if i in old_remaining_set]
                    harmful_points = [i for i in remaining_set if liability_counts[i] > 0]

            if len(harmful_points) > 0:
                c = harmful_points.pop(0)
            else:
                break
        
        all_points = set([i for i in range(n_points)])
        removed_points = list(all_points - set(remaining_set))
        return removed_points


def create() -> Algorithm:
    return BlameBasedNoiseDetector(k=5)
