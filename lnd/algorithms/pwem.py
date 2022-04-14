from typing import List

import numpy as np
import numba
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

from lnd.algorithms import Algorithm
from lnd.data.dataset import DataSet
from lnd.algorithms.enn import compute_knn


class PairwiseExpectationMaximization(Algorithm):
    def __init__(self) -> None:
        pass
    
    def run(self, data_set: DataSet) -> List[int]:
        confidence_scores = self.compute_confidence_scores(data_set=data_set)

        given_labels = data_set.labels
        pred_labels = np.argmax(confidence_scores, axis=1)

        likely_mislabelled_indices = np.where(given_labels != pred_labels)[0]
        
        return likely_mislabelled_indices

    def compute_confidence_scores(self, data_set: DataSet) -> np.ndarray:
        scores = np.zeros(shape=(data_set.size, data_set.n_classes))
        for split in data_set.split_one_vs_one():
            print(f"Train GMM for split {split.split_no}")
            feature_transformer = data_set.create_feature_transformer()
            label_transformer = LabelEncoder()

            df_combined_data = pd.concat([split.class1_data, split.class2_data])
            X = feature_transformer.fit_transform(df_combined_data)
            y = label_transformer.fit_transform(df_combined_data[data_set.label_attribute])

            gmm = GaussianMixture(n_components=5,) #random_state=42)
            gmm.fit(X)

            # Generate cluster labels
            cluster_labels = gmm.predict(X)

            # A GMM model provides the probability of a data point x belonging to
            # cluster j in split s i.e., P(cluster = j | data_point = x, data_split = s)
            proba_cluster_given_point = gmm.predict_proba(X)

            # Compute the probability of a label L given a cluster j in split s:
            #  P(label = L | cluster = j, data_split = s)
            proba_label_given_cluster = self.compute_probability_of_label_given_cluster(
                labels=y,
                cluster_labels=cluster_labels
            )

            # Assign a uniform prior of clustering as we assume that each
            # clustering model is equally likely
            prior_proba_clustering = 1 / (data_set.n_classes - 1)

            # Compute confidence score for the points in the current split
            confidences_binary = np.dot(proba_cluster_given_point, proba_label_given_cluster)
            confidences_class1 = confidences_binary[:,0]
            confidences_class2 = confidences_binary[:,1]

            # Sum up confidence score weighted by the prior probability of clustering
            scores[df_combined_data.index, split.class1_index] += confidences_class1 * prior_proba_clustering
            scores[df_combined_data.index, split.class2_index] += confidences_class2 * prior_proba_clustering

        return scores
    
    def compute_probability_of_label_given_cluster(self, labels: np.ndarray, cluster_labels: np.ndarray) -> pd.DataFrame:
        df_results = pd.DataFrame()
        df_results["point_index"] = list(range(labels.shape[0]))
        df_results["cluster_label"] = cluster_labels
        df_results["label"] = labels

        # Count the number of data points in cluster J with label L
        df_counts_cluster_labels = df_results.groupby(["cluster_label", "label"])[["point_index"]].count()
        
        # Count the total number of points in cluster J
        df_counts_cluster = df_results.groupby(["cluster_label"])[["point_index"]].count()

        # Compute the probability of label L in cluster J
        df_p_labels = df_counts_cluster_labels / df_counts_cluster
        df_p_labels = df_p_labels.reset_index().rename(
            columns={
                "point_index": "proba", 
            })

        # Create a matrix P with the probabilities. The rows correspond to clusters 
        # and columns correspond to labels: P[i,j] = Pr(label = j | cluster = i)
        P = df_p_labels.pivot(
            index="cluster_label",
            columns="label",
            values="proba"
        ).fillna(value=0.0).to_numpy()

        return P


def create() -> Algorithm:
    return PairwiseExpectationMaximization()
