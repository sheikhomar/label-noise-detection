import abc

from dataclasses import dataclass
from typing import Dict, Generator, List, Set

import pandas as pd
import numpy as np

from scipy.io.arff import loadarff

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


@dataclass
class DataSet(abc.ABC):
    @property
    @abc.abstractmethod
    def size(self) -> int:
        """The number of points in this data set."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def classes(self) -> List[str]:
        """The set of classes in this data set (i.e., the label space)."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def n_classes(self) -> int:
        """The number of classes in this data set."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def label_attribute(self) -> str:
        """The name for the label attribute."""
        raise NotImplementedError

    @abc.abstractmethod
    def create_feature_transformer(self) -> TransformerMixin:
        """Creates a transformer to convert the raw data into usable feature vectors."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def labels(self) -> np.ndarray:
        """The labels of all points in the data set."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def features(self) -> np.ndarray:
        """The transformed features of all points in the data set."""
        raise NotImplementedError


class UCIPPDataSet(DataSet):
    def __init__(self, file_path: str) -> None:
        raw_data, meta = loadarff(file_path)
        self._df_data = pd.DataFrame(raw_data, columns=meta.names())

        self._numerical_attrs = [c for c in self._df_data.columns if c.startswith("V")]
        self._categorical_attrs = []
        self._binary_attrs = []

        # Save a copy of the original labels for each sample.
        # This allows us to apply multiple label noise matrices.
        self._original_labels = list(self._df_data[self.label_attribute])

        # Encode labels
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(self._df_data[self.label_attribute])
        self._y_given = self._label_encoder.transform(self._df_data[self.label_attribute])
    
    @property
    def size(self) -> int:
        return self._df_data.shape[0]

    @property
    def n_classes(self) -> int:
        return self._label_encoder.classes_.size

    @property
    def classes(self) -> List[str]:
        return self._label_encoder.classes_

    @property
    def label_attribute(self) -> str:
        return "Class"

    def create_feature_transformer(self) -> TransformerMixin:
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        feature_extrators = ColumnTransformer(
            transformers=[
                ('numerical', numeric_transformer, self._numerical_attrs),
                ('categorical', categorical_transformer, self._categorical_attrs),
                ('binary', 'passthrough', self._binary_attrs)
            ]
        )
        return feature_extrators

    @property
    def features(self) -> np.ndarray:
        return self.create_feature_transformer().fit_transform(self._df_data)
    
    @property
    def labels(self) -> np.ndarray:
        return self._y_given


def load_dataset(file_path: str) -> DataSet:
    if "/ucipp/" in file_path:
        return UCIPPDataSet(file_path=file_path)
    raise Exception(f"Do not know how to parse file {file_path}")
