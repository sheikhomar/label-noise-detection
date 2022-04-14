import abc, itertools

from dataclasses import dataclass
from typing import Dict, Generator, List, Set

import pandas as pd
import numpy as np

from scipy.io.arff import loadarff

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


@dataclass
class PairwiseDataSplit:
    split_no: int
    class1_index: int
    class1_data: pd.DataFrame
    class2_index: int
    class2_data: pd.DataFrame


@dataclass
class TrainTestSplit:
    fold_no: int
    train_data: pd.DataFrame
    test_data: pd.DataFrame


@dataclass
class DataSet(abc.ABC):
    @property
    @abc.abstractmethod
    def size(self) -> int:
        """The number of points in this data set."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def class_names(self) -> List[str]:
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

    @property
    @abc.abstractmethod
    def feature_attributes(self) -> List[str]:
        """The names of the features."""
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

    @abc.abstractmethod
    def _get_raw_data_frame(self) -> pd.DataFrame:
        """Returns the internal data frame."""
        raise NotImplementedError

    def split_one_vs_one(self) -> Generator[PairwiseDataSplit, None, None]:
        """Partition the labelled data into pairs of classes using the one-vs-one approach.
        
        Notice that each split contains data points assigned a label in that pair.
        """
        class_pairings = itertools.combinations(np.unique(self.labels), 2)
        for split_no, (class1, class2) in enumerate(class_pairings):
            class1_indices = np.where(self.labels == class1)[0]
            class2_indices = np.where(self.labels == class2)[0]
            yield PairwiseDataSplit(
                split_no=split_no,
                class1_index=class1,
                class1_data=self._get_raw_data_frame().iloc[class1_indices].copy(),
                class2_index=class2,
                class2_data=self._get_raw_data_frame().iloc[class2_indices].copy(),
            )

    def split_k_fold(self,  n_splits: int, shuffle: bool = True, random_state=None) -> Generator[TrainTestSplit, None, None]:
        """Partition the labelled data into a number of folds."""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        df_data = self._get_raw_data_frame()
        for fold, (train_index, test_index) in enumerate(skf.split(X=df_data.index, y=self.labels)):
            yield TrainTestSplit(
                fold_no=fold,
                train_data=df_data.iloc[train_index],
                test_data=df_data.iloc[test_index],
            )


class UCIPPDataSet(DataSet):
    def __init__(self, file_path: str) -> None:
        raw_data, meta = loadarff(file_path)
        self._df_data = pd.DataFrame(raw_data, columns=meta.names())

        # Manually define the dtypes of each column in the dataframe.
        # TODO: Make it more efficient!
        for attr in meta.names():
            aarf_type, range = meta[attr]
            if aarf_type == "numeric":
                self._df_data[attr] = self._df_data[attr].astype(np.float32)
            elif aarf_type == "nominal":
                # Note: it is necessary to cast values to string before setting
                # dtype to `category` because `loadarff` does not convert categorical
                # values to string but keeps them as bytes.
                self._df_data[attr] = self._df_data[attr].astype(str).astype("category")
            else:
                raise ValueError(f"Do not know how to handle ARFF type: {aarf_type}")

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
    
    def _get_raw_data_frame(self) -> pd.DataFrame:
        return self._df_data

    @property
    def size(self) -> int:
        return self._df_data.shape[0]

    @property
    def n_classes(self) -> int:
        return self._label_encoder.classes_.size

    @property
    def class_names(self) -> List[str]:
        return self._label_encoder.classes_

    @property
    def label_attribute(self) -> str:
        return "Class"

    @property
    def feature_attributes(self) -> List[str]:
        return self._numerical_attrs + self._categorical_attrs + self._binary_attrs

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
