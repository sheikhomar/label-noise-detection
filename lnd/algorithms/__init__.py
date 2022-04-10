"""Package for label noise detection algorithms."""
import abc
from typing import List, Dict

from lnd.data.dataset import DataSet


class Algorithm:
    """Represents a label noise detection algorithm."""

    @abc.abstractmethod
    def run(self, data_set: DataSet) -> List[int]:
        """The name of this algorithm."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_params(self) -> Dict[str, object]:
        """Returns the current hyper-parameter values used in this instance."""
        raise NotImplementedError

    @abc.abstractmethod
    def set_params(self) -> Dict[str, object]:
        """Sets the the hyper-parameter values of this algorithm."""
        raise NotImplementedError
