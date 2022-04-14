import json, os

from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import List

import click

from lnd.algorithms import Algorithm
from lnd.data.dataset import load_dataset


def type_1_error_rate(
        data_set_size: int,
        known_mislabelled_indices: List[int],
        detected_mislabelled_indices: List[int]
    ) -> float:
    all_indices = list(range(data_set_size))
    correctly_labelled_indices = list(set(all_indices) - set(known_mislabelled_indices))

    # Find samples which are correctly labelled but are detected as mislabelled.
    type_1_error_indices = [
        index
        for index in detected_mislabelled_indices
        if index in correctly_labelled_indices
    ]

    # Type 1 errors are correctly labelled instances that
    # are erroneously identified as mislabelled.
    er1 = len(type_1_error_indices) / len(correctly_labelled_indices)

    return er1


def type_2_error_rate(
        data_set_size: int,
        known_mislabelled_indices: List[int],
        detected_mislabelled_indices: List[int]
    ) -> float:
    # Type 2 errors are known mislabeled instances which are not detected.
    type_2_indices = [
        index
        for index in known_mislabelled_indices
        if index not in detected_mislabelled_indices
    ]
    if len(known_mislabelled_indices) > 0:
        er2 = len(type_2_indices) / len(known_mislabelled_indices)
    else:
        er2 = 0.0
    return er2


def noise_elimination_precision_score(
        data_set_size: int,
        known_mislabelled_indices: List[int],
        detected_mislabelled_indices: List[int]
    ) -> float:
    # Noise elimination precision (NEP) is the percentage of
    # detected instances that are known to be mislabelled.
    detected_and_known_indices = [
        index
        for index in known_mislabelled_indices
        if index in detected_mislabelled_indices
    ]
    nep = len(detected_and_known_indices) / len(detected_mislabelled_indices)
    return nep


class Evaluator:
    def __init__(self, *, 
            algorithm_name: str,
            algorithm_params: str,
            data_path: str,
            noise_type: str,
            random_seed: int,
            output_dir: str
        ) -> None:
        self._algorithm_name = algorithm_name
        self._data_path = data_path
        self._noise_type = noise_type
        self._random_seed = random_seed
        self._output_dir = Path(output_dir)
        if self._output_dir.name != self._algorithm_name:
            self._output_dir = self._output_dir / self._algorithm_name
        experiment_no = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self._output_dir = self._output_dir / experiment_no
        if algorithm_params is not None and len(algorithm_params) > 0:
            if os.path.exists(algorithm_params):
                with open(algorithm_params, "r") as fp:
                    self._algorithm_params = json.load(fp)
            else:
                self._algorithm_params = json.loads(algorithm_params)
        print(f"Algorithm params: {self._algorithm_params}")

    def run(self) -> None:
        # Load label noise detection algorithm.
        detector = self._load_algorithm()

        # Load data
        data_set = load_dataset(self._data_path)

        start_time = datetime.now()
        likely_mislabelled = detector.run(data_set=data_set)
        end_time = datetime.now()

        print(end_time - start_time)

        print(likely_mislabelled)

    def _load_algorithm(self) -> Algorithm:
        try:
            m = import_module(f"lnd.algorithms.{self._algorithm_name}")
            algorithm = m.create()  # type: ignore
            return algorithm
        except ModuleNotFoundError:
            raise ValueError(f"Detector {self._algorithm_name} not found.")



@click.command(help="Evaluates a noise detector.")
@click.option(
    "-a",
    "--algorithm-name",
    type=click.STRING,
    required=True,
)
@click.option(
    "-p",
    "--algorithm-params",
    type=click.STRING,
    default="",
    required=False,
)
@click.option(
    "-i",
    "--input-path",
    type=click.STRING,
    required=True,
)
@click.option(
    "-n",
    "--noise-type",
    type=click.STRING,
    required=True,
)
@click.option(
    "-r",
    "--random-seed",
    type=click.INT,
    required=True,
)
@click.option(
    "-o",
    "--output-dir",
    type=click.STRING,
    required=True,
)
def main(
    algorithm_name: str,
    algorithm_params: str,
    input_path: str,
    noise_type: str,
    random_seed: int,
    output_dir: str,
):
    Evaluator(
        algorithm_name=algorithm_name,
        algorithm_params=algorithm_params,
        data_path=input_path,
        noise_type=noise_type,
        random_seed=random_seed,
        output_dir=output_dir,
    ).run()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
