import json, os

from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import List

import click
import requests

from tqdm import tqdm

from lnd.algorithms import Algorithm
from lnd.data.dataset import load_dataset

class DataPreparation:
    def _download_file(self, url: str, file_path: Path):
        """
        Downloads file from `url` to `file_path`.
        """
        print(f"Downloading {url} to {file_path}...")
        chunk_size = 1024
        r = requests.get(url, stream=True)
        with open(file_path, 'wb') as f:
            total_size = int(r.headers.get('Content-Length', 10 * chunk_size))
            pbar = tqdm( unit="B", unit_scale=True, total=total_size)
            for chunk in r.iter_content(chunk_size=chunk_size): 
                if chunk: # filter out keep-alive new chunks
                    pbar.update (len(chunk))
                    f.write(chunk)

class KDDCup2009DataPreparation(DataPreparation):
    """Preparation for the KDD Cup 2009 Data.
    
     https://kdd.org/kdd-cup/view/kdd-cup-2009/Data
    """
    def __init__(self) -> None:
        pass


class DataPreparationRunner:
    def __init__(self) -> None:
        pass
    

@click.command(help="Prepares the given data set.")
@click.option(
    "-d",
    "--data-set",
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
    DataPreparationRunner(
        algorithm_name=algorithm_name,
        algorithm_params=algorithm_params,
        data_path=input_path,
        noise_type=noise_type,
        random_seed=random_seed,
        output_dir=output_dir,
    ).run()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
