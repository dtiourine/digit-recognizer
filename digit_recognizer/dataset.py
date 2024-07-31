from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from digit_recognizer.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

import os
from zipfile import ZipFile
import kaggle

app = typer.Typer()

kaggle.api.authenticate()

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # competition = 'digit-recognizer'
    #kaggle.api.dataset_download_files('', path=RAW_DATA_DIR, unzip=True)
    kaggle.api.authenticate()
    kaggle.api.competition_download_files('digit-recognizer', path=".")
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
