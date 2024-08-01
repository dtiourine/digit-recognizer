from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

import pandas as pd
import kaggle

from digit_recognizer.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
import os
import zipfile
from sklearn.model_selection import train_test_split

app = typer.Typer()

def download_kaggle_dataset(competition_name, path):
    """
    Download the dataset from Kaggle.
    Note: You need to have a kaggle.json file in ~/.kaggle/ with your API credentials.
    """
    kaggle.api.authenticate()
    with tqdm(total=1, desc="Downloading dataset", unit="file") as pbar:
        kaggle.api.competition_download_files(competition_name, path=path)
        pbar.update(1)
    logger.info(f"Dataset downloaded to {path}")

    # Find the zip file
    zip_file = next(file for file in os.listdir(path) if file.endswith('.zip'))
    zip_path = os.path.join(path, zip_file)

    # Extract the contents
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        total_files = len(zip_ref.infolist())
        with tqdm(total=total_files, desc="Extracting files", unit="file") as pbar:
            for file in zip_ref.infolist():
                zip_ref.extract(file, path)
                pbar.update(1)
    logger.info(f"Dataset extracted to {path}")

    # Remove the zip file after extraction
    os.remove(zip_path)
    logger.info(f"Removed zip file: {zip_file}")\

def process_data(raw_data_path, processed_data_path):
    """
    Process the raw data and split into train and validation sets.
    """
    if not os.path.exists(processed_data_path):
        os.mkdir(processed_data_path)

    logger.info("Reading CSV file...")
    full_train_df = pd.read_csv(os.path.join(raw_data_path, 'train.csv'))

    logger.info("Splitting data...")
    X = full_train_df.drop('label', axis=1)
    y = full_train_df['label']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    logger.info("Saving processed datasets...")
    train_data = pd.concat([y_train, X_train], axis=1)
    val_data = pd.concat([y_val, X_val], axis=1)

    with tqdm(total=3, desc="Saving files", unit="file") as pbar:
        train_data.to_csv(os.path.join(processed_data_path, 'train.csv'), index=False)
        pbar.update(1)
        val_data.to_csv(os.path.join(processed_data_path, 'val.csv'), index=False)
        pbar.update(1)

        test_df = pd.read_csv(os.path.join(raw_data_path, 'test.csv'))
        test_df.to_csv(os.path.join(processed_data_path, 'test.csv'), index=False)
        pbar.update(1)

    logger.info(f"Processed data saved to {processed_data_path}")

@app.command()
def main():
    logger.info("Downloading the dataset...")
    competition_name = 'digit-recognizer'  # Replace with the actual competition name
    download_kaggle_dataset(competition_name, RAW_DATA_DIR)

    # Process the data to create validation set
    process_data(RAW_DATA_DIR, PROCESSED_DATA_DIR)
    logger.success("Processing dataset complete.")

if __name__ == "__main__":
    app()
