from pathlib import Path
import torch
import typer
from loguru import logger
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
import os

from digit_recognizer.config import MODELS_DIR, PROCESSED_DATA_DIR, PREDICTIONS_DIR, BATCH_SIZE

app = typer.Typer()

class UnlabeledDigitsDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data.iloc[idx, 0:].values.astype(np.uint8).reshape(28, 28)

        if self.transform:
            img = Image.fromarray(img)
            img = self.transform(img)

        return img

def predict(model_path,
            features_path,
            predictions_path,
            batch_size):

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}.")

    if not features_path.exists():
        raise FileNotFoundError(f"Test data not found at {features_path}.")

    if not predictions_path.exists():
        os.mkdir(predictions_path)

    predictions_file_path = predictions_path / 'predictions.csv'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    test_set = UnlabeledDigitsDataset(features_path, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = torch.load(model_path)
    model.to(device)
    model.eval()

    with open(predictions_file_path, 'w') as f:
        f.write("Id,Label\n")

    with torch.no_grad():  # Disable gradient computation for inference
        for idx, images in enumerate(tqdm(test_loader)):
            images = images.to(device)

            # Get model predictions
            outputs = model(images)

            # Convert model outputs to predicted labels
            _, preds = torch.max(outputs, 1)

            # Convert predictions and indices to CPU for saving
            preds = preds.cpu().numpy()
            ids = range(idx * test_loader.batch_size, idx * test_loader.batch_size + len(preds))

            # Append the predictions to the CSV file
            df = pd.DataFrame({"Id": ids, "Label": preds})
            df.to_csv(predictions_file_path, mode='a', header=False, index=False)

    logger.success("Done!")
    logger.info(f"Saved predictions to {predictions_file_path}")

@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / 'test.csv',
    model_path: Path = MODELS_DIR / 'model.pth',
    predictions_path: Path = PREDICTIONS_DIR,
    batch_size: int = BATCH_SIZE
):
    logger.info("Predicting...")
    predict(model_path=model_path, features_path=features_path, predictions_path=predictions_path, batch_size=batch_size)

if __name__ == "__main__":
    app()
