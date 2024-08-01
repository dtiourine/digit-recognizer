from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from digit_recognizer.config import MODELS_DIR, PROCESSED_DATA_DIR, NUM_EPOCHS
from load_data import load_data
from architecture import MNISTModel

import torch
import torch.nn as nn
from torchvision import transforms

app = typer.Typer()


def train(num_epochs,
          train_loader,
          val_loader,
          model,
          criterion,
          optimizer,
          device,
          model_save_path):
    best_val_accuracy = 0
    overall_progress_bar = tqdm(total=num_epochs, desc='Overall Training Progress', position=0, leave=True)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        # Save the model if the validation accuracy is the best we've seen so far.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            # torch.save(model.state_dict(), model_save_path)
            torch.save(model, model_save_path)

        overall_progress_bar.set_postfix({
            'Best Val Accuracy': f'{best_val_accuracy:.2f}%',
            'Current Train Accuracy': f'{train_accuracy:.2f}%',
            'Current Val Accuracy': f'{val_accuracy:.2f}%'
        })
        overall_progress_bar.update(1)

    overall_progress_bar.close()


@app.command()
def main(model_path: Path = MODELS_DIR / "model.pth"):
    num_epochs = NUM_EPOCHS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    train_loader, val_loader = load_data(transform=transform)

    if not model_path.exists():
        if model_path.is_dir():
            raise FileNotFoundError(f"Expected a file but found {model_path} is a directory.")

        logger.info(f"Did not find existing model at {model_path}")
        logger.info("Initializing new model...")
        model = MNISTModel()
        model.to(device)
    elif model_path.exists() and model_path.is_file():
        logger.info(f"Found existing model at {model_path}")
        model = torch.load(model_path)
        model.to(device)
    else:
        raise FileNotFoundError(f"Expected a file but found {model_path} is not a file.")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    train(num_epochs=num_epochs, train_loader=train_loader, val_loader=val_loader, model=model, criterion=criterion,
          optimizer=optimizer, device=device, model_save_path=model_path)
    logger.success("Training Complete")


if __name__ == "__main__":
    app()
