from digit_recognizer.config import PROCESSED_DATA_DIR, BATCH_SIZE
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import numpy as np
from PIL import Image


class DigitsDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        # df = pd.read_csv(csv_file)
        # self.images = df.drop('label', axis=1).values
        # self.labels = df['label'].values
        # self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # image = self.images[idx].reshape(28, 28).astype(np.float32)
        # image = torch.from_numpy(image)
        #
        # if self.transform:
        #     image = self.transform(image)
        #
        # label = self.labels[idx]
        # return image, label
        img = self.data.iloc[idx, 1:].values.astype(np.uint8).reshape(28, 28)
        label = self.data.iloc[idx, 0]

        if self.transform:
            img = Image.fromarray(img)
            img = self.transform(img)

        return img, label


def load_data(transform, data_path=PROCESSED_DATA_DIR, batch_size=BATCH_SIZE):
    train_path = os.path.join(data_path, 'train.csv')
    val_path = os.path.join(data_path, 'val.csv')

    train_set = DigitsDataset(train_path, transform=transform)
    val_set = DigitsDataset(val_path, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader
