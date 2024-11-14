import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")


class CheXpertDataset(Dataset):
    """
    CheXpert Dataset
    """

    def __init__(
        self,
        csv_file: str,
        root_dir: str,
        transform: transforms.Compose = None,
    ):
        """
        Initializes the CheXpert Dataset based on the data from two .csv-Files.

        Args:
            csv_file (str): Path to the csv file with annotations.
            root_dir (str): Directory to path where CheXpert-v1.0-small ordner is placed.
            transform (transforms.Compose, optional): Optional transform to be applied on a sample. Defaults to None.
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # Load image
        img_path = os.path.join(self.root_dir, self.data.iloc[idx, 0])

        # Load dict with labels
        samples = {
            # TODO (FB): add patient ID
            "image": img_path,
            "sex": self.data.iloc[idx, 1],
            "age": self.data.iloc[idx, 2],
            "frontal/lateral": self.data.iloc[idx, 3],
            "ap/pa": self.data.iloc[idx, 4],
            "no_finding": self.data.iloc[idx, 5],
            "enlarged_cardiomediastinum": self.data.iloc[idx, 6],
            "cardiomegaly": self.data.iloc[idx, 7],
            "lung_opacity": self.data.iloc[idx, 8],
            "lung_lesion": self.data.iloc[idx, 9],
            "edema": self.data.iloc[idx, 10],
            "consolidation": self.data.iloc[idx, 11],
            "pneumonia": self.data.iloc[idx, 12],
            "atelectasis": self.data.iloc[idx, 13],
            "pneumothorax": self.data.iloc[idx, 14],
            "pleural_effusion": self.data.iloc[idx, 15],
            "pleural_other": self.data.iloc[idx, 16],
            "fracture": self.data.iloc[idx, 17],
            "support_devices": self.data.iloc[idx, 18],
        }

        if self.transform:
            samples = self.transform(samples)

        return samples
