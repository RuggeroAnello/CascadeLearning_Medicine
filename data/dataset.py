import os
import pandas as pd
import numpy as np 
import torch 
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

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
        targets: dict,
        transform: transforms.Compose = None,
        uncertainty_mapping: bool = True
    ):
        """
        Initializes the CheXpert Dataset.

        Args:
            csv_file (str): Path to the csv file with annotations.
            root_dir (str): Directory where CheXpert images are stored.
            targets (dict): Dictionary of target labels.
            uncertainty_mapping (bool): If True, maps uncertain labels (-1) based on column name.
                                        If False, keeps original values.
            transform (transforms.Compose, optional): Optional transform for images.
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.targets = targets
        self.uncertainty_mapping = uncertainty_mapping

        # Extract target columns
        target_columns = [list(self.data.columns)[idx] for idx in self.targets.values()]

        # Only apply uncertainty mapping if uncertainty_mapping is True
        if uncertainty_mapping:
            for column in target_columns:
                if column.lower() in ['edema', 'atelectasis', 'pleural_effusion']:
                    self.data[column] = self.data[column].replace([-1, -1.0], 1)
                else:
                    self.data[column] = self.data[column].replace([-1, -1.0], 0)
                
        # Extract labels
        self.labels = self.data[target_columns].values

        # Find unique labels
        self.unique_labels = np.unique(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.root_dir, self.data.iloc[idx, 0])

        # Load image
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        # Load dict with selected target labels
        samples = self.labels[idx].astype(np.float64)

        # Ensure the labels are a 1D array
        return img, samples