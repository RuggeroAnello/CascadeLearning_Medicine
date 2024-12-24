import os
import pandas as pd
import numpy as np
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
        uncertainty_mapping: dict,
        transform: transforms.Compose = None,
    ):
        """
        Initializes the CheXpert Dataset.

        Args:
            csv_file (str): Path to the csv file with annotations.
            root_dir (str): Directory where CheXpert images are stored.
            targets (dict): Dictionary of target labels.
            uncertainty_mapping (dict): Dictionary specifying how to handle uncertain values (-1) for each column.
                                        E.g., {"Cardiomegaly": 1, "No Finding": 0}.
            transform (transforms.Compose, optional): Optional transform for images.
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.uncertainty_mapping = uncertainty_mapping
        self.targets = targets

        # Extract target columns
        target_columns = [self.data.columns[i] for i in self.targets.values()]

        # Apply column-specific handling of uncertain values (-1)
        for column in target_columns:
            if column in self.uncertainty_mapping:
                self.data[column] = self.data[column].replace(-1, self.uncertainty_mapping[column])

        # Extract labels
        self.labels = self.data[target_columns].values.astype(int)

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
        samples = np.zeros(len(self.targets))
        for i, key in enumerate(self.targets.values()):
            samples[i] = self.data.iloc[idx, key]

        # Ensure the labels are a 1D array
        return img, samples

    ''' Could this implementation be faster?

        def __getitem__(self, idx):
            # Load image
            img_path = os.path.join(self.root_dir, self.data.iloc[idx, 0])
            img = Image.open(img_path)

            if self.transform:
                img = self.transform(img)

            # Extract label vector for the current index
            label_vector = self.labels[idx]

            return img, label_vector

    '''