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
        transform: transforms.Compose = None,
        handle_uncertainty: str = "remove",  # Options: 'remove', 'zero', 'one'
    ):
        """
        Initializes the CheXpert Dataset.

        Args:
            csv_file (str): Path to the csv file with annotations.
            root_dir (str): Directory where CheXpert images are stored.
            targets (dict): Dictionary of target labels.
            transform (transforms.Compose, optional): Optional transform for images.
            handle_uncertainty (str): How to handle -1 labels: 'remove', 'zero', or 'one'.
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.targets = targets
        self.handle_uncertainty = handle_uncertainty

        # Extract target columns
        target_columns = [self.data.columns[i] for i in self.targets.values()]

        # Handle uncertain (-1) labels
        if self.handle_uncertainty == "remove":
            self.data = self.data[
                self.data[target_columns].apply(lambda row: (row != -1.0).all(), axis=1)
            ]
        elif self.handle_uncertainty in ["zero", "one"]:
            replace_value = 0 if self.handle_uncertainty == "zero" else 1
            self.data[target_columns] = self.data[target_columns].replace(
                -1.0, replace_value
            )

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
