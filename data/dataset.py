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
        targets: set,
        transform: transforms.Compose = None,
    ):
        """
        Initializes the CheXpert Dataset based on the data from two .csv-Files.

        Args:
            csv_file (str): Path to the csv file with annotations.
            root_dir (str): Directory to path where CheXpert-v1.0-small ordner is placed.
            targets (set): Set of target labels.
            transform (transforms.Compose, optional): Optional transform to be applied on a sample. Defaults to None.
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.targets = targets

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

        # if only one sample cahnge shape to [1,1] instead of [1]

        return img, samples
