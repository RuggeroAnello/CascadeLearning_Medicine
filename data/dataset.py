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
        uncertainty_mapping: bool = True,
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
        if self.uncertainty_mapping:
            for column in target_columns:
                if column.lower() in ["edema", "atelectasis", "pleural_effusion"]:
                    self.data[column] = self.data[column].replace([-1, -1.0], 1)
                else:
                    self.data[column] = self.data[column].replace([-1, -1.0], 0)

        # Extract labels
        self.labels = self.data[target_columns].values

        # Find unique labels
        self.unique_labels = np.unique(self.labels)

        # Compute positive weight matrix
        self.compute_pos_weight_matrix()
        self.compute_class_weights()

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

    def compute_pos_weight_matrix(self):
        """
        Computes the positive weight matrix for weighted loss functions.
        For each label, pos_weight = (# negatives) / (# positives)
        """
        # Assume self.labels has shape [num_samples, num_labels] for multilabel classification
        num_labels = self.labels.shape[1] if self.labels.ndim > 1 else 1
        pos_weights = []
        for i in range(num_labels):
            # Get column i if multilabel otherwise use self.labels
            col = self.labels[:, i] if num_labels > 1 else self.labels
            num_pos = np.sum(col == 1)
            num_neg = np.sum(col == 0)
            # Avoid division by zero - if there are no positives, use 1
            pos_weight = num_neg / (num_pos if num_pos > 0 else 1)
            pos_weights.append(pos_weight)

        # Create a tensor of shape [C]. If your target is 2D (or 3D with spatial dims) and you want the weight applied uniformly,
        # you can reshape it to [C, 1, 1] as described in the BCEWithLogitsLoss docs.
        pos_weights = torch.tensor(pos_weights, dtype=torch.float32)
        # Uncomment the following line if your target has shape [C, H, W] and you want to broadcast on the H and W dimensions:
        # pos_weights = pos_weights.unsqueeze(1).unsqueeze(2)
        self.pos_weights = pos_weights

    def compute_class_weights(self):
        """
        Computes the class weights for the focal loss. The class weights are computed as:
        weight = 1 + |log(ratio)|, where ratio = (# negatives) / (# positives)
        """
        num_labels = self.labels.shape[1] if self.labels.ndim > 1 else 1
        class_weights = []
        for i in range(num_labels):
            col = self.labels[:, i] if num_labels > 1 else self.labels
            num_pos = np.sum(col == 1)
            num_neg = np.sum(col == 0)
            ratio = num_neg / num_pos
            weight = 1 + np.abs(np.log(ratio))

            # Update class weights
            class_weights.append(weight)

        self.class_weights = class_weights

        print(f"Class weights: {self.class_weights}")
