import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


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
        label_smoothing: float = 0.0,
    ):
        """
        Initializes the CheXpert Dataset.

        Args:
            csv_file (str): Path to the csv file with annotations.
            root_dir (str): Directory where CheXpert images are stored.
            targets (dict): Dictionary of target labels.
            transform (transforms.Compose, optional): Optional transform for images.
            uncertainty_mapping (bool): If True, maps uncertain labels (-1) based on column name.
                                        If False, keeps original values.
            label_smoothing (float): Label smoothing factor. If label smoothing should be used, you have to use uncertainty_mapping=True.
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.targets = targets
        self.uncertainty_mapping = uncertainty_mapping
        self.label_smoothing = label_smoothing

        # Extract target columns
        target_columns = [list(self.data.columns)[idx] for idx in self.targets.values()]

        self.zero = 0.0
        self.one = 1.0

        # Only apply uncertainty mapping if uncertainty_mapping is True
        if self.uncertainty_mapping:
            # If label smoothing is enabled, apply label smoothing
            if self.label_smoothing > 0:
                for column in target_columns:
                    # Store the mapped values for calculating the class weights
                    self.zero = 0.0
                    self.one = 1.0
                    # Apply label smoothing: move the labels closer to 0.5
                    self.zero = (
                        1 - self.label_smoothing
                    ) * self.zero + self.label_smoothing * 0.5
                    self.one = (
                        1 - self.label_smoothing
                    ) * self.one + self.label_smoothing * 0.5

                    # Map uncertain labels as explained in Berrada et al. 2023
                    if column.lower() in ["edema", "atelectasis", "pleural_effusion"]:
                        self.data[column] = self.data[column].replace(
                            [-1, -1.0], self.one
                        )
                    else:
                        self.data[column] = self.data[column].replace(
                            [-1, -1.0], self.zero
                        )
            else:
                # Map uncertain labels as explained in Berrada et al. 2023
                for column in target_columns:
                    if column.lower() in ["edema", "atelectasis", "pleural_effusion"]:
                        self.data[column] = self.data[column].replace([-1, -1.0], 1)
                    else:
                        self.data[column] = self.data[column].replace([-1, -1.0], 0)

        # Extract labels
        self.labels = self.data[target_columns].values

        # Find unique labels
        self.unique_labels = np.unique(self.labels)

        # Compute positive weight matrix for weighted loss functions
        self.compute_pos_weight_matrix()
        # Compute class weights for focal loss
        self.compute_class_weights()

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx) -> tuple:
        """
        Returns the item at the given index.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: Tuple with the image and the target labels
        """
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
            if self.label_smoothing > 0:
                num_pos += np.sum(col == self.one)
                num_neg += np.sum(col == self.zero)
            # Avoid division by zero - if there are no positives, use 1
            pos_weight = num_neg / (num_pos if num_pos > 0 else 1)
            pos_weights.append(pos_weight)

        # Create a tensor of shape [C]. C is the number of classes
        pos_weights = torch.tensor(pos_weights, dtype=torch.float32)
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
            if self.label_smoothing > 0:
                num_pos += np.sum(col == self.one)
                num_neg += np.sum(col == self.zero)

            ratio = num_neg / num_pos
            weight = 1 + np.abs(np.log(ratio))

            # Update class weights
            class_weights.append(weight)

        self.class_weights = class_weights

        print(f"Class weights: {self.class_weights}")
