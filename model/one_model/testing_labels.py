import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F

###### Binary Test

# Define a random dataset to simulate the label data
class RandomLabelDataset(Dataset):
    def __init__(self, num_samples, num_classes=1, imbalance_factor=2):
        self.num_samples = num_samples
        self.num_classes = num_classes
        
        # Create imbalanced labels (more zeros than ones)
        self.labels = np.random.choice([0, 1], size=(num_samples,), p=[imbalance_factor / (1 + imbalance_factor), 1 / (1 + imbalance_factor)])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return a dummy image tensor (not needed for testing label smoothing)
        image = torch.randn(3, 224, 224)
        label = torch.tensor(self.labels[idx])
        return image, label

# Function to apply label smoothing
def apply_label_smoothing(labels, smoothing_value=0.2):
    smoothed_labels = labels.float() * (1 - smoothing_value) + (smoothing_value / 2)
    return smoothed_labels

# Function to create weighted sampler (for binary classification)
def create_weighted_sampler(dataset):
    # Get the count of 1's for each class
    labels = np.array(dataset.labels)
    class_counts = {0: np.sum(labels == 0), 1: np.sum(labels == 1)}
    
    # Compute weights: inverse of class frequency
    weights = {l: 1.0 / class_counts[l] for l in [0, 1]}
    
    # Calculate sample weights based on label frequencies
    sample_weights = np.array([weights[l] for l in labels])
    
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# Test 1: Label Smoothing
def test_label_smoothing():
    labels = torch.tensor([0, 1, 0, 1, 0, 1])  # Binary labels
    smoothed_labels = apply_label_smoothing(labels, smoothing_value=0.2)
    
    print("Original Labels:", labels)
    print("Smoothed Labels:", smoothed_labels)
    
    # Ensure the smoothed labels are not exactly 0 or 1
    assert torch.all(smoothed_labels != 0) and torch.all(smoothed_labels != 1), "Label smoothing failed"
    print("Label smoothing test passed!\n")

# Test 2: Weighted Random Sampler
def test_weighted_sampler():
    # Create dataset with imbalanced labels (more 0s than 1s)
    dataset = RandomLabelDataset(num_samples=1000, imbalance_factor=2)
    
    # Create weighted sampler using the function for binary classification
    sampler = create_weighted_sampler(dataset)
    
    # Check if the sampler works as expected
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)
    
    # Check the distribution of labels in the sampled batch
    batch_labels = []
    for images, labels in dataloader:
        batch_labels.extend(labels.numpy())
    
    batch_labels = np.array(batch_labels)
    if len(batch_labels) > 0:  # Ensure we only compute unique counts if the batch has data
        unique, counts = np.unique(batch_labels, return_counts=True)
        print("Sampled Label Distribution:", dict(zip(unique, counts)))
        
        # Assert that the distribution is more balanced due to the sampler
        assert abs(counts[0] - counts[1]) <= 20, "Sampling failed, imbalance is too high"
    else:
        print("No data sampled in batch.")

    print("Weighted sampler test passed!\n")

# Run the tests
test_label_smoothing()
test_weighted_sampler()


##### MultiLabel test

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F

class RandomMultilabelDataset(Dataset):
    def __init__(self, num_samples, num_classes=3, imbalance_factor=2):
        self.num_samples = num_samples
        self.num_classes = num_classes
        # Define probabilities for the two possible labels (0 and 1)
        p_0 = imbalance_factor / (1 + imbalance_factor)  # Probability for label 0
        p_1 = 1 / (1 + imbalance_factor)  # Probability for label 1
        
        # Normalize probabilities to sum to 1
        p_sum = p_0 + p_1
        
        # Generate random labels for each sample, each label can be 0 or 1
        self.labels = np.random.choice([0, 1], size=(num_samples, num_classes),
                                      p=[p_0/p_sum, p_1/p_sum])

        print("Initial Label Distribution (per class):")
        for i in range(self.num_classes):
            unique, counts = np.unique(self.labels[:, i], return_counts=True)
            print(f"Class {i} distribution: {dict(zip(unique, counts))}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(3, 224, 224)  # Random image tensor
        label = torch.tensor(self.labels[idx])
        return image, label


def apply_label_smoothing(labels, smoothing_value=0.2):
    smoothed_labels = labels.float() * (1 - smoothing_value) + (smoothing_value / 2)
    return smoothed_labels


def create_weighted_sampler(dataset):
    label_counts = np.sum(dataset.labels, axis=0)  # Count occurrences of each label
    weights = 1.0 / np.maximum(label_counts, 1)  # Inverse frequency of each label

    sample_weights = np.zeros(len(dataset))
    for i, labels in enumerate(dataset.labels):
        sample_weights[i] = np.sum(weights[labels == 1])  # Add weight only for valid labels (1's)

    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def test_multilabel_smoothing_and_sampler():
    # Create dataset with imbalanced multilabel labels
    dataset = RandomMultilabelDataset(num_samples=1000, num_classes=3)

    # Apply label smoothing
    smoothed_labels = apply_label_smoothing(torch.tensor(dataset.labels), smoothing_value=0.2)
    print("\nSmoothed Labels (first 5 samples):")
    print(smoothed_labels[:5])

    # Print distribution after smoothing
    print("\nSmoothed Label Distribution (per class):")
    for i in range(smoothed_labels.shape[1]):
        unique, counts = np.unique(smoothed_labels[:, i], return_counts=True)
        print(f"Class {i} smoothed distribution: {dict(zip(unique, counts))}")

    # Create weighted sampler based on dataset labels
    sampler = create_weighted_sampler(dataset)

    # Check the distribution of labels after sampling
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)
    batch_labels = []
    for images, labels in dataloader:
        batch_labels.extend(labels.numpy())

    batch_labels = np.array(batch_labels)
    print("\nSampled Label Distribution (first batch):")
    unique, counts = np.unique(batch_labels, return_counts=True)
    print(f"Sampled label distribution: {dict(zip(unique, counts))}")

    print("\nWeighted sampler test passed!")


# Run the tests
test_multilabel_smoothing_and_sampler()



