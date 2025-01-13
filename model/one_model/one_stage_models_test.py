import torch
import json
import torchvision
import os
import wandb
import numpy as np
import torch.nn.functional as F
from torch import nn
from pytorch_lightning import LightningModule
from typing import List, Dict
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics.classification import MultilabelAccuracy, MultilabelPrecision, MultilabelRecall, MultilabelF1Score, MultilabelAUROC
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC

from tqdm import tqdm


class AbstractOneStageModel(torch.nn.Module):
    def __init__(
        self,
        params: dict,
        **kwargs,
    ):
        """
        Initialize the model with the given hyperparameters.

        Args:
            params (dict): Dictionary containing the hyperparameters.
            loss_fn (torch.nn.Module): Loss function to use for training
            optimizer (torch.optim.Optimizer): Optimizer to use for training
        """
        super().__init__()
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = params.get("labels", [])

        # Save hyperparameters
        self.params = params
        self._configure_hyperparameters(params)
        self._configure_metrics(params)

        # Save results
        self.results = {}

    def forward(self, x):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError

    def _configure_metrics(self, params):
        """
        Configure both per-label binary metrics and overall multilabel metrics
        """
        threshold = params.get("confidence_threshold", 0.5)
        
        # Initialize overall multilabel metrics
        self.val_metrics_multilabel = nn.ModuleDict({
            "accuracy": MultilabelAccuracy(num_labels=self.num_classes, threshold=threshold).to(self.device),
            "precision": MultilabelPrecision(num_labels=self.num_classes, threshold=threshold).to(self.device),
            "recall": MultilabelRecall(num_labels=self.num_classes, threshold=threshold).to(self.device),
            "f1": MultilabelF1Score(num_labels=self.num_classes, threshold=threshold).to(self.device),
            "auroc": MultilabelAUROC(num_labels=self.num_classes).to(self.device)
        })
        
        self.test_metrics_multilabel = nn.ModuleDict({
            "accuracy": MultilabelAccuracy(num_labels=self.num_classes, threshold=threshold).to(self.device),
            "precision": MultilabelPrecision(num_labels=self.num_classes, threshold=threshold).to(self.device),
            "recall": MultilabelRecall(num_labels=self.num_classes, threshold=threshold).to(self.device),
            "f1": MultilabelF1Score(num_labels=self.num_classes, threshold=threshold).to(self.device),
            "auroc": MultilabelAUROC(num_labels=self.num_classes).to(self.device)
        })
        
        # Initialize per-label binary metrics
        self.val_metrics_per_label = nn.ModuleDict()
        self.test_metrics_per_label = nn.ModuleDict()
        
        for label_idx in range(self.num_classes):
            self.val_metrics_per_label[str(label_idx)] = nn.ModuleDict({
                "accuracy": BinaryAccuracy(threshold=threshold).to(self.device),
                "precision": BinaryPrecision(threshold=threshold).to(self.device),
                "recall": BinaryRecall(threshold=threshold).to(self.device),
                "f1": BinaryF1Score(threshold=threshold).to(self.device),
                "auroc": BinaryAUROC().to(self.device)
            })
            
            self.test_metrics_per_label[str(label_idx)] = nn.ModuleDict({
                "accuracy": BinaryAccuracy(threshold=threshold).to(self.device),
                "precision": BinaryPrecision(threshold=threshold).to(self.device),
                "recall": BinaryRecall(threshold=threshold).to(self.device),
                "f1": BinaryF1Score(threshold=threshold).to(self.device),
                "auroc": BinaryAUROC().to(self.device)
            })

    def save_model(self, path: str, epoch: int = None):
        """
        Save the model and its weights to the given path.
        """
        model = self.model.to("cpu")
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        if epoch is not None:
            path = os.path.join(path, f"model_epoch_{epoch}.pth")
        else:
            path = os.path.join(path, "model.pth")
        torch.save(model, path)

    def load_model(self, path: str):
        """
        Load the model from the given path.
        """
        self.model = torch.load(path, weights_only=False)

    def save_hparams(self, path: str):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        path = os.path.join(path, "hparams.json")
        with open(path, "w") as f:
            json.dump(self.params, f)

    def load_hparams(self, path: str):
        self.params = json.load(open(path, "r"))

    def get_num_labels(self, targets):
        return len(targets)

    def create_weighted_sampler(self, dataset):
        """
        Create a WeightedRandomSampler for multilabel data.
        Computes weights based on positive/negative ratio for each label.
        """
        labels = torch.stack([label for _, label in dataset])  # Get all labels
        pos_weights = []
        neg_weights = []
        
        # Calculate weights for each label
        for i in range(self.num_classes):
            # Count positive and negative samples for this label
            pos_count = torch.sum(labels[:, i] == 1).item()
            neg_count = torch.sum(labels[:, i] == 0).item()
            total = pos_count + neg_count
            
            # Calculate weights (inverse frequency)
            pos_weight = total / (2 * pos_count) if pos_count > 0 else 0
            neg_weight = total / (2 * neg_count) if neg_count > 0 else 0
            
            pos_weights.append(pos_weight)
            neg_weights.append(neg_weight)
        
        # Create sample weights for each instance
        sample_weights = []
        for label in labels:
            weight = 0
            for i, l in enumerate(label):
                if l == 1:
                    weight += pos_weights[i]
                else:
                    weight += neg_weights[i]
            weight /= self.num_classes  # Average weight across all labels
            sample_weights.append(weight)
        
        return WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True
        )
    
    def _prepare_dataloaders(self, train_dataset, val_dataset):
        if self.use_weighted_sampler:
            sampler = self.create_weighted_sampler(train_dataset)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return train_loader, val_loader

    def apply_label_smoothing(self, labels):
        """
        Applies label smoothing to the labels tensor.
        """
        num_classes = self.num_classes
        smoothed_labels = labels * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        return smoothed_labels

    def _training_step(self, batch, loss_fn):
        """
        Perform a single training step on the given batch.
        """
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)

        # Forward pass
        outputs = self.forward(images)

        # Apply label smoothing if enabled
        if self.label_smoothing > 0.0:
            smoothed_labels = self.apply_label_smoothing(labels)
            loss = loss_fn(outputs, smoothed_labels)
        else:
            loss = loss_fn(outputs, labels)

        return loss

    def _validation_step(self, batch, metrics_multilabel, metrics_per_label, loss_fn):
        """
        Perform a single validation/test step and update both multilabel and per-label metrics
        """
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)

        # Forward pass
        outputs = self.forward(images)
        loss = loss_fn(outputs, labels)

        # Update multilabel metrics
        with torch.no_grad():
            for metric_name, metric in metrics_multilabel.items():
                metric.update(outputs, labels.int())
            
            # Update per-label metrics
            for label_idx in range(self.num_classes):
                for _, metric in metrics_per_label[str(label_idx)].items():
                    metric.update(outputs[:, label_idx], labels[:, label_idx].int())

        return loss

    def _configure_optimizer(self):
        if self.optimizer_name.lower() == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.lr)
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, train_dataset, val_dataset, tb_logger, path):
        """
        Train the model using the provided datasets.
        """
        train_loader, val_loader = self._prepare_dataloaders(train_dataset, val_dataset)
        optimizer = self._configure_optimizer()
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(len(train_loader) / 5), gamma=0.7
        )

        self.model = self.model.to(self.device)
        print(f"Device used {self.device}")

        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            train_loop = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{self.num_epochs}")
            training_loss = 0

            for train_iteration, batch in enumerate(train_loop):
                optimizer.zero_grad()
                loss = self._training_step(batch, self.loss_fn)
                loss.backward()
                optimizer.step()

                training_loss += loss.item()
                train_loop.set_postfix(train_loss=f"{training_loss / (train_iteration + 1):.6f}")

                # Logging
                tb_logger.add_scalar("Train/loss", loss.item(), epoch * len(train_loader) + train_iteration)
                wandb.log({"epoch": epoch, "train_loss": loss.item()})

            scheduler.step()

            # Validation
            self.model.eval()
            val_loop = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{self.num_epochs}")
            validation_loss = 0

            # Reset all metrics
            for metric in self.val_metrics_multilabel.values():
                metric.reset()
            for label_metrics in self.val_metrics_per_label.values():
                for metric in label_metrics.values():
                    metric.reset()

            with torch.no_grad():
                for val_iteration, batch in enumerate(val_loop):
                    loss = self._validation_step(
                        batch, 
                        self.val_metrics_multilabel,
                        self.val_metrics_per_label,
                        self.loss_fn
                    )
                    validation_loss += loss.item()
                    val_loop.set_postfix(val_loss=f"{validation_loss / (val_iteration + 1):.6f}")

            # Log overall multilabel metrics
            for metric_name, metric in self.val_metrics_multilabel.items():
                try:
                    metric_value = metric.compute()
                    if isinstance(metric_value, torch.Tensor) and metric_value.dim() > 0:
                        metric_value = metric_value.mean()
                    tb_logger.add_scalar(f"Val/overall_{metric_name}", metric_value, epoch)
                    wandb.log({f"Val/overall_{metric_name}": metric_value})
                except Exception as e:
                    print(f"Error computing overall {metric_name}: {e}")

            # Log per-label metrics
            for label_idx in range(self.num_classes):
                for metric_name, metric in self.val_metrics_per_label[str(label_idx)].items():
                    try:
                        metric_value = metric.compute()
                        tb_logger.add_scalar(f"Val/label_{label_idx}_{metric_name}", metric_value, epoch)
                        wandb.log({f"Val/label_{label_idx}_{metric_name}": metric_value})
                    except Exception as e:
                        print(f"Error computing {metric_name} for label {label_idx}: {e}")

    def test(self, test_dataset, tb_logger):
        """
        Test the model using the provided dataset.
        """
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        self.model.eval()
        test_loop = tqdm(test_loader, desc="Testing")
        test_loss = 0.0

        # Reset all metrics
        for metric in self.test_metrics_multilabel.values():
            metric.reset()
        for label_metrics in self.test_metrics_per_label.values():
            for metric in label_metrics.values():
                metric.reset()

        with torch.no_grad():
            for test_iteration, batch in enumerate(test_loop):
                loss = self._validation_step(
                    batch,
                    self.test_metrics_multilabel,
                    self.test_metrics_per_label,
                    self.loss_fn
                )
                test_loss += loss.item()
                test_loop.set_postfix(test_loss=f"{test_loss / (test_iteration + 1):.6f}")

        # Average test loss
        test_loss /= len(test_loader)
        tb_logger.add_scalar("Test/loss", test_loss)
        wandb.log({"test_loss": test_loss})
        print(f"\nTest loss: {test_loss:.6f}")

        # Log overall multilabel metrics
        print("\nOverall Metrics:")
        for metric_name, metric in self.test_metrics_multilabel.items():
            try:
                metric_value = metric.compute()
                if isinstance(metric_value, torch.Tensor) and metric_value.dim() > 0:
                    metric_value = metric_value.mean()
                tb_logger.add_scalar(f"Test/overall_{metric_name}", metric_value)
                wandb.log({f"Test/overall_{metric_name}": metric_value})
                print(f"Overall {metric_name}: {metric_value:.4f}")
            except Exception as e:
                print(f"Error computing overall {metric_name}: {e}")

        # Log per-label metrics
        print("\nPer-Label Metrics:")
        for label_idx in range(self.num_classes):
            print(f"\nLabel {label_idx}:")
            for metric_name, metric in self.test_metrics_per_label[str(label_idx)].items():
                try:
                    metric_value = metric.compute()
                    tb_logger.add_scalar(f"Test/label_{label_idx}_{metric_name}", metric_value)
                    wandb.log({f"Test/label_{label_idx}_{metric_name}": metric_value})
                    print(f"  {metric_name}: {metric_value:.4f}")
                except Exception as e:
                    print(f"Error computing {metric_name} for label {label_idx}: {e}")



class ResNet50OneStage(AbstractOneStageModel):
    def __init__(
        self,
        params: dict,
        input_channels: int = 1,
        **kwargs,
    ):
        """
        Initialize the model with the given hyperparameters.

        Args:
            params (dict): Dictionary containing the hyperparameters.
            # input_size (np.array): Size of the input image. Shape: [channels, height, width]
            num_labels (int): Number of classes in the dataset
        """
        super().__init__(
            params=params,
            **kwargs,
        )

        # Load pretrained model
        # Best available weights (currently alias for IMAGENET1K_V2)
        self.model = torchvision.models.resnet50(weights="IMAGENET1K_V2").to(
            self.device
        )

        # Adapt input size of model to the image channels
        if input_channels != 3:
            self.model.conv1 = torch.nn.Conv2d(
                input_channels,
                self.model.conv1.out_channels,
                kernel_size=self.model.conv1.kernel_size,
                stride=self.model.conv1.stride,
                padding=self.model.conv1.padding,
                bias=False,
            )

        # Replace the output layer
        self.model.fc = torch.nn.Linear(
            self.model.fc.in_features,
            self.get_num_labels(self.labels),
        )

        # Set device
        self.model.to(self.device)

    def forward(self, x):
        self.model = self.model.to(
            self.device
        )  # Ensure the entire model is on the correct device
        x = x.to(self.device)
        return self.model(x)

    @property
    def name(self):
        return "ResNet50OneStage"


class ResNet18OneStage(AbstractOneStageModel):
    def __init__(
        self,
        params: dict,
        input_channels: int = 1,
        **kwargs,
    ):
        """
        Initialize the model with the given hyperparameters.

        Args:
            params (dict): Dictionary containing the hyperparameters.
            # input_size (np.array): Size of the input image. Shape: [channels, height, width]
            num_labels (int): Number of classes in the dataset
        """
        super().__init__(
            params=params,
            **kwargs,
        )

        # Load pretrained model
        self.model = torchvision.models.resnet18(weights="IMAGENET1K_V1")

        # Adapt input size of model to the image channels
        if input_channels != 3:
            self.model.conv1 = torch.nn.Conv2d(
                input_channels,
                self.model.conv1.out_channels,
                kernel_size=self.model.conv1.kernel_size,
                stride=self.model.conv1.stride,
                padding=self.model.conv1.padding,
                bias=False,
            )

        # Replace the output layer
        self.model.fc = torch.nn.Linear(
            self.model.fc.in_features,
            self.get_num_labels(self.labels),
        )

        # Set device
        self.model.to(self.device)

    def forward(self, x):
        # TODO remove
        # self.model = self.model.to(
        #    self.device
        # )  # Ensure the entire model is on the correct device
        # x = x.to(self.device)
        return self.model(x)

    @property
    def name(self):
        return "ResNet18OneStage"


class ResNet34OneStage(AbstractOneStageModel):
    def __init__(
        self,
        params: dict,
        input_channels: int = 1,
        num_labels: int = None,
        **kwargs,
    ):
        """
        Initialize the model with the given hyperparameters.

        Args:
            params (dict): Dictionary containing the hyperparameters.
            # input_size (np.array): Size of the input image. Shape: [channels, height, width]
            num_labels (int): Number of classes in the dataset
        """
        super().__init__(
            params=params,
            **kwargs,
        )

        # Load pretrained model
        self.model = torchvision.models.resnet34(weights="IMAGENET1K_V1").to(
            self.device
        )

        # Adapt input size of model to the image channels
        if input_channels != 3:
            self.model.conv1 = torch.nn.Conv2d(
                input_channels,
                self.model.conv1.out_channels,
                kernel_size=self.model.conv1.kernel_size,
                stride=self.model.conv1.stride,
                padding=self.model.conv1.padding,
                bias=False,
            )

        # Replace the output layer
        self.model.fc = torch.nn.Linear(
            self.model.fc.in_features,
            self.get_num_labels(self.labels),
        )

        # Set device
        self.model.to(self.device)

    def forward(self, x):
        self.model = self.model.to(
            self.device
        )  # Ensure the entire model is on the correct device
        x = x.to(self.device)
        return self.model(x)

    @property
    def name(self):
        return "ResNet18OneStage"