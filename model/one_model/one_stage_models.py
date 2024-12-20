import torch
import json
import torchvision
import os
import wandb
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torcheval.metrics import (
    BinaryRecall,
    BinaryPrecision,
    BinaryF1Score,
    BinaryAccuracy,
    BinaryAUROC,
)

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
        self.labels = None
        self.unique_labels = None

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

    def _configure_hyperparameters(self, params):
        self.lr = params.get("lr", 1e-3)
        self.batch_size = params.get("batch_size", 32)
        self.num_epochs = params.get("num_epochs", 10)
        self.optimizer_name = params.get("optimizer", "adam")
        # Map string loss function names to actual loss function classes
        loss_fn_str = params.get("loss_fn", "BCEWithLogitsLoss")
        loss_fn_mapping = {
            "cross_entropy": torch.nn.CrossEntropyLoss(),
            "mse_loss": torch.nn.MSELoss(),
            "BCEWithLogitsLoss": torch.nn.BCEWithLogitsLoss(),
            # Add other loss functions here if needed
        }  # Set the loss function, defaulting to BCEWithLogitsLoss if not specified
        self.loss_fn = loss_fn_mapping.get(loss_fn_str, torch.nn.BCEWithLogitsLoss())
        self.use_weighted_sampler = params.get("use_weighted_sampler", False)
        self.save_epoch = params.get("save_epoch", 1)
        self.confidence_threshold = params.get("confidence_threshold", 0.5)

    def _configure_metrics(self, params):
        # TODO: Add more metrics if needed
        self.val_metrics = {}
        self.test_metrics = {}

        metrics = params.get("metrics", [])

        threshold = params.get("confidence_threshold", 0.5)

        # Accuracy is always calculated
        self.val_metrics["accuracy"] = BinaryAccuracy(
            threshold=threshold
        )  # These do NOT accept the confidence_threshold as argument -> done in validation_step
        self.test_metrics["accuracy"] = BinaryAccuracy(threshold=threshold)

        if "precision" in metrics:
            self.val_metrics["precision"] = BinaryPrecision(threshold=threshold)
            self.test_metrics["precision"] = BinaryPrecision(threshold=threshold)
        if "recall" in metrics:
            self.val_metrics["recall"] = BinaryRecall(threshold=threshold)
            self.test_metrics["recall"] = BinaryRecall(threshold=threshold)
        if "f1" in metrics:
            self.val_metrics["f1"] = BinaryF1Score(threshold=threshold)
            self.test_metrics["f1"] = BinaryF1Score(threshold=threshold)
        if "auc" in metrics:
            self.val_metrics["auc"] = BinaryAUROC(threshold=threshold)
            self.test_metrics["auc"] = BinaryAUROC(threshold=threshold)

    def set_labels(self, labels):
        self.labels = labels  # Set labels from dataset
        self.unique_labels = np.unique(self.labels)
        print(f"Model labels: {self.unique_labels}")

    def save_model(self, path: str, epoch: int = None):
        """
        Save the model and its weights to the given path.

        Args:
            path (str): Path where the model should be saved
        """
        model = self.model.to("cpu")
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        if epoch is not None:
            path = os.path.join(path, f"model_epoch_{epoch}.pth")
        else:
            path = os.path.join(path, "model.pth")
        torch.save(model, path)

    def save_hparams(self, path: str):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        path = os.path.join(path, "hparams.json")
        with open(path, "w") as f:
            json.dump(self.params, f)

    def load_hparams(self, path: str):
        self.params = json.load(open(path, "r"))

    def create_weighted_sampler(self, dataset):
        """
        Create a WeightedRandomSampler for class imbalances
        """
        class_counts = {l: np.sum(dataset.labels == l) for l in self.unique_labels}
        weights = {l: 1.0 / max(class_counts[l], 1) for l in self.unique_labels}
        sample_weights = np.array([weights[l] for l in dataset.labels])
        return WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True
        )

    def _prepare_dataloaders(self, train_dataset, val_dataset):
        if self.use_weighted_sampler:
            sampler = self.create_weighted_sampler(train_dataset)
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, sampler=sampler, num_workers=22
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=22
            )

        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=22)
        return train_loader, val_loader

    def _general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + "_loss"] for x in outputs]).mean()
        total_correct = (
            torch.stack([x[mode + "_n_correct"] for x in outputs]).sum().cpu().numpy()
        )
        acc = total_correct / len(self.dataset[mode])
        return avg_loss, acc

    def _training_step(self, batch, loss_fn):
        """
        Perform a single training step on the given batch.

        Args:
            batch: image, labels
            loss_fn: loss function to use for training

        Returns:
            loss: loss value for the batch
        """

        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)

        # Forward pass
        outputs = self.forward(images)

        # Compute loss
        loss = loss_fn(outputs, labels)

        return loss

    def _validation_step(self, batch, metrics, loss_fn):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)

        # Forward pass
        outputs = self.forward(images)

        # Compute loss
        loss = loss_fn(outputs, labels)

        # Activate the outputs to get the predictions
        outputs = torch.sigmoid(outputs).squeeze()

        # Update metrics
        for metric_name, metric in metrics.items():
            # TODO (for the future): doesn't work for multiclass
            # TODO check what input is needed for the metrics
            # if metric_name == "accuracy":
            #    metric.update(outputs, labels.squeeze().long())
            labels = labels.squeeze().int()
            metric.update(outputs, labels)  # Ensure labels are 1D

        return loss

    def _configure_optimizer(self):
        if self.optimizer_name.lower() == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.lr)
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, train_dataset, val_dataset, tb_logger, path):
        # Prepare data loaders
        train_loader, val_loader = self._prepare_dataloaders(train_dataset, val_dataset)

        optimizer = self._configure_optimizer()
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(len(train_loader) / 5), gamma=0.7
        )
        loss_fn = self.loss_fn  # Use the loss function configured in the model

        self.model = self.model.to(self.device)

        print(f"Device used {self.device}")

        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            train_loop = tqdm(
                train_loader,
                desc=f"Training Epoch {epoch + 1}/{self.num_epochs}",
                total=len(train_loader),
                ncols=200,
            )
            training_loss = 0

            for train_iteration, batch in enumerate(train_loop):
                optimizer.zero_grad()
                loss = self._training_step(batch, loss_fn)
                loss.backward()
                optimizer.step()

                training_loss += loss.item()
                train_loop.set_postfix(
                    train_loss=f"{training_loss / (train_iteration + 1):.6f}"
                )

                # Log training loss with tensorboard
                tb_logger.add_scalar(
                    "Train/loss",
                    loss.item(),
                    epoch * len(train_loader) + train_iteration,
                )
                # Log training loss with wandb
                wandb.log({'epoch': epoch, 'train_loss': loss.item()})
                
            scheduler.step()

            # Validation
            self.model.eval()
            val_loop = tqdm(
                val_loader,
                desc=f"Validation Epoch {epoch + 1}/{self.num_epochs}",
                total=len(val_loader),
                ncols=200,
            )

            validation_loss = 0

            with torch.no_grad():
                # Reset metrics before loop
                for metric in self.val_metrics.values():
                    metric.reset()
                for val_iteration, batch in enumerate(val_loop):
                    loss = self._validation_step(batch, self.val_metrics, loss_fn)
                    validation_loss += loss.item()

                    # Update progress bar
                    val_loop.set_postfix(
                        val_loss=f"{validation_loss / (val_iteration + 1):.6f}",
                    )

            # Validation metrics computation and logging
            validation_loss /= len(val_loader)  # Average validation loss

            if tb_logger:
                tb_logger.add_scalar("Val/loss", validation_loss, epoch)
            
            # Log validation loss with wandb
            wandb.log({'validation_loss': validation_loss})

            for metric_name, metric in self.val_metrics.items():
                try:
                    metric_value = metric.compute()
                except ZeroDivisionError:
                    metric_value = 0.0  # Handle edge case
                tb_logger.add_scalar(f"Val/{metric_name}", metric_value, epoch)
                
                # Log validation metrics with wandb
                wandb.log({f"Val/{metric_name}": metric_value})

            # Save model at specified intervals
            if self.save_epoch and (epoch + 1) % self.save_epoch == 0:
                self.save_model(path, epoch + 1)

    def test(self, test_dataset, tb_logger):
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=22
        )

        self.model.eval()
        test_loop = tqdm(
            test_loader,
            desc="Testing",
            total=len(test_loader),
            ncols=200,
        )

        test_loss = 0

        with torch.no_grad():
            for metric in self.test_metrics.values():
                metric.reset()
            for test_iteration, batch in enumerate(test_loop):
                # Perform the test step and accumulate the loss
                loss = self._validation_step(
                    batch,
                    self.test_metrics,
                    self.loss_fn,
                )
                test_loss += loss.item()

                # Update progress bar
                test_loop.set_postfix(
                    test_loss=f"{test_loss / (test_iteration + 1):.6f}",
                )

        # Average test loss over the entire dataset
        test_loss /= len(test_loader)

        if tb_logger:
            tb_logger.add_scalar("Test/loss", test_loss)  # Log test loss
 
        # Log test loss with wandb
        wandb.log({'test_loss': test_loss})

        # Compute and log test metrics
        for metric_name, metric in self.test_metrics.items():
            try:
                metric_value = metric.compute()
            except ZeroDivisionError:
                metric_value = 0.0  # Handle edge case
            tb_logger.add_scalar(f"Test/{metric_name}", metric_value)  # Log metrics
            
            # Log test metrics with wandb
            wandb.log({'f"Test/{metric_name}"': metric_value})

            # TODO Test metrics computation and logging: Done
            # Analogous to validation metrics just use self.test_metrics: Done


class ResNet50OneStage(AbstractOneStageModel):
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
            num_labels,
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
            num_labels,
        )

        # Set device
        self.model.to(self.device)

    def forward(self, x):
        # TODO remove
        #self.model = self.model.to(
        #    self.device
        #)  # Ensure the entire model is on the correct device
        #x = x.to(self.device)
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
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_labels)

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
