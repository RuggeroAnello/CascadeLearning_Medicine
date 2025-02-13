import pandas as pd
import torch
import json
import torchvision
import os
import wandb
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torcheval.metrics import (
    BinaryRecall,
    BinaryPrecision,
    BinaryF1Score,
    BinaryAccuracy,
    BinaryAUROC,
    BinaryPrecisionRecallCurve,
    AUC,
    MultilabelAccuracy,
    MultilabelAUPRC,
    MultilabelPrecisionRecallCurve,
    BinaryConfusionMatrix,
)
from torchmetrics.classification import MultilabelMatthewsCorrCoef
from torch.utils.tensorboard import SummaryWriter
from model.loss import MultilabelFocalLoss

from tqdm import tqdm


class AbstractOneStageModel(torch.nn.Module):
    """
    Abstract class for one-stage models. The class provides a template for training and testing one-stage models. The class should be inherited by specific one-stage models. The specific models should implement the forward method and the name property.
    """

    def __init__(
        self,
        params: dict,
        targets: dict,
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
        self.labels = targets.keys()

        # Save hyperparameters
        self.params = params
        self._configure_hyperparameters(params)
        self._configure_metrics(params)

        # Save results
        self.results = {}

        # Best model results
        self.best_val_loss = np.inf
        self.best_auprc = (
            -1 if "multilabel_auprc" in self.params.get("metrics", []) else 0
        )
        self.best_mcc = -1 if "mcc" in self.params.get("metrics", []) else 0
        self.best_epoch = -1

    def forward(self, x):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError

    def _configure_hyperparameters(self, params: dict) -> None:
        """
        Configure the hyperparameters for the model.

        Args:
            params (dict): Dictionary containing the hyperparameters.
        """
        self.lr = params.get("lr", 1e-3)
        self.batch_size = params.get("batch_size", 32)
        self.num_epochs = params.get("num_epochs", 100)
        self.optimizer_name = params.get("optimizer", "adam")
        self.label_smoothing = params.get("label_smoothing", 0.2)
        # Map string loss function names to actual loss function classes
        loss_fn_str = params.get("loss_fn")
        if loss_fn_str is None:
            raise ValueError("Loss function not specified or invalid loss function!")
        loss_fn_mapping = {
            "cross_entropy": torch.nn.CrossEntropyLoss(),
            "mse_loss": torch.nn.MSELoss(),
            "BCEWithLogitsLoss": torch.nn.BCEWithLogitsLoss(),
        }  # Set the loss function, defaulting to BCEWithLogitsLoss if not specified
        if loss_fn_str == "weighted_bce_loss":
            self.use_loss_fn_val = True
            pos_weights_train = torch.tensor(params.get("pos_weights_train"))
            pos_weights_val = torch.tensor(params.get("pos_weights_val"))
            pos_weights_train = pos_weights_train.to(self.device)
            pos_weights_val = pos_weights_val.to(self.device)
            self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights_train)
            self.loss_fn_val = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights_val)
        elif loss_fn_str == "multilabel_focal_loss":
            self.use_loss_fn_val = True
            # Pos_weights
            if (
                "pos_weights_train" in params
                and params.get("pos_weights_train") is not None
            ):
                pos_weights_train = torch.tensor(params.get("pos_weights_train"))
                pos_weights_train = pos_weights_train.to(self.device)
            else:
                pos_weights_train = None
            if (
                "pos_weights_val" in params
                and params.get("pos_weights_val") is not None
            ):
                pos_weights_val = torch.tensor(params.get("pos_weights_val"))
                pos_weights_val = pos_weights_val.to(self.device)
            else:
                pos_weights_val = None
            # Class weights
            if (
                "class_weights_train" in params
                and params.get("class_weights_train") is not None
            ):
                class_weights_train = torch.tensor(params.get("class_weights_train"))
                class_weights_train = class_weights_train.to(self.device)
            else:
                class_weights_train = None
            if (
                "class_weights_val" in params
                and params.get("class_weights_val") is not None
            ):
                class_weights_val = torch.tensor(params.get("class_weights_val"))
                class_weights_val = class_weights_val.to(self.device)
            else:
                class_weights_val = None

            self.loss_fn = MultilabelFocalLoss(
                gamma=params.get("gamma", 2),
                alpha=params.get("alpha", None),
                reduction=params.get("reduction", "mean"),
                num_classes=len(self.labels),
                pos_weight=pos_weights_train,
                class_weights=class_weights_train,
            )
            self.loss_fn_val = MultilabelFocalLoss(
                gamma=params.get("gamma", 2),
                alpha=params.get("alpha", None),
                reduction=params.get("reduction", "mean"),
                num_classes=len(self.labels),
                pos_weight=pos_weights_val,
                class_weights=class_weights_val,
            )
        else:
            self.use_loss_fn_val = False
            self.loss_fn = loss_fn_mapping.get(
                loss_fn_str, torch.nn.BCEWithLogitsLoss()
            )
        self.use_weighted_sampler = params.get("use_weighted_sampler", False)
        self.save_epoch = params.get("save_epoch", 1)
        self.confidence_threshold = params.get("confidence_threshold", 0.5)
        self.num_workers = params.get("num_workers", 22)
        self.lr_decay_gamma = params.get("lr_decay_gamma", 0.9)
        self.lr_decay_period = params.get("lr_decay_period", 2)

    def _configure_metrics(self, params: dict) -> None:
        """
        Configure the metrics for the model.

        Args:
            params (dict): Dictionary containing the hyperparameters.
        """
        self.val_metrics = {}
        self.test_metrics = {}

        self.val_metrics_multilabel = {}
        self.test_metrics_multilabel = {}

        metrics = params.get("metrics", [])

        threshold = params.get("confidence_threshold", 0.5)

        for label in self.labels:
            self.val_metrics[label] = {}
            self.test_metrics[label] = {}

            # Accuracy is always calculated
            self.val_metrics[label]["accuracy"] = BinaryAccuracy(threshold=threshold)
            self.test_metrics[label]["accuracy"] = BinaryAccuracy(threshold=threshold)

            if "precision" in metrics:
                self.val_metrics[label]["precision"] = BinaryPrecision(
                    threshold=threshold
                )
                self.test_metrics[label]["precision"] = BinaryPrecision(
                    threshold=threshold
                )
            if "recall" in metrics:
                self.val_metrics[label]["recall"] = BinaryRecall(threshold=threshold)
                self.test_metrics[label]["recall"] = BinaryRecall(threshold=threshold)
            if "f1" in metrics:
                self.val_metrics[label]["f1"] = BinaryF1Score(threshold=threshold)
                self.test_metrics[label]["f1"] = BinaryF1Score(threshold=threshold)
            if "auroc" in metrics:
                self.val_metrics[label]["auroc"] = BinaryAUROC()
                self.test_metrics[label]["auroc"] = BinaryAUROC()
            if "auc" in metrics:
                self.val_metrics[label]["auc"] = BinaryPrecisionRecallCurve()
                self.test_metrics[label]["auc"] = BinaryPrecisionRecallCurve()
            if "confusion_matrix" in metrics:
                self.val_metrics[label]["confusion_matrix"] = BinaryConfusionMatrix()
                self.test_metrics[label]["confusion_matrix"] = BinaryConfusionMatrix()

        # Multilabel metrics are only calculated if there are multiple labels
        if len(self.labels) > 1:
            if "multilabel_accuracy" in metrics:
                self.val_metrics_multilabel["multilabel_accuracy"] = MultilabelAccuracy(
                    threshold=threshold
                )
                self.test_metrics_multilabel["multilabel_accuracy"] = (
                    MultilabelAccuracy(threshold=threshold)
                )
            if "multilabel_auprc" in metrics:
                self.val_metrics_multilabel["multilabel_auprc"] = MultilabelAUPRC(
                    num_labels=len(self.labels)
                )
                self.test_metrics_multilabel["multilabel_auprc"] = MultilabelAUPRC(
                    num_labels=len(self.labels)
                )
            if "multilabel_precision_recall_curve" in metrics:
                self.val_metrics_multilabel["multilabel_precision_recall_curve"] = (
                    MultilabelPrecisionRecallCurve(num_labels=len(self.labels))
                )
                self.test_metrics_multilabel["multilabel_precision_recall_curve"] = (
                    MultilabelPrecisionRecallCurve(num_labels=len(self.labels))
                )
            if "mcc" in metrics:
                self.val_metrics_multilabel["mcc"] = MultilabelMatthewsCorrCoef(
                    num_labels=len(self.labels), threshold=threshold
                )
                self.test_metrics_multilabel["mcc"] = MultilabelMatthewsCorrCoef(
                    num_labels=len(self.labels), threshold=threshold
                )
                self.val_metrics_multilabel["mcc"].to(self.device)
                self.test_metrics_multilabel["mcc"].to(self.device)

    def save_model(
        self, path: str, epoch: int = None, best: bool = False, substring=""
    ) -> None:
        """
        Save the model and its weights to the given path.

        Args:
            path (str): Path where the model should be saved
            epoch (int, optional): Epoch number to include in the model name. Defaults to None.
            best (bool, optional): Whether to save the best model. Defaults to False.
            substring (str, optional): Substring to include in the model name. Defaults to "". If not empty, the substring is appended to the model name.
        """
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        if epoch is not None:
            path = os.path.join(path, f"model_epoch_{epoch}.pth")
        elif best:
            path = os.path.join(path, f"best_model_{substring}.pth")
        else:
            path = os.path.join(path, "model.pth")
        # self.model.to("cpu") if required move the model to self.device again after saving!
        torch.save(self.model, path)

    def load_model(self, path: str) -> None:
        """
        Load the model from the given path.

        Args:
            path (str): Path where the model is saved
        """
        self.model = torch.load(path, weights_only=False)

    def save_hparams(self, path: str) -> None:
        """
        Save the hyperparameters to the given path.

        Args:
            path (str): Path where the hyperparameters should be saved
        """
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        path = os.path.join(path, "hparams.json")
        params = self.params.copy()
        if "pos_weights_train" in self.params:
            del params["pos_weights_train"]
        if "pos_weights_val" in self.params:
            del params["pos_weights_val"]
        with open(path, "w") as f:
            json.dump(params, f)

    def load_hparams(self, path: str) -> None:
        """
        Load the hyperparameters from the given path.

        Args:
            path (str): Path where the hyperparameters are saved
        """
        self.params = json.load(open(path, "r"))

    def get_num_labels(self, targets: dict) -> int:
        """
        Get the number of labels in the dataset.

        Args:
            targets (dict): Dictionary containing the target labels.

        Returns:
            int: Number of labels in the dataset.
        """
        return len(targets)

    def create_weighted_sampler(
        self, dataset: torch.utils.data.Dataset
    ) -> WeightedRandomSampler:
        """
        Create a WeightedRandomSampler for multi-label class imbalances.
        Dynamically computes class weights based on the dataset's labels.

        Args:
            dataset (torch.utils.data.Dataset): Dataset for which to create the sampler.

        Returns:
            WeightedRandomSampler: Weighted sampler for the dataset.
        """
        data_labels = dataset.labels

        if len(data_labels.shape) == 1:  # Single-label case
            unique_labels, counts = np.unique(data_labels, return_counts=True)
            # Compute weights for each class
            weights = {
                label: 1.0 / max(count, 1)
                for label, count in zip(unique_labels, counts)
            }
            sample_weights = np.array([weights[label] for label in data_labels])

        else:  # Multi-label case
            num_samples = len(data_labels)
            sample_weights = np.ones(num_samples)
            # Calculate weights for each label column (multi-label scenario)
            for i in range(data_labels.shape[1]):
                pos_count = np.sum(data_labels[:, i] == 1)
                neg_count = np.sum(data_labels[:, i] == 0)
                # Avoid division by zero
                pos_weight = num_samples / (2 * max(pos_count, 1))
                neg_weight = num_samples / (2 * max(neg_count, 1))
                # Update sample weights based on positive/negative samples
                sample_weights[data_labels[:, i] == 1] *= pos_weight
                sample_weights[data_labels[:, i] == 0] *= neg_weight

            # Normalize weights
            sample_weights = sample_weights / sample_weights.sum() * num_samples

        # Return the sampler
        return WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True
        )

    def _prepare_dataloaders(
        self,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
    ) -> tuple:
        """
        Prepare the data loaders for training and validation.

        Args:
            train_dataset (torch.utils.data.Dataset): The training dataset.
            val_dataset (torch.utils.data.Dataset): The validation dataset.

        Returns:
            tuple: Tuple containing the training and validation data loaders.
        """
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

    def _training_step(self, batch: tuple, loss_fn: torch.nn.Module) -> torch.Tensor:
        """
        Perform a single training step on the given batch.

        Args:
            batch (tuple): Tuple containing the input images and labels.
            loss_fn (torch.nn.Module): Loss function to be used.

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

    def _validation_step(
        self,
        batch: tuple,
        metrics: dict,
        multilabel_metrics: dict,
        loss_fn: torch.nn.Module,
        test_labels: list = None,
    ) -> torch.Tensor:
        """
        Perform a single validation step on the given batch.

        Args:
            batch (tuple): Tuple containing the input images and labels.
            metrics (dict): Dictionary containing the metrics to be computed.
            multilabel_metrics (dict): Dictionary containing the multilabel metrics to be computed.
            loss_fn (torch.nn.Module): Loss function to be used.
            test_labels (list, optional): Idx of labels to test. The idx is the relative position of the labels in the output vector. The test labels have to be the of the same size as num_labels. Defaults to None.

        Returns:
            torch.Tensor: Loss value.
        """
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)

        # Forward pass
        outputs = self.forward(images)

        # Select only the test labels
        if test_labels is not None:
            outputs = outputs[:, test_labels]

        # Compute loss
        loss = loss_fn(outputs, labels)

        # Activate the outputs to get the predictions
        outputs = torch.sigmoid(outputs)

        # Update metrics
        # Iterate over all labels
        for i, label in enumerate(self.labels):
            # Update metrics for each label
            for _, metric in metrics[label].items():
                if len(self.labels) == 1:
                    labels_metric = labels.squeeze().int()
                    outputs_metric = outputs.squeeze()
                    metric.update(outputs_metric, labels_metric)
                else:
                    labels_metric = labels[:, i].squeeze().int()
                    outputs_metric = outputs[:, i]
                    metric.update(outputs_metric, labels_metric)  # Ensure labels are 1D

        # Update multilabel metrics
        for name, metric in multilabel_metrics.items():
            if name == "mcc":
                metric.update(outputs, labels.to(torch.int64))
            else:
                metric.update(outputs, labels)

        return loss

    def _configure_optimizer(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer for the model.

        Returns:
            torch.optim.Optimizer: Optimizer for the model.
        """
        if self.optimizer_name.lower() == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.lr)
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(
        self,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        path: str,
        tb_logger: SummaryWriter = None,
        log_wandb: bool = True,
    ):
        """
        Train the model on the given dataset.

        Args:
            train_dataset (torch.utils.data.Dataset): Training dataset
            val_dataset (torch.utils.data.Dataset): Validation dataset
            path (str): Path where the model should be saved
            tb_logger (SummaryWriter, optional): Tensorboard logger to log the training results. Defaults to None.
            log_wandb (bool, optional): Whether to log the training results with wandb. Defaults to True.
        """
        # Prepare data loaders
        train_loader, val_loader = self._prepare_dataloaders(train_dataset, val_dataset)

        # Configure optimizer and scheduler
        optimizer = self._configure_optimizer()
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(len(train_loader) * self.lr_decay_period),
            gamma=self.lr_decay_gamma,
        )

        # Use the loss function configured in the model
        loss_fn = self.loss_fn
        # Use the validation loss function if specified
        loss_fn_val = self.loss_fn_val if self.use_loss_fn_val else loss_fn

        # Initialize best metrics
        mcc = -1
        auprc = -1

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
                if tb_logger:
                    tb_logger.add_scalar(
                        "Train/loss",
                        loss.item(),
                        epoch * len(train_loader) + train_iteration,
                    )
                # Log training loss with wandb
                if log_wandb:
                    wandb.log({"epoch": epoch, "train_loss": loss.item()})

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
                for label in self.val_metrics.keys():
                    for metric in self.val_metrics[label].values():
                        metric.reset()
                # Reset multilabel metrics before loop
                for metric in self.val_metrics_multilabel.values():
                    metric.reset()
                for val_iteration, batch in enumerate(val_loop):
                    loss = self._validation_step(
                        batch=batch,
                        metrics=self.val_metrics,
                        multilabel_metrics=self.val_metrics_multilabel,
                        loss_fn=loss_fn_val,
                    )
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
                if log_wandb:
                    wandb.log({"validation_loss": validation_loss})

                for label in self.val_metrics.keys():
                    for metric_name, metric in self.val_metrics[label].items():
                        try:
                            metric_value = metric.compute()
                        except ZeroDivisionError:
                            metric_value = 0.0  # Handle edge case
                        # Log validation metrics with tensorboard
                        if tb_logger:
                            tb_logger.add_scalar(
                                f"Val/{label}_{metric_name}", metric_value, epoch
                            )
                        # Log validation metrics with wandb
                        if log_wandb:
                            wandb.log({f"Val/{label}_{metric_name}": metric_value})

                for metric_name, metric in self.val_metrics_multilabel.items():
                    try:
                        if metric_name == "auc":
                            # compute precision recall curve
                            metric_value = metric.compute()
                            # compute AUC
                            m = AUC()
                            m.update(metric_value[0], metric_value[1])
                            auc = m.compute()
                            if log_wandb:
                                wandb.log({f"Test/{metric_name}": auc})
                                wandb.log({"Test/PrecisionRecallCurve": metric_value})
                            metric_name = "PrecisionRecallCurve"
                        else:
                            metric_value = metric.compute()
                            if metric_name == "mcc":
                                mcc = metric_value
                            if metric_name == "multilabel_auprc":
                                auprc = metric_value
                    except ZeroDivisionError:
                        print("ZeroDivisionError")
                        metric_value = 0.0
                    if tb_logger:
                        tb_logger.add_scalar(
                            f"Val/multilabel_{metric_name}", metric_value, epoch
                        )
                    if log_wandb:
                        wandb.log({f"Val/multilabel_{metric_name}": metric_value})

                # Save model at specified intervals
                if self.save_epoch and (epoch + 1) % self.save_epoch == 0:
                    self.save_model(path, epoch=epoch + 1)

                # Save best model
                if validation_loss < self.best_val_loss:
                    self.best_val_loss = validation_loss
                    self.save_model(path, best=True)
                    self.best_epoch = epoch
                    self.save_model(path, best=True, substring=f"{epoch + 1}")

                if mcc > self.best_mcc:
                    self.best_mcc = mcc
                    self.save_model(path, best=True, substring=f"mcc_{epoch + 1}")

                if auprc > self.best_auprc:
                    self.best_auprc = auprc
                    self.save_model(path, best=True, substring=f"auprc_{epoch + 1}")

        print(f"Best validation loss: {self.best_val_loss} at epoch {self.best_epoch}")

    def test(
        self,
        test_dataset,
        name,
        tb_logger=None,
        log_wandb=False,
        test_labels: list = None,
    ) -> pd.DataFrame:
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        """
        Test the model on the given dataset.

        Args:
            test_dataset (torch.utils.data.Dataset): Dataset to test the model on.
            name (str): Name of the model.
            tb_logger (torch.utils.tensorboard.SummaryWriter, optional): Tensorboard logger to log the test results. Defaults to None.
            log_wandb (bool, optional): Whether to log the test results with wandb. Defaults to True.
            test_labels (list, optional): Idx of labels to test. The idx is the relative position of the labels in the output vector. The test labels have to be the of the same size as num_labels. Defaults to None.

        Returns:
            pd.DataFrame: Dataframe containing the test results.
        """
        # Use the validation loss function if specified
        loss_fn = self.loss_fn_val if self.use_loss_fn_val else self.loss_fn

        self.model.eval()
        test_loop = tqdm(
            test_loader,
            desc="Testing",
            total=len(test_loader),
            ncols=200,
        )

        test_loss = 0.0

        if test_labels is not None:
            if len(test_labels) != self.get_num_labels(self.labels):
                raise ValueError(
                    "The number of test labels has to be the same as the number of labels in the dataset."
                )

        with torch.no_grad():
            for label in self.test_metrics.keys():
                for metric in self.test_metrics[label].values():
                    metric.reset()
            for metric in self.test_metrics_multilabel.values():
                metric.reset()
            for test_iteration, batch in enumerate(test_loop):
                # Perform the test step and accumulate the loss
                loss = self._validation_step(
                    batch=batch,
                    metrics=self.test_metrics,
                    multilabel_metrics=self.test_metrics_multilabel,
                    loss_fn=loss_fn,
                    test_labels=test_labels,
                )
                test_loss += loss.item()

                # Update progress bar
                test_loop.set_postfix(
                    test_loss=f"{test_loss / (test_iteration + 1):.6f}",
                )

        # Average test loss over the entire dataset
        test_loss /= len(test_loader)

        # Log test loss with tensorboard
        if tb_logger:
            tb_logger.add_scalar("Test/loss", test_loss)  # Log test loss

        # Log test loss with wandb
        if log_wandb:
            wandb.log({"test_loss": test_loss})
        print(f"Test loss: {test_loss}")

        # Create a result dataframe
        res = pd.DataFrame()
        res["name"] = [name]

        # Compute and log test metrics
        for label in self.test_metrics.keys():
            for metric_name, metric in self.test_metrics[label].items():
                try:
                    if metric_name == "auc":
                        # compute precision recall curve
                        metric_value = metric.compute()

                        # compute auc
                        m = AUC()
                        m.update(metric_value[0], metric_value[1])
                        auc = m.compute()
                        auc = float(auc)

                        if log_wandb:
                            wandb.log({f"Test/{metric_name}": auc})
                            wandb.log({"Test/PrecisionRecallCurve": metric_value})

                        print(f"Test {label} {metric_name}: {auc}")
                        # Save the AUC in the result
                        res[f"{label}_{metric_name}"] = [f"{auc}"]
                        # Rename the metric name to PrecisionRecallCurve
                        metric_name = "PrecisionRecallCurve"
                    else:
                        metric_value = metric.compute()
                except ZeroDivisionError:
                    metric_value = 0.0  # Handle edge cases

                # Log test metrics with tensorboard
                if tb_logger:
                    tb_logger.add_scalar(f"Test/{label}_{metric_name}", metric_value)

                # Log test metrics with wandb
                if log_wandb:
                    wandb.log({f"Test/{metric_name}": metric_value})

                # Print all test metrics except PrecisionRecallCurve
                print(
                    f"Test {label} {metric_name}: {metric_value}"
                ) if metric_name != "PrecisionRecallCurve" else None
                res[f"{label}_{metric_name}"] = [f"{metric_value}"]

        for metric_name, metric in self.test_metrics_multilabel.items():
            try:
                metric_value = metric.compute()
            except ZeroDivisionError:
                metric_value = 0.0

            # Log test multilabel test metrics with tensorboard
            if tb_logger:
                tb_logger.add_scalar(f"Test/{metric_name}", metric_value)

            # Log test multilabel test metrics with wandb
            if log_wandb:
                wandb.log({f"Test/{metric_name}": metric_value})

            # Analogous to validation metrics just use self.test_metrics
            print(
                f"Test {metric_name}: {metric_value}"
            ) if metric_name != "multilabel_precision_recall_curve" else None
            res[metric_name] = [f"{metric_value}"]

        # Append res to result

        return res


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
            input_channels (int, optional): Number of input channels. Defaults to 1.
        """
        super().__init__(
            params=params,
            **kwargs,
        )

        # Load pretrained model
        # Best available weights (currently alias for IMAGENET1K_V2)
        self.model = torchvision.models.resnet50().to(self.device)

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
            input_channels (int, optional): Number of input channels. Defaults to 1.
        """
        super().__init__(
            params=params,
            **kwargs,
        )

        # Load pretrained model
        self.model = torchvision.models.resnet18()

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
        return self.model(x)

    @property
    def name(self):
        return "ResNet18OneStage"


class ResNet34OneStage(AbstractOneStageModel):
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
            input_channels (int, optional): Number of input channels. Defaults to 1.
        """
        super().__init__(
            params=params,
            **kwargs,
        )

        # Load pretrained model
        self.model = torchvision.models.resnet34().to(self.device)

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
