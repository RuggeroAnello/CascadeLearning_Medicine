import torch
import json
import torchvision
import os
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
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
            "BCEWithLogitsLoss": torch.nn.BCEWithLogitsLoss()
        # Add other loss functions here if needed
    }   # Set the loss function, defaulting to BCEWithLogitsLoss if not specified
        self.loss_fn = loss_fn_mapping.get(loss_fn_str, torch.nn.BCEWithLogitsLoss())
    
        self.save_epoch = params.get("save_epoch", 1)
        self.use_weighted_sampler = params.get("use_weighted_sampler", False)
        self.save_epoch = params.get("save_epoch", 1)
        self.use_weighted_sampler = params.get("use_weighted_sampler", False)
    
    def set_labels(self, labels):
        self.labels = labels # Set labels from dataset
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
        torch.save(model.state_dict(), path)

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
        weights = {l: 1.0 / class_counts[l] for l in self.unique_labels}
        sample_weights = np.array([weights[l] for l in dataset.labels])
        return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        
    def _prepare_dataloaders(self, train_dataset, val_dataset):
        if self.use_weighted_sampler:
            sampler = self.create_weighted_sampler(train_dataset)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler)
        else:
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader
        
    def _general_step(self, batch, loss_fn=F.cross_entropy):
        if len(batch) == 0:  # Defensive check for empty batch
            return 0, 0
        
        images, labels = batch 
        images, labels = images.to(self.device), labels.to(self.device) 

        # Forward pass
        outputs = self.forward(images)
        
        # Compute loss
        # Ensure loss function matches
        if labels.dim() > 1:  # For multi-label classification
            loss = loss_fn(outputs, labels)
        else:  
            loss = loss_fn(outputs, labels.long())

        # Compute number of correct predictions
        preds = outputs.argmax(dim=1)  
        if labels.dim() > 1:  
            n_correct = (preds.unsqueeze(1) == labels.nonzero(as_tuple=True)[1]).sum().item()
        else: 
            n_correct = (preds == labels).sum().item()
        
        return loss, n_correct

    def _general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + "_loss"] for x in outputs]).mean()
        total_correct = (
            torch.stack([x[mode + "_n_correct"] for x in outputs]).sum().cpu().numpy()
        )
        acc = total_correct / len(self.dataset[mode])
        return avg_loss, acc

    def _training_step(self, batch, loss_fn):
        loss, _ = self._general_step(batch, loss_fn=loss_fn)
        return loss

    def _validation_step(self, batch, loss_fn=F.cross_entropy):
        loss, n_correct = self._general_step(batch, loss_fn=loss_fn)
        return loss, n_correct

    def _test_step(self, batch):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        outputs = self.forward(images)
        loss = self.loss_fn(outputs, labels)
        n_correct = (outputs.argmax(dim=1) == labels).sum()
        return loss, n_correct, outputs  

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
                scheduler.step()

                training_loss += loss.item()
                train_loop.set_postfix(train_loss=f"{training_loss / (train_iteration + 1):.6f}")

                # Log training loss
                tb_logger.add_scalar("Train/loss", loss.item(), epoch * len(train_loader) + train_iteration)

            # Validation
            self.model.eval()
            val_loop = tqdm(
                val_loader,
                desc=f"Validation Epoch {epoch + 1}/{self.num_epochs}",
                total=len(val_loader),
                ncols=200,
            )
            validation_loss = 0
            total_correct = 0
            all_labels = []
            all_preds = []

            with torch.no_grad():
                with torch.no_grad():
                    for val_iteration, batch in enumerate(val_loop):
                        loss, n_correct = self._validation_step(batch, loss_fn)
                        validation_loss += loss.item()
                        total_correct += n_correct

                        # Collect labels and predictions for metrics
                        images, labels = batch
                        labels = labels.to(self.device)
                        outputs = self.forward(images.to(self.device))
                        preds = outputs.argmax(dim=1)

                        all_labels.append(labels.cpu())
                        all_preds.append(preds.cpu())

                        # Running accuracy
                        running_accuracy = total_correct / ((val_iteration + 1) * val_loader.batch_size)

                        # Update progress bar
                        val_loop.set_postfix(
                            val_loss=f"{validation_loss / (val_iteration + 1):.6f}",
                            val_acc=f"{running_accuracy:.4f}",
        )

                    # Log validation loss
                    tb_logger.add_scalar("Val/loss", loss.item(), epoch * len(val_loader) + val_iteration)

            # Validation metrics computation and logging
            validation_loss /= len(val_loader)  # Average validation loss
            validation_acc = total_correct / len(val_loader.dataset)  # Accuracy

            # Concatenate all labels and predictions
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)

            # Compute additional metrics
            metrics = {
                "accuracy": validation_acc
            }

            if "precision" in self.params:  # Check if metric is configured
                metrics["precision"] = precision_score(all_labels, all_preds, average="weighted")
            if "recall" in self.params:
                metrics["recall"] = recall_score(all_labels, all_preds, average="weighted")
            if "f1" in self.params:
                metrics["f1"] = f1_score(all_labels, all_preds, average="weighted")
            if "roc_auc" in self.params and len(self.unique_labels) == 2:  # Binary classification
                probabilities = F.softmax(outputs, dim=1)[:, 1]  # Use probabilities for AUC
                metrics["roc_auc"] = roc_auc_score(all_labels, probabilities)

            # Log validation metrics to TensorBoard
            tb_logger.add_scalar("Val/loss", validation_loss, epoch)
            tb_logger.add_scalar("Val/accuracy", validation_acc, epoch)

            for metric_name, metric_value in metrics.items():
                tb_logger.add_scalar(f"Val/{metric_name}", metric_value, epoch)

            # Save model at specified intervals
            if self.save_epoch and (epoch + 1) % self.save_epoch == 0:
                self.save_model(path, epoch + 1)

    def test(self, test_dataset, tb_logger, path):
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        test_loss = 0
        total_correct = 0
        all_labels = []
        all_preds = []

        self.model.eval()
        with torch.no_grad():
            for _, batch in enumerate(tqdm(test_loader, desc="Testing", ncols=200)):
                loss, n_correct, outputs = self._test_step(batch)
                test_loss += loss.item()
                total_correct += n_correct.item()

                # Store labels and predictions for metric calculations
                _, labels = batch
                labels = labels.to(self.device)
                preds = outputs.argmax(dim=1)
                all_labels.append(labels.cpu())
                all_preds.append(preds.cpu())

        # Calculate average loss and accuracy
        test_loss /= len(test_loader)
        test_acc = total_correct / len(test_loader.dataset)

        # Concatenate all labels and predictions
        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)

        # Calculate metrics
        metrics = {}
        metrics["accuracy"] = test_acc
        if "precision" in self.params:
            metrics["precision"] = precision_score(all_labels, all_preds, average="weighted")
        if "recall" in self.params:
            metrics["recall"] = recall_score(all_labels, all_preds, average="weighted")
        if "f1" in self.params:
            metrics["f1"] = f1_score(all_labels, all_preds, average="weighted")
        if "roc_auc" in self.params and len(self.unique_labels) == 2:  # ROC AUC only for binary classification
            metrics["roc_auc"] = roc_auc_score(all_labels, F.softmax(outputs, dim=1)[:, 1])

        # Log results
        for key, value in metrics.items():
            tb_logger.add_scalar(f"Test/{key}", value)

        return metrics


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
        self.model = torchvision.models.resnet50(weights="IMAGENET1K_V2")

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
        self.model = torchvision.models.resnet18(weights="IMAGENET1K_V1:")

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
        x = x.to(self.device)
        return self.model(x)

    @property
    def name(self):
        return "ResNet18OneStage"