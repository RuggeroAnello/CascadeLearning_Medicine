import torch
import json
import os
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from torcheval.metrics import (
    BinaryRecall,
    BinaryPrecision,
    BinaryF1Score,
    BinaryAccuracy,
    BinaryAUROC,
)


class AbstractMultiStageModel(torch.nn.Module):
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
            targets (dict): Dictionary containing the target labels.
        """
        super().__init__()
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = targets.keys()
        self.unique_labels = None

        # Set hyperparameters
        self.params = params
        self._configure_hyperparameters(params)
        self._configure_metrics(params)

    def _configure_metrics(self, params: dict):
        """
        Configure the metrics to be used for validation and testing.

        Args:
            params (dict): Dictionary containing the hyperparameters.
        """
        # TODO: Add more metrics if needed
        self.val_metrics = {}
        self.test_metrics = {}

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
            if "auc" in metrics:
                self.val_metrics[label]["auc"] = BinaryAUROC(threshold=threshold)
                self.test_metrics[label]["auc"] = BinaryAUROC(threshold=threshold)

    def _configure_hyperparameters(self, params: dict):
        """
        Configure the hyperparameters for the model.

        Args:
            params (dict): Dictionary containing the hyperparameters.
        """
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
        self.confidence_threshold_first_ap_pa = params.get(
            "confidence_threshold_first_ap_pa", 0.5
        )
        self.num_workers = params.get("num_workers", 22)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Raises:
            NotImplementedError: This method should be implemented in the subclass.
        """
        raise NotImplementedError

    @property
    def name(self):
        """
        Return the name of the model.

        Raises:
            NotImplementedError: This method should be implemented in the subclass.
        """
        raise NotImplementedError

    def save_hparams(self, path: str):
        """
        Save the hyperparameters to a file.

        Args:
            path (str): Path to save the hyperparameters.
        """
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        path = os.path.join(path, "hparams.json")
        with open(path, "w") as f:
            json.dump(self.params, f)

    def load_hparams(self, path: str):
        """
        Load the hyperparameters from a file.

        Args:
            path (str): Path to load the hyperparameters.
        """
        self.params = json.load(open(path, "r"))

    def _validation_step(self, batch, metrics, loss_fn):
        """
        Perform a single validation step.

        Args:
            batch (tuple): Tuple containing the input images and labels.
            metrics (dict): Dictionary containing the metrics to be computed.
            loss_fn (torch.nn.Module): Loss function to be used.

        Returns:
            torch.Tensor: Loss value.
        """
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)

        # Forward pass
        outputs = self.forward(images)

        # Compute loss
        loss = loss_fn(outputs, labels)

        # Activate the outputs to get the predictions
        outputs = torch.sigmoid(outputs).squeeze()

        # Update metrics
        for i, label in enumerate(self.labels):
            for _, metric in metrics[label].items():
                if len(self.labels) == 1:
                    labels_metric = labels.squeeze().int()
                    outputs_metric = outputs.squeeze()
                    metric.update(outputs_metric, labels_metric)
                else:
                    labels_metric = labels[:, i].squeeze().int()
                    outputs_metric = outputs[:, i]
                    metric.update(outputs_metric, labels_metric)

        return loss

    def test(self, test_dataset, tb_logger=None, log_wandb=True):
        """
        Test the model on the given dataset.

        Args:
            test_dataset (torch.utils.data.Dataset): Dataset to test the model on.
            tb_logger (torch.utils.tensorboard.SummaryWriter, optional): Tensorboard logger to log the test results. Defaults to None.
            log_wandb (bool, optional): Whether to log the test results with wandb. Defaults to True.
        """
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        self.model_ap_pa_classification.eval()
        self.model_ap.eval()
        self.model_pa.eval()

        test_loop = tqdm(
            test_loader,
            desc="Testing",
            total=len(test_loader),
            ncols=200,
        )

        test_loss = 0.0

        with torch.no_grad():
            for label in self.test_metrics.keys():
                for metric in self.test_metrics[label].values():
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
        if log_wandb:
            wandb.log({"test_loss": test_loss})
        print(f"Test loss: {test_loss}")

        # Compute and log test metrics
        for label in self.test_metrics.keys():
            for metric_name, metric in self.test_metrics[label].items():
                try:
                    metric_value = metric.compute()
                except ZeroDivisionError:
                    metric_value = 0.0  # Handle edge case
                # Log test metrics with tensorboard
                if tb_logger:
                    tb_logger.add_scalar(f"Test/{label}_{metric_name}", metric_value)
                # Log test metrics with wandb
                if log_wandb:
                    wandb.log({'f"Test/{metric_name}"': metric_value})
                print(f"Test {label} {metric_name}: {metric_value}")


class TwoStageModel(AbstractMultiStageModel):
    """
    Two-stage model. The first stage classifies the images into AP and PA views. The second stage processes the images based on the classification.

    Args:
        AbstractMultiStageModel (torch.nn.Module): Abstract class for multi-stage models.
    """

    def __init__(
        self,
        params: dict,
        targets: dict,
        model_ap_pa_classification: str,
        model_ap: str,
        model_pa: str,
    ):
        """
        Initialize the two-stage model with the given hyperparameters.

        Args:
            params (dict): Dictionary containing the hyperparameters.
            targets (dict): Dictionary containing the target labels.
            model_ap_pa_classification (str): Path to the model for AP-PA classification.
            model_ap (str): Path to the model for AP images.
            model_pa (str): Path to the model for PA images.
        """
        super().__init__(
            params=params,
            targets=targets,
        )

        # Define the two models
        self.model_ap_pa_classification = torch.load(
            model_ap_pa_classification, weights_only=False
        )
        self.model_ap = torch.load(model_ap, weights_only=False)
        self.model_pa = torch.load(model_pa, weights_only=False)

        # Move models to device
        self.model_ap_pa_classification.to(self.device)
        self.model_ap.to(self.device)
        self.model_pa.to(self.device)

        # Save the used models
        self.params["model_ap_pa_classification"] = model_ap_pa_classification
        self.params["model_ap"] = model_ap
        self.params["model_pa"] = model_pa

    def forward(self, x):
        """
        Forward pass of the two-stage model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out_stage_one = self.model_ap_pa_classification(x)
        conf = torch.sigmoid(out_stage_one)
        pred = conf > self.confidence_threshold_first_ap_pa

        idx_ap = pred.squeeze().nonzero(as_tuple=True)[0]
        idx_pa = (~pred.squeeze()).nonzero(as_tuple=True)[0]

        x_ap = x[idx_ap]
        x_pa = x[idx_pa]

        out_ap = self.model_ap(x_ap) if x_ap.size(0) > 0 else None
        out_pa = self.model_pa(x_pa) if x_pa.size(0) > 0 else None

        output = torch.zeros(
            (x.size(0), out_ap.size(1) if out_ap is not None else out_pa.size(1))
        ).to(self.device)
        if out_ap is not None:
            output[idx_ap] = out_ap
        if out_pa is not None:
            output[idx_pa] = out_pa

        return output

    @property
    def name(self):
        return "TwoStageModel_AP-PA"
