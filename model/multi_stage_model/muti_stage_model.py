import torch
import json
import torchvision
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.one_model.one_stage_models import (
    ResNet50OneStage,
    ResNet18OneStage,
    ResNet34OneStage,
)

from torcheval.metrics import (
    BinaryRecall,
    BinaryPrecision,
    BinaryF1Score,
    BinaryAccuracy,
    BinaryAUROC,
)


class TwoStageModel(torch.nn.Module):
    def __init__(
        self,
        params: dict,
        model_ap_pa_classification: str,
        model_ap: str,
        model_pa: str,
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

        # Set hyperparameters
        if "batch_size" in params:
            self.batch_size = params["batch_size"]

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

        self.params = params
        self._configure_metrics(params)
        self._configure_hyperparameters(params)

        # Save the used models
        self.params["model_ap_pa_classification"] = model_ap_pa_classification
        self.params["model_ap"] = model_ap
        self.params["model_pa"] = model_pa

    def _configure_metrics(self, params):
        # TODO: Add more metrics if needed
        self.val_metrics = {}
        self.test_metrics = {}

        metrics = params.get("metrics", [])

        # Accuracy is always calculated
        self.val_metrics["accuracy"] = (
            BinaryAccuracy()
        )  # These do NOT accept the confidence_threshold as argument -> done in validation_step
        self.test_metrics["accuracy"] = BinaryAccuracy()

        if "precision" in metrics:
            self.val_metrics["precision"] = BinaryPrecision()
            self.test_metrics["precision"] = BinaryPrecision()
        if "recall" in metrics:
            self.val_metrics["recall"] = BinaryRecall()
            self.test_metrics["recall"] = BinaryRecall()
        if "f1" in metrics:
            self.val_metrics["f1"] = BinaryF1Score()
            self.test_metrics["f1"] = BinaryF1Score()
        if "auc" in metrics:
            self.val_metrics["auc"] = BinaryAUROC()
            self.test_metrics["auc"] = BinaryAUROC()

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
        self.confidence_threshold_first_ap_pa = params.get(
            "confidence_threshold_first_ap_pa", 0.5
        )

    def forward(self, x):
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

    def save_hparams(self, path: str):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        path = os.path.join(path, "hparams.json")
        with open(path, "w") as f:
            json.dump(self.params, f)

    def load_hparams(self, path: str):
        self.params = json.load(open(path, "r"))

    def _validation_step(self, batch, metrics, loss_fn):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)

        # Forward pass
        outputs = self.forward(images)

        # Compute loss
        loss = loss_fn(outputs, labels)

        # Activate the outputs to get the predictions
        outputs = torch.sigmoid(outputs).squeeze()

        predictions = (outputs > self.confidence_threshold).long()

        # Update metrics
        for metric_name, metric in metrics.items():
            # TODO (for the future): doesn't work for multiclass
            # TODO check what input is needed for the metrics
            # if metric_name == "accuracy":
            #    metric.update(outputs, labels.squeeze().long())
            metric.update(predictions, labels.squeeze().long())  # Ensure labels are 1D

        return loss

    def test(self, test_dataset, tb_logger):
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=22
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

        with torch.no_grad():
            for metric in self.test_metrics.values():
                metric.reset()
            for test_iteration, batch in enumerate(test_loop):
                # Perform the test step and accumulate the loss
                _ = self._validation_step(
                    batch,
                    self.test_metrics,
                    self.loss_fn,
                )

        # Compute and log test metrics
        for metric_name, metric in self.test_metrics.items():
            try:
                metric_value = metric.compute()
            except ZeroDivisionError:
                metric_value = 0.0  # Handle edge case
            tb_logger.add_scalar(f"Test/{metric_name}", metric_value)  # Log metrics
