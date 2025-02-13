import pandas as pd
import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import wandb
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


class AbstractMultiStageModel(torch.nn.Module):
    """
    Abstract class for multi-stage models. In this class the test method is implemented.
    All multi-stage models inherit from this class and implement the forward, name and _set_model_to_eval methods.
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
            targets (dict): Dictionary containing the target labels.
        """
        super().__init__()
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = targets.keys()

        # Set hyperparameters
        self.params = params
        self._configure_hyperparameters(params)
        self._configure_metrics(params)

    def _configure_hyperparameters(self, params: dict) -> None:
        """
        Configure the hyperparameters of the model.

        Args:
            params (dict): Dictionary containing the hyperparameters.
        """
        self.lr = params.get("lr", 1e-3)
        self.batch_size = params.get("batch_size", 32)
        self.num_epochs = params.get("num_epochs", 10)
        self.optimizer_name = params.get("optimizer", "adam")
        self.label_smoothing = params.get("label_smoothing", 0.2)
        # Map string loss function names to actual loss function classes
        loss_fn_str = params.get("loss_fn", "BCEWithLogitsLoss")
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

    def _configure_metrics(self, params: dict) -> None:
        """
        Configure the metrics to be used for validation and testing.

        Args:
            params (dict): Dictionary containing the hyperparameters.
        """
        # Create dictionaries to store the metrics
        self.val_metrics = {}
        self.test_metrics = {}

        self.val_metrics_multilabel = {}
        self.test_metrics_multilabel = {}

        metrics = params.get("metrics", [])

        threshold = params.get("confidence_threshold", 0.5)

        # Initialie all metrics that are specified in the hyperparameters
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

    def _validation_step(
        self,
        batch: tuple,
        metrics: dict,
        multilabel_metrics: dict,
        loss_fn: torch.nn.Module,
        test_labels: list = None,
    ) -> torch.Tensor:
        """
        Perform a single validation step om tje given batch.

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
        outputs = torch.sigmoid(outputs).squeeze()

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
                    metric.update(outputs_metric, labels_metric)

        # Update multilabel metrics
        for name, metric in multilabel_metrics.items():
            if name == "mcc":
                metric.update(outputs, labels.to(torch.int64))
            else:
                metric.update(outputs, labels)

        return loss

    def _set_model_to_eval():
        raise NotImplementedError

    def test(
        self,
        test_dataset,
        name,
        tb_logger=None,
        log_wandb=False,
        test_labels: list = None,
    ) -> pd.DataFrame:
        """
        Test the model on the given dataset.

        Args:
            test_dataset (torch.utils.data.Dataset): Dataset to test the model on.
            name (str): Name of the model.
            tb_logger (SummaryWriter, optional): Tensorboard logger to log the test results. Defaults to None.
            log_wandb (bool, optional): Whether to log the test results with wandb. Defaults to True.
            test_labels (list, optional): Idx of labels to test. The idx is the relative position of the labels in the output vector. The test labels have to be the of the same size as num_labels. Defaults to None.

        Returns:
            pd.DataFrame: Dataframe containing the test results.
        """
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        # Use the validation loss function if specified
        loss_fn = self.loss_fn_val if self.use_loss_fn_val else self.loss_fn

        self._set_model_to_eval()

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

        # Create a dataframe to store the results
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
                        # Save the auc in the result
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
                    wandb.log({f"Test/{label}_{metric_name}": metric_value})

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

        return res


class TwoStageModelAPPA(AbstractMultiStageModel):
    """
    Two-stage model. The first stage classifies the images into AP and PA views. The second stage processes the images based on the classification.
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

        # Save additional hyperparameters
        self.confidence_threshold_first_ap_pa = params.get(
            "confidence_threshold_first_ap_pa", 0.5
        )

        # Define the three models
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

    def _set_model_to_eval(self):
        """
        Set all models to evaluation mode.
        """
        self.model_ap_pa_classification.eval()
        self.model_ap.eval()
        self.model_pa.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the two-stage model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        # First stage: AP-PA classification
        out_stage_one = self.model_ap_pa_classification(x)
        conf = torch.sigmoid(out_stage_one)
        pred = conf < self.confidence_threshold_first_ap_pa

        # Get the indices of the AP and PA images
        idx_ap = pred.squeeze().nonzero(as_tuple=True)[0]
        idx_pa = (~pred.squeeze()).nonzero(as_tuple=True)[0]

        # Create the AP and PA tensors
        x_ap = x[idx_ap]
        x_pa = x[idx_pa]

        # Second stage: classification
        out_ap = self.model_ap(x_ap) if x_ap.size(0) > 0 else None
        out_pa = self.model_pa(x_pa) if x_pa.size(0) > 0 else None

        # Combine the outputs
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


class TwoStageModelFrontalLateral(AbstractMultiStageModel):
    """
    Two-stage model. The first stage classifies the images into frontal and lateral views. The second stage processes the images based on the classification.
    """

    def __init__(
        self,
        params: dict,
        targets: dict,
        model_frontal_lateral_classification: str,
        model_frontal: str,
        model_lateral: str,
    ):
        """
        Initialize the two-stage model with the given hyperparameters.

        Args:
            params (dict): Dictionary containing the hyperparameters.
            targets (dict): Dictionary containing the target labels.
            model_frontal_lateral_classification (str): Path to the model for frontal-lateral classification.
            model_frontal (str): Path to the model for frontal images.
            model_lateral (str): Path to the model for lateral images.
        """
        super().__init__(
            params=params,
            targets=targets,
        )

        # Save additional hyperparameters
        self.confidence_threshold_first_frontal_lateral = params.get(
            "confidence_threshold_first_fronal_lateral", 0.5
        )

        # Define the three models
        self.model_frontal_lateral_classification = torch.load(
            model_frontal_lateral_classification, weights_only=False
        )
        self.model_frontal = torch.load(model_frontal, weights_only=False)
        self.model_lateral = torch.load(model_lateral, weights_only=False)

        # Move models to device
        self.model_frontal_lateral_classification.to(self.device)
        self.model_frontal.to(self.device)
        self.model_lateral.to(self.device)

        # Save the used models
        self.params["model_frontal_lateral_classification"] = (
            model_frontal_lateral_classification
        )
        self.params["model_frontal"] = model_frontal
        self.params["model_lateral"] = model_lateral

    def forward(self, x):
        """
        Forward pass of the two-stage model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # First stage: frontal-lateral classification
        out_stage_one = self.model_frontal_lateral_classification(x)
        conf = torch.sigmoid(out_stage_one)
        pred = conf < self.confidence_threshold_first_frontal_lateral

        # Get the indices of the frontal and lateral images
        idx_frontal = pred.squeeze().nonzero(as_tuple=True)[0]
        idx_lateral = (~pred.squeeze()).nonzero(as_tuple=True)[0]

        # Create the frontal and lateral tensors
        x_frontal = x[idx_frontal]
        x_lateral = x[idx_lateral]

        # Second stage: classification
        out_frontal = self.model_frontal(x_frontal) if x_frontal.size(0) > 0 else None
        out_lateral = self.model_lateral(x_lateral) if x_lateral.size(0) > 0 else None

        # Combine the outputs
        output = torch.zeros(
            (
                x.size(0),
                out_frontal.size(1) if out_frontal is not None else out_lateral.size(1),
            )
        ).to(self.device)
        if out_frontal is not None:
            output[idx_frontal] = out_frontal
        if out_lateral is not None:
            output[idx_lateral] = out_lateral

        return output

    def _set_model_to_eval(self):
        """
        Set all models to evaluation mode.
        """
        self.model_frontal_lateral_classification.eval()
        self.model_frontal.eval()
        self.model_lateral.eval()

    @property
    def name(self):
        return "TwoStageModel_frontal-lateral"


class ThreeStageModelFrontalLateralAPPA(AbstractMultiStageModel):
    """
    Two-stage model. The first stage classifies the images into Frontal/Lateral. The second stage classifies the Frontal images into AP/PA. The third stage processes the images based on the classification.
    """

    def __init__(
        self,
        params: dict,
        targets: dict,
        model_frontal_lateral_classification: str,
        model_frontal_ap_pa_classification: str,
        model_frontal_ap: str,
        model_frontal_pa: str,
        model_lateral: str,
    ):
        """
        Initialize the two-stage model with the given hyperparameters.

        Args:
            params (dict): Dictionary containing the hyperparameters.
            targets (dict): Dictionary containing the target labels.
            model_frontal_lateral_classification (str): Path to the model for frontal-lateral classification.
            model_frontal_ap_pa_classification (str): Path to the model for frontal AP-PA classification.
            model_frontal_ap (str): Path to the model for frontal AP images.
            model_frontal_pa (str): Path to the model for frontal PA images.
            model_lateral (str): Path to the model for lateral images.
        """
        super().__init__(
            params=params,
            targets=targets,
        )

        # Save additional hyperparameters
        self.confidence_threshold_frontal_lateral = params.get(
            "confidence_threshold_frontal_lateral", 0.5
        )
        self.confidence_threshold_frontal_ap_pa = params.get(
            "confidence_threshold_frontal_ap_pa", 0.5
        )

        # Define the five models
        self.model_frontal_lateral_classification = torch.load(
            model_frontal_lateral_classification, weights_only=False
        )
        self.model_frontal_ap_pa_classification = torch.load(
            model_frontal_ap_pa_classification, weights_only=False
        )
        self.model_frontal_ap = torch.load(model_frontal_ap, weights_only=False)
        self.model_frontal_pa = torch.load(model_frontal_pa, weights_only=False)
        self.model_lateral = torch.load(model_lateral, weights_only=False)

        # Move models to device
        self.model_frontal_lateral_classification.to(self.device)
        self.model_frontal_ap_pa_classification.to(self.device)
        self.model_frontal_ap.to(self.device)
        self.model_frontal_pa.to(self.device)
        self.model_lateral.to(self.device)

        # Save the used models
        self.params["model_frontal_lateral_classification"] = (
            model_frontal_lateral_classification
        )
        self.params["model_frontal_ap_pa_classification"] = (
            model_frontal_ap_pa_classification
        )
        self.params["model_frontal_ap"] = model_frontal_ap
        self.params["model_frontal_pa"] = model_frontal_pa
        self.params["model_lateral"] = model_lateral

    def _set_model_to_eval(self):
        """
        Set all models to evaluation mode.
        """
        self.model_frontal_lateral_classification.eval()
        self.model_frontal_ap_pa_classification.eval()
        self.model_frontal_ap.eval()
        self.model_frontal_pa.eval()
        self.model_lateral.eval()

    def forward(self, x):
        """
        Forward pass of the two-stage model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        # First stage: frontal-lateral classification
        out_frontal_lateral = self.model_frontal_lateral_classification(x)
        conf = torch.sigmoid(out_frontal_lateral)
        pred = conf < self.confidence_threshold_frontal_lateral

        # Get the indices of the frontal and lateral images
        idx_frontal = pred.squeeze().nonzero(as_tuple=True)[0]
        idx_lateral = (~pred.squeeze()).nonzero(as_tuple=True)[0]

        # Create the frontal and lateral tensors
        x_frontal = x[idx_frontal]
        x_lateral = x[idx_lateral]

        # Second stage: frontal AP-PA classification for frontal images
        out_frontal_ap_pa = self.model_frontal_ap_pa_classification(x_frontal)
        conf = torch.sigmoid(out_frontal_ap_pa)
        pred = conf < self.confidence_threshold_frontal_ap_pa

        # Get the indices of the AP and PA images
        idx_ap = pred.squeeze().nonzero(as_tuple=True)[0]
        idx_pa = (~pred.squeeze()).nonzero(as_tuple=True)[0]

        # Create the AP and PA tensors
        x_ap = x[idx_frontal[idx_ap]]
        x_pa = x[idx_frontal[idx_pa]]

        # Final stage: classification
        out_ap = self.model_frontal_ap(x_ap) if x_ap.size(0) > 0 else None
        out_pa = self.model_frontal_pa(x_pa) if x_pa.size(0) > 0 else None
        out_lateral = self.model_lateral(x_lateral) if x_lateral.size(0) > 0 else None

        # Create the output tensor
        output = torch.zeros(
            x.size(0),
            out_lateral.size(1)
            if out_lateral is not None
            else out_ap.size(1)
            if out_ap is not None
            else out_pa.size(1),
        ).to(self.device)

        # Combine the outputs
        if out_ap is not None:
            output[idx_frontal[idx_ap]] = out_ap
        if out_pa is not None:
            output[idx_frontal[idx_pa]] = out_pa
        if out_lateral is not None:
            output[idx_lateral] = out_lateral

        return output

    @property
    def name(self):
        return "TwoStageModel_frontal_lateral_AP-PA"
