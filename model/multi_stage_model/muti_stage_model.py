import torch
import torchvision
import os
import torch.nn.functional as F

from networks.one_model.one_stage_models import ResNet18OneStage, ResNet50OneStage


class TwoStageModel(torch.nn.Module):

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

        # Set hyperparameters
        if "lr" in params:
            self.lr = params["lr"]
        if "batch_size" in params:
            self.batch_size = params["batch_size"]
        if "num_epochs" in params:
            self.num_epochs = params["num_epochs"]
        # TODO add more hyperparameters if needed

        # Define the two models
        self.model_ap_pa_classification = ResNet18OneStage()
        self.model_ap = ResNet50OneStage()
        self.model_pa = ResNet50OneStage()

        # Move models to device
        self.model_ap_pa_classification.to(self.device)
        self.model_ap.to(self.device)
        self.model_pa.to(self.device)

    def forward(self, x):
        x = self.model_ap_pa_classification(x)
        if x == 0:
            x = self.model_ap(x)
        elif x == 1:
            x = self.model_pa(x)
