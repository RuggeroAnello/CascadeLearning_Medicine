import torch
import torch.nn as nn
import torch.nn.functional as F


class MultilabelFocalLoss(nn.Module):
    """
    Multilabel Focal Loss for multilabel classification tasks. The focal loss can use in-class and inter-class weights.
    """

    def __init__(
        self,
        gamma=2,
        alpha=None,
        reduction="mean",
        num_classes=5,
        pos_weight=None,
        class_weights=None,
    ):
        """
        Initializes the Multilabel Focal Loss.

        Args:
            gamma (int): Focusing parameter for modulating factor (default: 2).
            alpha (float): Weighting factor for the rare class (default: None).
            reduction (str): Reduction method (default: "mean").
            num_classes (int): Number of classes (default: 5).
            pos_weight (torch.Tensor): Positive weight for each class (default: None).
        """
        if reduction not in ["mean", "sum"]:
            print("Invalid reduction method specified. Defaulting to 'mean'.")
            reduction = "mean"
        super(MultilabelFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.num_classes = num_classes
        self.pos_weight = pos_weight
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        """
        Computes the Multilabel Focal Loss.

        Args:
            inputs (torch.Tensor): Raw model predictions (logits).
            targets (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Multilabel Focal Loss.
        """
        p = torch.sigmoid(inputs)

        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha and self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.pos_weight is not None:
            loss = loss * self.class_weights

        # Check reduction option and return loss accordingly
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss
