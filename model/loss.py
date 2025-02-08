import torch
import torch.nn as nn
import torch.nn.functional as F


class MultilabelFocalLoss(nn.Module):
    def __init__(
        self, gamma=2, alpha=None, reduction="mean", num_classes=5, pos_weight=None
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
        super(MultilabelFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.num_classes = num_classes
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        """
        Computes the Multilabel Focal Loss.

        Args:
            inputs (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Multilabel Focal Loss.
        """
        # Compute the logit
        logit = F.sigmoid(inputs)

        # Compute the log loss
        log_loss = F.binary_cross_entropy_with_logits(
            logit, targets, reduction="none", pos_weight=self.pos_weight
        )

        # Compute the focal weight
        p_t = logit * targets + (1 - logit) * (1 - targets)
        focal_weight = torch.pow(1 - p_t, self.gamma)

        # Apply alpha if provided
        if self.alpha:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            log_loss = alpha_t * log_loss

        # Apply focal weight
        loss = focal_weight * log_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
