import torch
import torch.nn.functional as F


def focal_loss(
    logits: torch.Tensor, targets: torch.Tensor, gamma: float, weight: torch.Tensor
):
    """Compute the focal loss.

    If gamma == 0, this is equivalent to the cross entropy loss, with no correction.
    if gamma != 0, the system will be penalized for overly relying on increasing the confidence in prediction,
    and will be prompted to resolve the most difficult cases.

    Input shapes are as follows (where B = batch size, C = number of classes):
    logits: (B, C)
    targets: (B, )
    weight: (C, )
    gamma: float (scalar)

    It is returned the weighted average of the loss

    """

    # Get the probabilities, shape (B, C)
    probabilities = F.softmax(logits, dim=1)

    # Get the log probabilities, shape (B, C)
    log_probabilities = F.log_softmax(logits, dim=1)

    # Correction factor due to focal loss
    correction = (1 - probabilities) ** gamma

    if weight is None:
        # Use default value for weight, where each class has the same weight (equal to 1)
        weight = torch.ones(logits.shape[1])

    # Corrected log probabilities
    weighted_corrected_log_probabilities = weight * correction * log_probabilities

    # Sum of weights to be used in weighted average
    weight_sum = torch.gather(
        weight.repeat(logits.shape[0], 1), 1, targets.view(-1, 1)
    ).sum()

    # Compute the loss function
    loss = -torch.gather(weighted_corrected_log_probabilities, 1, targets.view(-1, 1))

    # Return the weighted average
    return loss.sum() / weight_sum


class FocalLoss(torch.nn.Module):
    """Class implementation of the focal loss.

    Higher gamma means higher penalization for overly confident predictions.
    Assign weights to each class to account for class imbalance. Default is some weight to each class"""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        # Use constructor of parent class
        super().__init__()

        # Check that gamma is a scalar
        if isinstance(gamma, (int, float)):
            # Store gamma
            self.gamma = gamma
        else:
            raise ValueError("Gamma must be a scalar, either int or float")

        # Check that weight is a tensor
        if weight is not None:
            if isinstance(weight, torch.Tensor):
                # Store weight
                self.weight = weight
            else:
                raise ValueError("Weight must be a Torch tensor")
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """Apply the focal loss"""

        # Compute the loss
        return focal_loss(
            logits=logits, targets=targets, gamma=self.gamma, weight=self.weight
        )
