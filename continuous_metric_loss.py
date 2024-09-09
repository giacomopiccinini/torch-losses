import torch


def get_continuous_confusion_matrix(p: torch.Tensor, y: torch.Tensor):
    """Get the continuous confusion matrix
    p: (B, )
    y: (B, )
    """

    TP = torch.dot(p, y)
    FP = torch.dot(p, 1 - y)
    FN = torch.dot(1 - p, y)
    TN = torch.dot(1 - p, 1 - y)

    return TP, TN, FP, FN


class AccuracyLoss(torch.nn.Module):
    """Class implementation of the continuous Accuracy loss."""

    def __init__(self):
        # Use constructor of parent class
        super().__init__()

        # Sigmoid activation function
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """Apply the accuracy loss"""

        # Convert to probs
        probabilities = self.sigmoid(logits)

        # Reshape to (B, ) and compute the continuous confusion matrix
        TP, TN, FP, FN = get_continuous_confusion_matrix(
            p=probabilities.view(-1), y=targets.view(-1)
        )

        # Compute the loss
        return -((TP + TN) / (TP + TN + FP + FN))


class PrecisionLoss(torch.nn.Module):
    """Class implementation of the continuous Precision loss."""

    def __init__(self):
        # Use constructor of parent class
        super().__init__()

        # Sigmoid activation function
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """Apply the precision loss"""

        # Convert to probs
        probabilities = self.sigmoid(logits)

        # Reshape to (B, ) and compute the continuous confusion matrix
        TP, _TN, FP, _FN = get_continuous_confusion_matrix(
            p=probabilities.view(-1), y=targets.view(-1)
        )

        # Compute the precision loss
        return -(TP / (TP + FP))


class RecallLoss(torch.nn.Module):
    """Class implementation of the continuous Recall loss."""

    def __init__(self):
        # Use constructor of parent class
        super().__init__()

        # Sigmoid activation function
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """Apply the recall loss"""

        # Convert to probs
        probabilities = self.sigmoid(logits)

        # Reshape to (B, ) and compute the continuous confusion matrix
        TP, _TN, _FP, FN = get_continuous_confusion_matrix(
            p=probabilities.view(-1), y=targets.view(-1)
        )

        # Compute the recall loss
        return -(TP / (TP + FN))


class FBetaLoss(torch.nn.Module):
    """Class implementation of the continuous F-beta loss."""

    def __init__(self, beta: float = 1.0):
        # Use constructor of parent class
        super().__init__()

        # Sigmoid activation function
        self.sigmoid = torch.nn.Sigmoid()

        # Store beta
        self.beta = beta

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """Apply the F-beta loss"""

        # Convert to probs
        probabilities = self.sigmoid(logits)

        # Reshape to (B, ) and compute the continuous confusion matrix
        TP, _TN, FP, FN = get_continuous_confusion_matrix(
            p=probabilities.view(-1), y=targets.view(-1)
        )

        # Compute precision and recall
        P = TP / (TP + FP)
        R = TP / (TP + FN)

        # Compute the F-beta loss
        return -((1 + self.beta**2) * P * R / ((self.beta**2 * P) + R))


class MarkednessLoss(torch.nn.Module):
    """Class implementation of the continuous Markedness loss."""

    def __init__(self):
        # Use constructor of parent class
        super().__init__()

        # Sigmoid activation function
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """Apply the F-beta loss"""

        # Convert to probs
        probabilities = self.sigmoid(logits)

        # Reshape to (B, ) and compute the continuous confusion matrix
        TP, TN, FP, FN = get_continuous_confusion_matrix(
            p=probabilities.view(-1), y=targets.view(-1)
        )

        # Compute the F-beta loss
        return -(TP / (TP + FP) - FN / (TN + FN))


class InformednessLoss(torch.nn.Module):
    """Class implementation of the continuous Informedness loss."""

    def __init__(self):
        # Use constructor of parent class
        super().__init__()

        # Sigmoid activation function
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """Apply the F-beta loss"""

        # Convert to probs
        probabilities = self.sigmoid(logits)

        # Reshape to (B, ) and compute the continuous confusion matrix
        TP, TN, FP, FN = get_continuous_confusion_matrix(
            p=probabilities.view(-1), y=targets.view(-1)
        )

        # Compute the F-beta loss
        return -(TP / (TP + FN) - FP / (TN + FP))


class PhiBetaLoss(torch.nn.Module):
    """Class implementation of the continuous phi-beta loss."""

    def __init__(self, beta: float = 1.0):
        # Use constructor of parent class
        super().__init__()

        # Sigmoid activation function
        self.sigmoid = torch.nn.Sigmoid()

        # Store beta
        self.beta = beta

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """Apply the F-beta loss"""

        # Convert to probs
        probabilities = self.sigmoid(logits)

        # Reshape to (B, ) and compute the continuous confusion matrix
        TP, TN, FP, FN = get_continuous_confusion_matrix(
            p=probabilities.view(-1), y=targets.view(-1)
        )

        # Compute informedness and markedness
        i = TP / (TP + FN) - FP / (TN + FP)
        m = TP / (TP + FP) - FN / (TN + FN)

        return (1 + self.beta**2) * i * m / (self.beta**2 * m + i)
