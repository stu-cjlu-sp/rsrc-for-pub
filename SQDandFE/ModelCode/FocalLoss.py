import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """
        alpha: A list or tensor of weights for each class.
        gamma: Focusing parameter.
        reduction: Specifies the reduction to apply to the output.
        """
        super(FocalLoss, self).__init__()
        if alpha is not None:
            if isinstance(alpha, (list, torch.Tensor)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                raise TypeError("alpha must be a list or a tensor.")
        else:
            self.alpha = torch.tensor([1.0] * len(alpha), dtype=torch.float32)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):

        # Compute the cross entropy loss
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Get the probabilities of the true classes
        pt = torch.exp(-BCE_loss)

        # Compute the focal loss
        alpha_t = self.alpha[targets]
        F_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class AsymmetricLoss(nn.Module):
    def __init__(self, alpha=None, beta=None, reduction='mean'):
        """
        alpha: A list or tensor of weights for positive samples of each class.
        beta: A list or tensor of weights for negative samples of each class.
        reduction: Specifies the reduction to apply to the output.
        """
        super(AsymmetricLoss, self).__init__()
        if alpha is not None:
            if isinstance(alpha, (list, torch.Tensor)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                raise TypeError("alpha must be a list or a tensor.")
        else:
            raise ValueError("alpha must be provided as a list or tensor.")

        if beta is not None:
            if isinstance(beta, (list, torch.Tensor)):
                self.beta = torch.tensor(beta, dtype=torch.float32)
            else:
                raise TypeError("beta must be a list or a tensor.")
        else:
            raise ValueError("beta must be provided as a list or tensor.")

        self.reduction = reduction

    def forward(self, inputs, targets):
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).to(dtype=torch.float32)
        
        # Compute the binary cross entropy loss
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets_one_hot, reduction='none')
        
        # Compute positive and negative weights
        positive_loss = BCE_loss * targets_one_hot * self.alpha[targets].unsqueeze(1)
        negative_loss = BCE_loss * (1 - targets_one_hot) * self.beta[targets].unsqueeze(1)
        
        # Sum the positive and negative loss
        loss = positive_loss + negative_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

