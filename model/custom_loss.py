import torch
import torch.nn as nn


# Weighted MSE Loss Function
class WeightedMSELoss(nn.Module):
    def __init__(self, weight_type='linear', alpha=0.1):
        super(WeightedMSELoss, self).__init__()
        self.weight_type = weight_type
        self.alpha = alpha

    def forward(self, predictions, targets):
        timesteps = predictions.size(1)
        if self.weight_type == 'linear':
            weights = torch.linspace(1, 1 + self.alpha * (timesteps - 1), timesteps).to(predictions.device)
        elif self.weight_type == 'exponential':
            weights = torch.exp(self.alpha * torch.arange(timesteps).float()).to(predictions.device)
        else:
            raise ValueError("Invalid weight_type. Choose 'linear' or 'exponential'.")
        weights = weights.unsqueeze(0).unsqueeze(-1)
        mse_loss = (predictions - targets) ** 2
        weighted_loss = mse_loss * weights
        return weighted_loss.mean()