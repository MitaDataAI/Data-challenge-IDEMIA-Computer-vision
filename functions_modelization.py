import torch
import torch.nn as nn

# loss function
class WeightedMSELoss(nn.Module):
    """
    Weighted Mean Squared Error Loss:
    Applies a custom weight to each element of the squared error
    before computing the mean. This allows certain samples to 
    have more or less influence on the total loss.
    """
    def __init__(self, weights):
        super(WeightedMSELoss, self,).__init__()
        self.weights = weights
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, input, target, weights):
        # Compute the element-wise squared error
        squared_error = (input - target) ** 2
        # Apply weights to the squared errors
        weighted_squared_error = weights * squared_error
        # Compute the mean of the weighted squared errors
        loss = torch.mean(weighted_squared_error)
        return loss

class WeightedMSELoss2(nn.Module):
    """
    Context-Aware Weighted MSE Loss:
    Computes MSE and applies dynamic weights based on two factors:
    - 'ranges' (e.g., age groups or categories)
    - 'genders' (e.g., male or female)
    
    The weights are retrieved from user-provided dictionaries and 
    combined multiplicatively to emphasize/de-emphasize specific groups.
    """
    def __init__(self, range_weights, gender_weights):
        super(WeightedMSELoss2, self).__init__()
        self.range_weights = range_weights
        self.gender_weights = gender_weights
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, inputs, targets, ranges, genders):
        loss = self.mse(inputs, targets)
        range_weights = torch.tensor([self.range_weights[str(label)] for label in ranges], device=inputs.device)
        gender_weights = torch.tensor([self.gender_weights['male'] if gender > 0.5 else self.gender_weights['female'] for gender in genders], device=inputs.device)
        total_weights = range_weights * gender_weights
        weighted_loss = loss * total_weights
        return weighted_loss.mean()