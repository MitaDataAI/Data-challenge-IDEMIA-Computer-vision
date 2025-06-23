# Core libraries
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Torchvision
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import mobilenet_v3_small

# Evaluation
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


# loss function
import torch
import torch.nn as nn

#############################################
# FUNCTIONS
#############################################

class WeightedMSELoss(nn.Module):
    """
    Weighted Mean Squared Error Loss.

    This loss function applies a weight to each individual squared error
    based on a categorical range (e.g., occlusion level). The weights are
    defined in advance via a dictionary and are used to emphasize or
    de-emphasize the contribution of each sample to the total loss.

    Args:
        weights (dict): A dictionary mapping string labels (range categories)
                        to numerical weight values. Example:
                        {'0-0.025': 1.0, '0.025-0.05': 1.5, ...}

    Forward Args:
        inputs (Tensor): Predicted values (batch_size, 1)
        targets (Tensor): Ground truth values (batch_size, 1)
        ranges (list or array-like): Categorical labels (as strings or pandas.Categorical)
                                     that map each target to a weight via `self.weights`.

    Returns:
        Tensor: Scalar tensor representing the mean weighted MSE loss.
    """
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights
        self.mse = nn.MSELoss(reduction='none')  # Keep individual losses per sample

    def forward(self, inputs, targets, ranges):
        # Compute unweighted element-wise squared error
        loss = self.mse(inputs, targets)

        # Map each range label to its corresponding weight
        weights = torch.tensor(
            [self.weights[str(label)] for label in ranges],
            device=inputs.device
        )

        # Apply the weights to the losses
        weighted_loss = loss * weights

        # Return the mean weighted loss over the batch
        return weighted_loss.mean()

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


def error_fn(df):
    """
    Calcule l'erreur quadratique moyenne pondérée pour un DataFrame contenant des prédictions.

    Les poids sont définis comme 1/30 + valeur cible, ce qui donne plus d'importance
    aux observations ayant une cible plus élevée.

    Args:
        df (pd.DataFrame): Un DataFrame avec au moins deux colonnes :
            - "pred" : les prédictions du modèle
            - "target" : les valeurs réelles

    Returns:
        float: Erreur quadratique moyenne pondérée
    """
    # Extraire les prédictions et les vraies valeurs
    pred = df.loc[:, "pred"]
    ground_truth = df.loc[:, "target"]

    # Définir les poids : plus la cible est grande, plus l'observation est pondérée
    weight = 1 / 30 + ground_truth

    # Calcul de l'erreur quadratique moyenne pondérée
    return np.sum(((pred - ground_truth) ** 2) * weight, axis=0) / np.sum(weight, axis=0)

def metric_fn(female, male):
    """
    Calcule une métrique combinée d'erreur pour les sous-groupes "female" et "male".

    La métrique combine :
      - la moyenne des erreurs des deux groupes
      - un terme d'équité basé sur leur différence absolue

    Args:
        female (pd.DataFrame): Données pour le groupe "female", même format que `error_fn`.
        male (pd.DataFrame): Données pour le groupe "male", même format que `error_fn`.

    Returns:
        float: Score final intégrant performance moyenne et équité inter-groupe.
    """
    # Calcul de l'erreur pour le groupe masculin
    err_male = error_fn(male)

    # Calcul de l'erreur pour le groupe féminin
    err_female = error_fn(female)

    # Affichage des erreurs pour diagnostic
    print("Male error: ", err_male)
    print("Female error: ", err_female)

    # Retourne la moyenne des erreurs + écart absolu (pénalise l'injustice)
    return (err_male + err_female) / 2 + abs(err_male - err_female)


def plot_confusion_matrix(results, title):
    y_true = results['true_range']
    y_pred = results['pred_range']
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()
    
    print(f"Classification Report for {title}:\n")
    print(classification_report(y_true, y_pred, labels=labels, target_names=labels))
