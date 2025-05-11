import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        return self.bce(logits, targets)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.type_as(probs)

        bce_loss = F.binary_cross_entropy(probs, targets, reduction='none')
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal.mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)

        tp = (probs * targets).sum()
        fp = ((1 - targets) * probs).sum()
        fn = (targets * (1 - probs)).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce = BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalDiceLoss(nn.Module):
    def __init__(self, focal_weight=0.5, dice_weight=0.5, alpha=0.8, gamma=2.0):
        super(FocalDiceLoss, self).__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss

def plot_losses_and_dice(train_losses, val_losses, dice_scores):
    """
    Plots training/validation losses and Dice metric side by side.

    Parameters:
    - train_losses: list or array of training loss values
    - val_losses: list or array of validation loss values
    - dice_scores: list or array of Dice metric values
    """

    if isinstance(train_losses, torch.Tensor):
        train_losses = train_losses.cpu().numpy()
    if isinstance(val_losses, torch.Tensor):
        val_losses = val_losses.cpu().numpy()
    if isinstance(dice_scores, torch.Tensor):
        dice_scores = dice_scores.cpu().numpy()

    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot losses
    ax1.plot(epochs, train_losses, label='Train Loss', color='blue')
    ax1.plot(epochs, val_losses, label='Validation Loss', color='orange')
    ax1.set_title('Losses')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot Dice metric
    ax2.plot(epochs, dice_scores, label='Dice Score', color='green')
    ax2.set_title('Dice Metric')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()



def plot_images_and_predictions(model, dataloader, dispositivo, n_rows=5):
    """
    Function to plot the original images, ground truth and predicted segmentation masks for a given number of random rows.

    Args:
        model (torch.nn.Module): The trained U-Net model.
        dataloader (torch.utils.data.DataLoader): The DataLoader for the dataset (test set).
        dispositivo (str): The device ('cpu', 'cuda', or 'mps') where the model is loaded.
        n_rows (int): The number of random test samples to visualize.

    """
    model.eval()  # Set model to evaluation mode
    
    # Create a figure with n_rows rows and 3 columns (original, GT, and predicted images)
    fig, axs = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    
    # Iterate through random rows from the dataset
    for i in range(n_rows):
        # Get a random sample
        idx = np.random.randint(0, len(dataloader.dataset) - 1)
        image, gt = dataloader.dataset[idx]
        
        # Move the image and ground truth to the same device as the model
        image = image.to(dispositivo)
        gt = gt.to(dispositivo)
        
        # Make a prediction
        with torch.no_grad():
            pred = model(image.unsqueeze(0))  # Add batch dimension
        
        # Convert to numpy arrays for visualization
        image_np = image.squeeze().cpu().numpy()
        gt_np = gt.squeeze().cpu().numpy()
        pred_np = torch.sigmoid(pred).squeeze().cpu().numpy()  # Apply sigmoid to get probabilities
        
        # Plot the original image
        axs[i, 0].imshow(image_np, cmap='gray')
        axs[i, 0].set_title(f"Original {i + 1}")
        axs[i, 0].axis('off')
        
        # Plot the ground truth
        axs[i, 1].imshow(gt_np, cmap='gray')
        axs[i, 1].set_title(f"GT {i + 1}")
        axs[i, 1].axis('off')
        
        # Plot the predicted segmentation
        axs[i, 2].imshow(pred_np, cmap='gray')
        axs[i, 2].set_title(f"Predicted {i + 1}")
        axs[i, 2].axis('off')
    
    # Show the plots
    plt.tight_layout()
    plt.show()
