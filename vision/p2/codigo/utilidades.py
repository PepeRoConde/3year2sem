import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import cv2

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
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss, (bce_loss, dice_loss)


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

def plot_losses_and_dice(train_losses, val_losses, dice_scores, dp, bce, nome, mostra):
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
    if isinstance(dp, torch.Tensor):
        dp = dp.cpu().numpy()
    if isinstance(bce, torch.Tensor):
        bce = bce.cpu().numpy()

    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot losses
    ax1.plot(epochs, train_losses, label='Train Loss', color='blue')
    ax1.plot(epochs, val_losses, label='Validation Loss', color='orange')
    ax1.plot(epochs, dp, label='dice perdida', color='red')
    ax1.plot(epochs, bce, label='bce', color='cyan')
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
    if mostra:
        plt.show()
    plt.savefig('../figuras/'+nome+'.jpg')

def aplica_opening(mask_np, kernel_size=3):
    """
    Applies morphological opening to a binary mask using a square kernel.

    Args:
        mask_np (np.ndarray): Binary mask (0 or 1).
        kernel_size (int): Size of the structuring element.

    Returns:
        np.ndarray: Processed mask after opening.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask_np.astype(np.uint8), cv2.MORPH_OPEN, kernel)


def plot_images_and_predictions(model, dataloader, dispositivo, nome, n_rows=3 ,mostra=False):
    """
    Plot original images, ground truth, predictions, morphological opening of predictions,
    and comparison overlay between GT and opened prediction.
    """
    model.eval()
    fig, axs = plt.subplots(n_rows, 5, figsize=(25, 5 * n_rows))

    for i in range(n_rows):
        idx = np.random.randint(0, len(dataloader.dataset) - 1)
        image, gt = dataloader.dataset[idx]

        image = image.to(dispositivo)
        gt = gt.to(dispositivo)

        with torch.no_grad():
            pred = model(image.unsqueeze(0))

        # Visualization prep
        if image.shape[0] > 1:
            image_vis = image[0].cpu().numpy()
        else:
            image_vis = image.squeeze().cpu().numpy()

        gt_np = gt.squeeze().cpu().numpy()
        pred_np = torch.sigmoid(pred).squeeze().cpu().numpy()
        pred_bin = (pred_np > 0.5).astype(np.uint8)

        # Apply opening
        pred_opened = aplica_opening(pred_bin)

        # Overlay: Red = GT, Green = Opening, Yellow = overlap
        overlay = np.stack([gt_np, pred_opened, np.zeros_like(gt_np)], axis=-1)
        overlay = np.clip(overlay, 0, 1)

        # Plot original
        axs[i, 0].imshow(image_vis, cmap='gray')
        axs[i, 0].set_title(f"Original {i + 1}")
        axs[i, 0].axis('off')

        # Plot GT
        axs[i, 1].imshow(gt_np, cmap='gray')
        axs[i, 1].set_title(f"GT {i + 1}")
        axs[i, 1].axis('off')

        # Plot prediction
        axs[i, 2].imshow(pred_np, cmap='gray')
        axs[i, 2].set_title(f"Prediction {i + 1}")
        axs[i, 2].axis('off')

        # Plot opened prediction
        axs[i, 3].imshow(pred_opened, cmap='gray')
        axs[i, 3].set_title(f"Opened {i + 1}")
        axs[i, 3].axis('off')

        # Overlay comparison
        axs[i, 4].imshow(image_vis, cmap='gray')
        axs[i, 4].imshow(overlay, alpha=0.5)
        axs[i, 4].set_title(f"Overlay GT vs Opened {i + 1}")
        axs[i, 4].axis('off')

    plt.tight_layout()
    if mostra:
        plt.show()
    plt.savefig('../figuras/'+nome+'.jpg')

def plot_tensor_channels(tensor, titles=None):
    """
    This function receives a tensor and plots its channels side by side using the 'plasma' colormap.
    Optionally, it can receive a list of titles for each channel.

    Args:
        tensor (torch.Tensor): Input tensor of shape (C, H, W) or (N, C, H, W) where C is the number of channels.
        titles (list[str], optional): List of titles for each channel. If not provided, defaults to generic channel names.
    """
    if tensor.ndimension() == 4:
        tensor = tensor[0]  # Take the first batch item (assuming you want to plot one sample)

    assert tensor.ndimension() == 3, "Tensor must have 3 dimensions (C, H, W)"
    
    C, H, W = tensor.shape
    fig, axes = plt.subplots(1, C, figsize=(15, 5))

    if titles is None:
        titles = [f"Channel {i + 1}" for i in range(C)]

    for i in range(C):
        ax = axes[i]
        channel_data = tensor[i].cpu().numpy()  # Convert tensor to numpy for plotting
        im = ax.imshow(channel_data, cmap='plasma')
        ax.set_title(titles[i] if i < len(titles) else f"Channel {i + 1}")
        ax.axis('off')

    
    plt.tight_layout()
    plt.show()



# https://gist.github.com/fepegar/1fb865494cb44ac043c3189ec415d411
def redondea_a_tamano_valido(tamano, profundidade):
    """
    Redondea unha tupla de tamaño ao valor máis próximo que sexa divisible por 2 ** profundidade.

    Args:
        tamano (tuple): (alto, ancho)
        profundidade (int): profundidade do modelo UNet

    Returns:
        tuple: novo_tamano validado
    """
    multiplo = 2 ** profundidade
    alto, ancho = tamano
    novo_alto = round(alto / multiplo) * multiplo
    novo_ancho = round(ancho / multiplo) * multiplo
    return (novo_alto, novo_ancho)
