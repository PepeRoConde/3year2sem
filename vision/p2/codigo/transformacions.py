from torchvision import transforms
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import torch
import numpy as np

class SegmentationTransform:
    """Applies the same geometric transforms to both image and mask."""
    def __init__(self, resize=(416,624), is_train=True, normalize=True):
        self.resize = resize
        self.is_train = is_train

    def __call__(self, image, mask):
        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            image = transforms.ToPILImage()(image)
        if not isinstance(mask, Image.Image):
            mask = transforms.ToPILImage()(mask)

        image = transforms.Resize(self.resize, interpolation=InterpolationMode.BILINEAR)(image)
        mask = transforms.Resize(self.resize, interpolation=InterpolationMode.NEAREST)(mask)

        # Aumento de datos 
        if self.is_train:
            if torch.rand(1) > 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)

        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)

        return image, mask


# https://gist.github.com/fepegar/1fb865494cb44ac043c3189ec415d411
def axusta_tamano(atallo, x, mode='pad'):
    """
    Aligns tensor `x` to match the spatial size of `atallo` using the specified mode.

    Args:
        atallo (Tensor): The reference tensor for size (C, H, W).
        x (Tensor): The tensor to align.
        mode (str): Either 'pad' or 'crop'.
    Returns:
        Tensor: Aligned tensor with the same H and W as `atallo`.
    """
    diffY = atallo.size(2) - x.size(2)
    diffX = atallo.size(3) - x.size(3)

    if mode == 'pad':
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
    elif mode == 'crop':
        x = x[:, :,
              diffY // 2 : x.size(2) + diffY // 2,
              diffX // 2 : x.size(3) + diffX // 2]
    else:
        raise ValueError("mode must be 'pad' or 'crop'")

    return x
