from torchvision import transforms
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import torch
import numpy as np
import cv2
from skimage.filters import frangi

class Transformacions:
    def __init__(self,
                 novo_tamano=(416, 624),
                 aumento_datos=False,
                 anade_canny=False,
                 anade_sobel=False,
                 anade_laplacian=False,
                 anade_frangi=False,
                 volteo_horizotal=False,
                 rotacion_aleatoria=True,
                 escalado_aleatorio=True,
                 anade_ruido_gaussiano=True,
                 jitter_color=True,
                 variar_enfoque=True,
                 deformacion_elastica=True):
        
        self.novo_tamano = novo_tamano
        self.aumento_datos = aumento_datos

        self.anade_canny = anade_canny
        self.anade_sobel = anade_sobel
        self.anade_laplacian = anade_laplacian
        self.anade_frangi = anade_frangi

        self.volteo_horizotal = volteo_horizotal
        self.rotacion_aleatoria = rotacion_aleatoria
        self.escalado_aleatorio = escalado_aleatorio
        self.anade_ruido_gaussiano = anade_ruido_gaussiano
        self.jitter_color = jitter_color
        self.variar_enfoque = variar_enfoque
        self.deformacion_elastica = deformacion_elastica

    def __call__(self, imaxe, mascara):

        if not isinstance(imaxe, Image.Image):
            imaxe = transforms.ToPILImage()(imaxe)
        if not isinstance(mascara, Image.Image):
            mascara = transforms.ToPILImage()(mascara)

        imaxe = transforms.Resize(self.novo_tamano, interpolation=InterpolationMode.BILINEAR)(imaxe)
        mascara = transforms.Resize(self.novo_tamano, interpolation=InterpolationMode.NEAREST)(mascara)

        # ----------------
        # Aumento de datos
        # ----------------

        if self.aumento_datos: # en numpy

            if self.volteo_horizotal: imaxe, mascara = self.volteo_horizotal_fn(imaxe, mascara)
            if self.rotacion_aleatoria: imaxe, mascara = self.rotacion_aleatoria_fn(imaxe, mascara)
            if self.escalado_aleatorio: imaxe, mascara = self.escalado_aleatorio_fn(imaxe, mascara)
            if self.jitter_color: imaxe = self.jitter_color_fn(imaxe)
            if self.variar_enfoque: imaxe = self.variar_enfoque_fn(imaxe)

        imaxe = transforms.ToTensor()(imaxe)
        mascara = transforms.ToTensor()(mascara)

        if self.aumento_datos: # en torch
            if self.anade_ruido_gaussiano: imaxe = self.anade_ruido_gaussiano_fn(imaxe)
            if self.deformacion_elastica: imaxe, mascara = self.deformacion_elastica_fn(imaxe, mascara)

        # -----------------
        # Canles adicionais
        # -----------------

        if self.anade_canny or self.anade_sobel or self.anade_laplacian:
            canles_adicionais = []

            imaxe_np = imaxe.numpy()
            gris = imaxe_np.mean(axis=0) * 255  # [H, W] a [0,255]
            gris = gris.astype(np.uint8)

            if self.anade_canny: canles_adicionais.append(self.canny(gris))
            if self.anade_sobel: canles_adicionais.append(self.sobel(gris))
            if self.anade_laplacian: canles_adicionais.append(self.laplacian(gris))
            if self.anade_frangi: canles_adicionais.append(self.frangi_fn(gris))

            imaxe = torch.cat([imaxe] + canles_adicionais, dim=0)

        return imaxe, mascara

    # Funcions de Aumento de Datos

    def volteo_horizotal_fn(self, imaxe, mascara):
        if torch.rand(1) > 0.5:
            imaxe = transforms.functional.hflip(imaxe)
            mascara = transforms.functional.hflip(mascara)
        return imaxe, mascara

    def rotacion_aleatoria_fn(self, imaxe, mascara):
        angle = torch.empty(1).uniform_(-45, 45).item()
        imaxe = transforms.functional.rotate(imaxe, angle, interpolation=InterpolationMode.BILINEAR)
        mascara = transforms.functional.rotate(mascara, angle, interpolation=InterpolationMode.NEAREST)
        return imaxe, mascara

    def escalado_aleatorio_fn(self, imaxe, mascara):
        scale = torch.empty(1).uniform_(0.8, 1.6).item()
        h, w = imaxe.size[1], imaxe.size[0]
        new_size = (int(h * scale), int(w * scale))
        imaxe = transforms.Resize(new_size, interpolation=InterpolationMode.BILINEAR)(imaxe)
        mascara = transforms.Resize(new_size, interpolation=InterpolationMode.NEAREST)(mascara)
        imaxe = transforms.CenterCrop(self.novo_tamano)(imaxe)
        mascara = transforms.CenterCrop(self.novo_tamano)(mascara)
        return imaxe, mascara

    def jitter_color_fn(self, imaxe):
        jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)
        return jitter(imaxe)

    def variar_enfoque_fn(self, imaxe):
        if torch.rand(1) > 0.5:
            return transforms.functional.adjust_sharpness(imaxe, 2)
        else:
            return transforms.functional.gaussian_blur(imaxe, kernel_size=3)

    def anade_ruido_gaussiano_fn(self, imaxe):
        noise = torch.randn_like(imaxe) * 0.05
        return torch.clamp(imaxe + noise, 0.0, 1.0)

    def deformacion_elastica_fn(self, imaxe, mascara, alpha=60, sigma=10):
        # Simple elastic deformation
        c, h, w = imaxe.shape
        dx = torch.from_numpy(cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha).float()
        dy = torch.from_numpy(cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha).float()

        x, y = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        x = x.float() + dx
        y = y.float() + dy

        x = torch.clamp(x, 0, h - 1)
        y = torch.clamp(y, 0, w - 1)

        grid = torch.stack((y / (w - 1) * 2 - 1, x / (h - 1) * 2 - 1), dim=-1).unsqueeze(0)

        imaxe = F.grid_sample(imaxe.unsqueeze(0), grid, mode='bilinear', padding_mode='border', align_corners=True).squeeze(0)
        mascara = F.grid_sample(mascara.unsqueeze(0), grid, mode='nearest', padding_mode='border', align_corners=True).squeeze(0)

        return imaxe, mascara

    # Funcions de Aumento de Canles

    def canny(self, gris):
        canny = cv2.Canny(gris, 100, 200)
        return torch.tensor(canny / 255.0, dtype=torch.float32).unsqueeze(0)
        
    def sobel(self, gris):
        sobelx = cv2.Sobel(gris, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gris, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.hypot(sobelx, sobely)
        sobel = (sobel / sobel.max() if sobel.max() > 0 else sobel).astype(np.float32)
        return torch.tensor(sobel).unsqueeze(0)

    def laplacian(self, gris):
        laplacian = cv2.Laplacian(gris, cv2.CV_64F)
        laplacian = np.abs(laplacian)
        laplacian = (laplacian / laplacian.max() if laplacian.max() > 0 else laplacian).astype(np.float32)
        return torch.tensor(laplacian).unsqueeze(0)

    def frangi_fn(self, gris):
        gris_norm = gris.astype(np.float32) / 255.0
        frangi_imx = frangi(gris_norm)
        frangi_imx = (frangi_imx / frangi_imx.max() if frangi_imx.max() > 0 else frangi_img).astype(np.float32)
        return torch.tensor(frangi_imx).unsqueeze(0)



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
    
    if atallo.size(2) == x.size(2) and atallo.size(3) == x.size(3):
        return x  # Already aligned

    print('.',sep='')

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
