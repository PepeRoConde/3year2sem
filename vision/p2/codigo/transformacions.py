from torchvision import transforms
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import torch
import numpy as np
import cv2
from skimage.filters import frangi
from skimage.morphology import remove_small_objects


class Transformacions:
    def __init__(self,
                 novo_tamano=(416, 624),
                 aumento_datos=False,
                 anade_canny=False,
                 anade_sobel=False,
                 anade_laplacian=False,
                 anade_frangi=False,
                 volteo_horizotal=True,
                 anade_ruido_gaussiano=True,
                 jitter_color=True,
                 variar_enfoque=True,
                 transformacion_afin=True,
                 deformacion_elastica=True):
        
        self.novo_tamano = novo_tamano
        self.aumento_datos = aumento_datos

        self.anade_canny = anade_canny
        self.anade_sobel = anade_sobel
        self.anade_laplacian = anade_laplacian
        self.anade_frangi = anade_frangi

        self.volteo_horizotal = volteo_horizotal
        self.anade_ruido_gaussiano = anade_ruido_gaussiano
        self.jitter_color = jitter_color
        self.variar_enfoque = variar_enfoque
        self.transformacion_afin = transformacion_afin
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
            if self.jitter_color: imaxe = self.jitter_color_fn(imaxe)
            if self.variar_enfoque: imaxe = self.variar_enfoque_fn(imaxe)

        imaxe = transforms.ToTensor()(imaxe)
        mascara = transforms.ToTensor()(mascara)

        if self.aumento_datos: # en torch
            if self.anade_ruido_gaussiano: imaxe = self.anade_ruido_gaussiano_fn(imaxe)
            if self.transformacion_afin: imaxe, mascara = self.transformacion_afin_fn(imaxe,mascara)
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

    def jitter_color_fn(self, imaxe):
        jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)
        return jitter(imaxe)

    def variar_enfoque_fn(self, imaxe):
        if torch.rand(1) > 0.5:
            return transforms.functional.adjust_sharpness(imaxe, 2)
        else:
            return transforms.functional.gaussian_blur(imaxe, kernel_size=3)

    def transformacion_afin_fn(self, imaxe, mascara):
        if torch.rand(1) > 0.7:
            angle = torch.empty(1).uniform_(-15, 15).item()  # Rotation angle
            translate = (torch.empty(2).uniform_(-0.4, 0.4).tolist())  # Translation (dx, dy)
            scale = torch.empty(1).uniform_(0.8, 1.1).item()  # Scaling factor
            shear = torch.empty(1).uniform_(-0.4, 0.4).item()  # Shear factor
            
            imaxe = transforms.functional.affine(imaxe, angle=angle, translate=translate, scale=scale, shear=shear, interpolation=InterpolationMode.BILINEAR)
            mascara = transforms.functional.affine(mascara, angle=angle, translate=translate, scale=scale, shear=shear, interpolation=InterpolationMode.NEAREST)
    
        return imaxe, mascara


    def anade_ruido_gaussiano_fn(self, imaxe):
        noise = torch.randn_like(imaxe) * 0.05
        return torch.clamp(imaxe + noise, 0.0, 1.0)

    def deformacion_elastica_fn(self, imaxe, mascara, alpha=600, sigma=17):
        if torch.rand(1) > 0.7:
            c, h, w = imaxe.shape
            
           
            dx = np.random.rand(h, w) * 2 - 1  # Desplazamentos aleatorios
            dy = np.random.rand(h, w) * 2 - 1
            
            dx = cv2.GaussianBlur(dx, (0, 0), sigma) # suavizado dos dx e dy
            dy = cv2.GaussianBlur(dy, (0, 0), sigma)
            
            dx = torch.from_numpy(dx.astype(np.float32)) * alpha # escalado
            dy = torch.from_numpy(dy.astype(np.float32)) * alpha
            
            x, y = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            x = x.float() + dx
            y = y.float() + dy
        
            x = torch.clamp(x, 0, h - 1) # [0,1]
            y = torch.clamp(y, 0, w - 1)
            
            grid = torch.stack((y / (w - 1) * 2 - 1, x / (h - 1) * 2 - 1), dim=-1).unsqueeze(0)
            
                # Aplicar á imaxxe e á mascara
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
        frangi_imx = (frangi_imx / frangi_imx.max() if frangi_imx.max() > 0 else frangi_imx).astype(np.float32)
        return torch.tensor(frangi_imx).unsqueeze(0)


class PostProcesado:
    def __init__(self,
                 aplicar_umbral=True,
                 valor_umbral=0.5,
                 aplicar_opening=True,
                 tamano_kernel_opening=3,
                 eliminar_objetos_pequenos=True,
                 tamano_minimo_objeto=100,
                 suavizado=True,
                 realce_bordes=False):
        
        self.aplicar_umbral = aplicar_umbral
        self.valor_umbral = valor_umbral

        self.aplicar_opening = aplicar_opening
        self.tamano_kernel_opening = tamano_kernel_opening

        self.eliminar_objetos_pequenos = eliminar_objetos_pequenos
        self.tamano_minimo_objeto = tamano_minimo_objeto

        self.suavizado = suavizado
        self.realce_bordes = realce_bordes

    def __call__(self, prediccion):
        pred_np = self.preparar_prediccion(prediccion)

        if self.aplicar_umbral:
            pred_np = self.umbral(pred_np)

        if self.aplicar_opening:
            pred_np = self.apertura(pred_np)

        if self.eliminar_objetos_pequenos:
            pred_np = self.limpiar_objetos_pequenos(pred_np)

        if self.suavizado:
            pred_np = self.suavizar(pred_np)

        if self.realce_bordes:
            pred_np = self.resaltar_bordes(pred_np)

        #return self.tensorizar(pred_np)
        return pred_np

    def preparar_prediccion(self, tensor):
        if isinstance(tensor, torch.Tensor): arr = tensor.detach().cpu().numpy()
        else: arr = tensor

        if arr.ndim == 4: arr = arr[0, 0]
        elif arr.ndim == 3: arr = arr[0]
        
        return np.clip(arr, 0, 1)

    def umbral(self, pred):
        return (pred > self.valor_umbral).astype(np.uint8)

    def apertura(self, binaria):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (self.tamano_kernel_opening, self.tamano_kernel_opening))
        return cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)

    def limpiar_objetos_pequenos(self, binaria):
        limpia = remove_small_objects(binaria.astype(bool), min_size=self.tamano_minimo_objeto)
        return limpia.astype(np.uint8)

    def suavizar(self, binaria):
        suav = cv2.GaussianBlur(binaria.astype(np.float32), (3, 3), 0)
        return (suav > 0.5).astype(np.uint8)

    def resaltar_bordes(self, binaria):
        bordes = cv2.Canny((binaria * 255).astype(np.uint8), 50, 150)
        return np.maximum(binaria, bordes // 255)

    def tensorizar(self, arr):
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
