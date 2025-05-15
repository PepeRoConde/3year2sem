from transformacions import Transformacions
from torch.utils.data import Dataset, DataLoader 
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as transforms
import glob
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import cv2

class ConxuntoDatosOCT(Dataset):
    
    def __init__(self, ruta, aumento_datos=False, particion='adestramento', razon=0.8, 
                 novo_tamano=(416, 624), transform=None, semilla=1942, 
                 anade_canny=False, anade_sobel=False, anade_laplacian=False, 
                 anade_frangi=False, apply_intelligent_crop=False, crop_margin=0.2):
        super().__init__()
        ruta_imaxes, ruta_mascaras = ruta + '/images', ruta + '/masks'
        self.rutas_imaxes = np.array(glob.glob(os.path.join(ruta_imaxes,'*.jpg')))
        self.rutas_mascaras = np.array([os.path.join(ruta_mascaras, os.path.basename(imaxe)) for imaxe in self.rutas_imaxes])
        
        self.apply_intelligent_crop = apply_intelligent_crop
        self.crop_margin = crop_margin  # Amount of margin to keep around the mask (relative to mask size)
        
        N = len(self.rutas_imaxes)
        num_exemplos_adestramento = int(razon * N)
        num_exemplos_validacion = int((N - num_exemplos_adestramento)/2)
        
        np.random.seed(semilla)
        indices_aleatorios = np.random.permutation(np.arange(N)) 
        
        if particion == 'adestramento':
            self.rutas_imaxes = self.rutas_imaxes[indices_aleatorios[:num_exemplos_adestramento]]
            self.rutas_mascaras = self.rutas_mascaras[indices_aleatorios[:num_exemplos_adestramento]]
            self.transform = Transformacions(novo_tamano=novo_tamano, aumento_datos=aumento_datos, 
                                           anade_canny=anade_canny, anade_sobel=anade_sobel, 
                                           anade_laplacian=anade_laplacian, anade_frangi=anade_frangi)
        if particion == 'validacion':
            self.rutas_imaxes = self.rutas_imaxes[indices_aleatorios[num_exemplos_adestramento:num_exemplos_adestramento + num_exemplos_validacion]]
            self.rutas_mascaras = self.rutas_mascaras[indices_aleatorios[num_exemplos_adestramento:num_exemplos_adestramento + num_exemplos_validacion]]
            self.transform = Transformacions(novo_tamano=novo_tamano, 
                                           anade_canny=anade_canny, anade_sobel=anade_sobel, 
                                           anade_laplacian=anade_laplacian, anade_frangi=anade_frangi)
        if particion == 'proba':
            self.rutas_imaxes = self.rutas_imaxes[indices_aleatorios[num_exemplos_adestramento + num_exemplos_validacion:]]
            self.rutas_mascaras = self.rutas_mascaras[indices_aleatorios[num_exemplos_adestramento + num_exemplos_validacion:]]
            self.transform = Transformacions(novo_tamano=novo_tamano, 
                                           anade_canny=anade_canny, anade_sobel=anade_sobel, 
                                           anade_laplacian=anade_laplacian, anade_frangi=anade_frangi)
        
    def _comproba2D(self, imaxe):
        if len(imaxe.shape) > 2:
            return imaxe[:,:,0]
        return imaxe
    
    def intelligent_crop(self, image, mask):
        """
        Crop the image and mask based on the mask boundaries with added margin
        """
        # Find contours in the mask to locate the region of interest
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:  # If no contours found, return original
            return image, mask
        
        # Find bounding box for all contours
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = 0, 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        
        # Calculate dimensions of the mask region
        width = x_max - x_min
        height = y_max - y_min
        
        # Add margin around the mask (proportional to mask size)
        margin_x = int(width * self.crop_margin)
        margin_y = int(height * self.crop_margin)
        
        # Calculate new crop coordinates with margins
        crop_x_min = max(0, x_min - margin_x)
        crop_y_min = max(0, y_min - margin_y)
        crop_x_max = min(image.shape[1], x_max + margin_x)
        crop_y_max = min(image.shape[0], y_max + margin_y)
        
        # Crop both image and mask
        cropped_image = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        cropped_mask = mask[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        
        return cropped_image, cropped_mask
    
    def __getitem__(self, indice):
        imaxe = self._comproba2D(plt.imread(self.rutas_imaxes[indice]))
        mascara = self._comproba2D(plt.imread(self.rutas_mascaras[indice]))
        
        # Binarize the mask
        _, mascara = cv2.threshold(mascara, 100, 255, cv2.THRESH_BINARY)
        
        # Apply intelligent cropping if enabled
        if self.apply_intelligent_crop:
            if np.random.random() < 0.2:
                imaxe, mascara = self.intelligent_crop(imaxe, mascara)
        
        return self.transform(imaxe, mascara)
    
    def __len__(self):
        return len(self.rutas_imaxes)