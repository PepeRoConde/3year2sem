from transformacions import SegmentationTransform

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
    
    def __init__(self, ruta, particion: int = 0,  razon = 0.8, novo_tamano = (416,624), transform = None, semilla=1942):
        super().__init__()
        ruta_imaxes, ruta_mascaras = ruta + '/images', ruta + '/masks'
        self.rutas_imaxes = np.array(glob.glob(os.path.join(ruta_imaxes,'*.jpg')))
        self.rutas_mascaras = np.array([os.path.join(ruta_mascaras, os.path.basename(imaxe)) for imaxe in self.rutas_imaxes])
        
        N = len(self.rutas_imaxes)

        num_exemplos_adestramento = int(razon * N)
        num_exemplos_validacion = int((N - num_exemplos_adestramento)/2)
        
        np.random.seed(semilla)
        indices_aleatorios = np.random.permutation(np.arange(N)) 
        
        if particion == 'adestramento':
            self.rutas_imaxes = self.rutas_imaxes[indices_aleatorios[:num_exemplos_adestramento]]
            self.rutas_mascaras = self.rutas_mascaras[indices_aleatorios[:num_exemplos_adestramento]]
            self.transform = SegmentationTransform(resize=novo_tamano, is_train=True)
        if particion == 'validacion':
            self.rutas_imaxes = self.rutas_imaxes[indices_aleatorios[num_exemplos_adestramento:num_exemplos_adestramento + num_exemplos_validacion]]
            self.rutas_mascaras = self.rutas_mascaras[indices_aleatorios[num_exemplos_adestramento:num_exemplos_adestramento + num_exemplos_validacion]]
            self.transform = SegmentationTransform(resize=novo_tamano, is_train=False)
        if particion == 'proba':
            self.rutas_imaxes = self.rutas_imaxes[indices_aleatorios[num_exemplos_adestramento + num_exemplos_validacion:]]
            self.rutas_mascaras = self.rutas_mascaras[indices_aleatorios[num_exemplos_adestramento + num_exemplos_validacion:]]
            self.transform = SegmentationTransform(resize=novo_tamano, is_train=False)

        # parámetros do Aumento de Datos
        #self.rsize = rsize  # Size to use in default Resize transform
        self.novo_tamano= novo_tamano
    def _comproba2D(self, imaxe):
        if len(imaxe.shape) > 2:
            return imaxe[:,:,0]
        return imaxe

    def __getitem__(self, indice):
        imaxe = self._comproba2D(plt.imread(self.rutas_imaxes[indice]))
        mascara = self._comproba2D(plt.imread(self.rutas_mascaras[indice]))
        
        _, mascara = cv2.threshold(mascara, 100, 255, cv2.THRESH_BINARY) # todo o superior a 100 sera 255 e o inferior 0 (por ter especificado binary)

        #t = transforms.Compose([
        #    transforms.ToPILImage(),
        #    transforms.Resize(self.novo_tamano, interpolation= InterpolationMode.NEAREST),
        #    transforms.ToTensor()])

        #imaxe = t(imaxe)
        #mascara = t(mascara)
        
        imaxe, mascara = self.transform(imaxe, mascara)

        return imaxe, mascara

    def __len__(self):
        return len(self.rutas_imaxes)

