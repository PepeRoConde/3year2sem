import torch
import torch.nn as nn
from typing import Dict, Optional

class Perdidas:
    @staticmethod
    def perdida_dice(logits, mascara, suavizado = 1.0):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        mascara = mascara.view(-1)

        intersection = (probs * mascara).sum()
        dice = (2. * intersection + suavizado) / (probs.sum() + mascara.sum() + suavizado)
        
        return 1 - dice
    
    @staticmethod
    def perdida_focal(logits, mascara, alpha = 0.25, gamma = 2.0):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(logits, mascara, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = alpha * (1-pt)**gamma * BCE_loss
        return F_loss.mean()
    
    @staticmethod
    def perdida_iou(logits, mascara, suavizado = 1.0):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        mascara = mascara.view(-1)
        
        interseccion = (probs * mascara).sum()
        union = probs.sum() + mascara.sum() - interseccion
        
        iou = (interseccion + suavizado) / (union + suavizado)
        return 1 - iou

    @staticmethod
    def perdida_combinada(logits, mascara, pesos = None):
        
        defecto = {
            'bce': (nn.functional.binary_cross_entropy_with_logits, 1.0),
            'dice': (Perdidas.perdida_dice, 1.0),
            'focal': (Perdidas.perdida_focal, 0.5),
            'iou': (Perdidas.perdida_iou, 0.5)
        }

        if pesos:
            for nome, peso in pesos.items():
                if nome in defecto:
                    defecto[nome] = (defecto[nome][0], peso)
        
        componentes = {}
        perdida_total = 0.0
        
        for nome, (funcion, peso) in defecto.items():
            if peso > 0:
                perdida = funcion(logits, mascara)
                componentes[nome] = perdida
                perdida_total += peso * perdida
        
        componentes['total'] = perdida_total
        
        return componentes
