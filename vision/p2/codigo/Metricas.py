import torch
import torch.nn as nn
from typing import Dict, Optional


class Metricas:
    @staticmethod
    def calcula_metricas(logits, mascara):
        probs = torch.sigmoid(logits)
        prediccions = (probs > 0.5).float()
        
        return  {'dice': Metricas.dice(prediccions, mascara),
                 'iou': Metricas.iou(prediccions, mascara),                 
                 'precision': Metricas.precision(prediccions, mascara),     
                 'recall': Metricas.recall(prediccions, mascara),           
                 'f1_score': Metricas.f1_score(prediccions, mascara)}       

    @staticmethod
    def dice(prediccions, mascara, suavizado = 1e-7):
        prediccions = prediccions.view(-1)
        mascara = mascara.view(-1)
        interseccion = (prediccions * mascara).sum()
        return (2. * interseccion + suavizado) / (prediccions.sum() + mascara.sum() + suavizado)
    
    @staticmethod
    def iou(prediccions, mascara, suavizado = 1e-7):
        prediccions = prediccions.view(-1)
        mascara = mascara.view(-1)
        interseccion = (prediccions * mascara).sum()
        union = prediccions.sum() + mascara.sum() - interseccion
        return (interseccion + suavizado) / (union + suavizado)
    
    @staticmethod
    def precision(prediccions, mascara, suavizado = 1e-7):
        prediccions = prediccions.view(-1)
        mascara = mascara.view(-1)
        verdadeiros_positivos = (prediccions * mascara).sum()
        return (verdadeiros_positivos + suavizado) / (prediccions.sum() + suavizado)
    
    @staticmethod
    def recall(prediccions, mascara, suavizado = 1e-7):
        prediccions = prediccions.view(-1)
        mascara = mascara.view(-1)
        verdadeiros_positivos = (prediccions * mascara).sum()
        return (verdadeiros_positivos + suavizado) / (mascara.sum() + suavizado)
    
    @staticmethod
    def f1_score(prediccions, mascara):
        prec = Metricas.precision(prediccions, mascara)
        rec = Metricas.recall(prediccions, mascara)
        return 2 * (prec * rec) / (prec + rec + 1e-7)
