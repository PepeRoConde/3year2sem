from transformacions import axusta_tamano

import torch
import torch.nn as nn
from torch import cat

class UNet(nn.Module):
    def __init__(self, canles_entrada=1, num_clases=1, canles_base=64, profundidade=4):
        super().__init__()
        self.profundidade = profundidade
        self.camino_contraente = nn.ModuleList()
        self.camino_expansivo = nn.ModuleList()
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        canles = canles_entrada # variable mutable

        # Camiño contraente
        for i in range(profundidade):
            canles_saida = canles_base * 2 ** i
            self.camino_contraente.append(self.doble_convolucion(canles, canles_saida))
            canles = canles_saida

        # Fondo da U
        self.fondo = self.doble_convolucion(canles, canles * 2)
        canles = canles * 2

        # Camiño expansivo
        for i in reversed(range(profundidade)):
            canles_atallo = canles_base * 2 ** i
            self.camino_expansivo.append(self.doble_convolucion(canles + canles_atallo, canles_atallo))
            canles = canles_atallo

        self.derradeira_convolucion = nn.Conv2d(canles_base, num_clases, 1)

    def doble_convolucion(self, canles_entrada, canles_saida):
        return nn.Sequential(
            nn.Conv2d(canles_entrada, canles_saida, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(canles_saida, canles_saida, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        atallos = []

        # Camiño contraente
        for capa in self.camino_contraente:
            x = capa(x)
            atallos.append(x)
            x = self.maxpool(x)

        # Fondo
        x = self.fondo(x)

        # Camiño expansivo
        for capa, atallo in zip(self.camino_expansivo, reversed(atallos)):
            x = self.upsample(x)
            x = axusta_tamano(atallo,x)
            x = cat([atallo, x], dim=1)
            x = capa(x)

        return self.derradeira_convolucion(x)

