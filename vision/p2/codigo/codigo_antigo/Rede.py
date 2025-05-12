import torch.nn as nn
from torch import cat

class UNet(nn.Module):

    def __init__(self, canles_entrada=1, num_clases=1):
        super().__init__()
        
        self.contrainte1 = self.doble_convolucion(canles_entrada, 2**6)
        self.contrainte2 = self.doble_convolucion(2**6, 2**7)
        self.contrainte3 = self.doble_convolucion(2**7, 2**8)
        self.contrainte4 = self.doble_convolucion(2**8, 2**9)
        self.contrainte5 = self.doble_convolucion(2**9, 2**10) # difire do exemplo do notebook pero así é igual (en canto a canles) ó paper
    
        self.maxpool = nn.MaxPool2d(2) # deste xeito a dimension espacial decrece a razon de 1/4
        
        self.expansivo4 = self.doble_convolucion(2**9 + 2**10, 2**9) 
        self.expansivo3 = self.doble_convolucion(2**8 + 2**9, 2**8)
        self.expansivo2 = self.doble_convolucion(2**7 + 2**8, 2**7)
        self.expansivo1 = self.doble_convolucion(2**6 + 2**7, 2**6)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # para que coincidan ca reduccion do MaxPool2d

        self.derradeira_convolucion = nn.Conv2d(2**6, num_clases, 1)
            
    def doble_convolucion(self, canles_entrada, canles_saida):
        return nn.Sequential(
            nn.Conv2d(canles_entrada, canles_saida, 3, padding=1),
            nn.ReLU(inplace=True), # deste xeito aforramos memoria
            nn.Conv2d(canles_saida, canles_saida, 3, padding=1),
            nn.ReLU(inplace=True))   

    def forward(self, x):

        c1 = self.contrainte1(x) #-- camiño contrainte
        x = self.maxpool(c1)

        c2 = self.contrainte2(x)
        x = self.maxpool(c2)

        c3 = self.contrainte3(x)
        x = self.maxpool(c3)

        x = self.contrainte4(x) # c4
        #x = self.maxpool(c4)
        
        #x = self.contrainte5(x)
        #x = self.upsample(x) #-- camiño expansivo

        #x = self.expansivo4(cat([c4, x], dim=1))
        x = self.upsample(x)
       
        x = self.expansivo3(cat([c3, x], dim=1))
        x = self.upsample(x)

        x = self.expansivo2(cat([c2, x], dim=1))
        x = self.upsample(x)

        x = self.expansivo1(cat([c1, x], dim=1))
        saida = self.derradeira_convolucion(x)
        
        return saida 
