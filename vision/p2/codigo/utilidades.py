import torch
import torch.nn as nn
import torch.nn.functional as F

def axusta_tamano(atallo, x, modo='pad'):
    if atallo.size(2) == x.size(2) and atallo.size(3) == x.size(3):
        return x  # Already aligned
    diffY = atallo.size(2) - x.size(2)
    diffX = atallo.size(3) - x.size(3)
    if modo == 'pad':
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
    elif modo == 'crop':
        x = x[:, :, diffY // 2 : x.size(2) + diffY // 2, diffX // 2 : x.size(3) + diffX // 2]
    else:
        raise ValueError("Modo é 'pad' ou 'crop'")
    return x


# https://gist.github.com/fepegar/1fb865494cb44ac043c3189ec415d411
def redondea_a_tamano_valido(tamano, profundidade):
    multiplo = 2 ** profundidade
    alto, ancho = tamano
    novo_alto = round(alto / multiplo) * multiplo
    novo_ancho = round(ancho / multiplo) * multiplo
    return (novo_alto, novo_ancho)
