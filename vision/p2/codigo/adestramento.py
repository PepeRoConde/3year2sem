import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

def adestra_unha_epoca(modelo, cargador, funcion_perdida, optimizador, dispositivo, epoca):
    
    modelo.train(mode=True) # indica á rede que estamos en modo adestramento
    perdida_epoca = 0.0
   
    bucle = tqdm(cargador, desc=f"Época {epoca} [Adestramento]", leave=False)
    for imaxes, mascaras in bucle:
        imaxes = imaxes.to(dispositivo)
        mascaras = mascaras.to(dispositivo)

        optimizador.zero_grad()
        saidas = modelo(imaxes)
        perdida = funcion_perdida(saidas, mascaras)
        perdida.backward()
        optimizador.step()

        dice = coeficiente_dice(saidas, mascaras)

        perdida_epoca += perdida.item()
        bucle.set_postfix(loss=perdida.item(), dice=dice.item())

    perdida_media = perdida_epoca / len(cargador)
    return perdida_media


def valida(modelo, cargador, funcion_perdida, dispositivo, epoca, verboso):
    
    modelo.train(mode=False) # indica á rede que xa no estamos en modo adestramento
    perdida_validacion = 0
    dice = 0

    with torch.no_grad():

        for imaxes, mascaras in tqdm(cargador, desc=f"Época {epoca} [Validacion]", leave=False):
            imaxes = imaxes.to(dispositivo)
            mascaras = mascaras.to(dispositivo)

            saidas = modelo(imaxes)
            perdida = funcion_perdida(saidas, mascaras)
            perdida_validacion += perdida.item()

            # Optional: calculate Dice or IoU metric
            prediccions = torch.sigmoid(saidas)
            prediccions = (prediccions > 0.5).float()
            dice += coeficiente_dice(prediccions, mascaras)

    perdida_media = perdida_validacion / len(cargador)
    dice_medio = dice / len(cargador)
    if verboso: print(f'VALIDACION: Perdida media: {perdida_media:.4f} || Dice medio: {dice_medio:.4f}')
    return perdida_media, dice_medio

def adestra(modelo, cargador_adestramento, cargador_validacion, funcion_perdida, optimizador, dispositivo, epocas, paciencia, planificador_paso, verboso):

    perdidas = []
    perdidas_validacion = []
    dices = []
    mellor_perda_validacion = float('inf')
    conta_sen_mellora = 0
    paso = planificador_paso.get_last_lr()
    
    for epoca in range(1, epocas + 1):

        p = adestra_unha_epoca(modelo, cargador_adestramento, funcion_perdida, optimizador, dispositivo, epoca)
        pv, d = valida(modelo, cargador_validacion, funcion_perdida, dispositivo, epoca, verboso)
        perdidas.append(float(p)); perdidas_validacion.append(float(pv)); dices.append(float(d))
         
        if (paso != planificador_paso.get_last_lr()) and verboso:
            print(f'Novo paso de aprendizaxe: {planificador_paso.get_last_lr()[0]}')
            paso = planificador_paso.get_last_lr()
    
        if float(pv) < mellor_perda_validacion: 
            mellor_perda_validacion = float(pv)
            conta_sen_mellora = 0
        else: 
            conta_sen_mellora += 1
            if conta_sen_mellora == paciencia:
                print(f"Early stopping na época {epoca+1}. Mellor perda de validación: {mellor_perda_validacion:.4f}")
                torch.save(modelo.state_dict(), f"../parametros/unet_perdida_val_{pv}.pth")
                break

        planificador_paso.step(float(pv))
    return perdidas, perdidas_validacion, dices

# mirar si dice la dejo aqui o va en losses (igual losses y metricas?)
# Utility Dice function (for binary segmentation)
def coeficiente_dice(preds: torch.Tensor, targets: torch.Tensor, smooth=1e-6):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
