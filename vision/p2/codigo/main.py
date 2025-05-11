from RedeC import UNet
from ConxuntoDatos import ConxuntoDatosOCT
from adestramento import adestra
from utilidades import BCEWithLogitsLoss, BCEDiceLoss, plot_losses_and_dice, plot_images_and_predictions 


import argparse
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

#temporal
from torch import optim
import matplotlib.pyplot as plt


def parsea_argumentos():
    parser = argparse.ArgumentParser(description='Segmentacion de imaxes OCT')
    parser.add_argument('--razon', type=float, default=0.8, help='Razon do conxunto de datos que destinarase a adestramento (validacion e proba repartirasen a metades iguais coas imaxes restantes).')
    parser.add_argument('--tamano_batch', type=int, default=8, help='Número de exemplos por minibatch.')
    parser.add_argument('--procesos', type=int, default=6, help='Número de procesos usados na carga de datos.')
    parser.add_argument('--epocas', type=int, default=50, help='Número de epocas no adestramento.')
    parser.add_argument('--canles_base', type=int, default=32, help='Número de canles na salida da primeira convolucion (No artigo 64).')
    parser.add_argument('--profundidade', type=int, default=3, help='Profundidade da UNet (No artigo 4).')
    parser.add_argument('--paso', type=float, default=1e-3, help='Paso de aprendizaxe.')
    parser.add_argument('--paciencia', type=int, default=10, help='Paciencia de adestramento.')
    parser.add_argument('--paciencia_paso', type=int, default=5, help='Paciencia de paso de aprendizaxe.')
    parser.add_argument("--novo_tamano", type=int, nargs=2, default=(416,624), metavar=('ancho', 'alto'), help="Tamaño do recorte.")

    parser.add_argument('--dispositivo', type=str, choices=['mps', 'cuda', 'cpu', 'gpu'], default='mps', help=f'Dispositivo, por defecto mps.')
    parser.add_argument('--verboso', action='store_true')

    return parser.parse_args()

if __name__ == "__main__":
    args = parsea_argumentos()

    ruta  = '../OCT-dataset' 
    semilla = np.random.randint(1e6)

    #args.novo_tamano = (416,624)

    adestramento = ConxuntoDatosOCT(ruta=ruta, particion='adestramento', novo_tamano=args.novo_tamano, razon=args.razon, semilla=semilla)
    validacion = ConxuntoDatosOCT(ruta=ruta, particion='validacion', novo_tamano=args.novo_tamano, razon=args.razon, semilla=semilla)
    proba = ConxuntoDatosOCT(ruta=ruta, particion='proba', novo_tamano=args.novo_tamano, razon=args.razon, semilla=semilla)

    cargador_adestramento = DataLoader(adestramento, batch_size=args.tamano_batch, num_workers=args.procesos,pin_memory=True, persistent_workers=True, shuffle=True)
    cargador_validacion = DataLoader(validacion, batch_size=args.tamano_batch, num_workers=args.procesos, pin_memory=True, persistent_workers=True)
    cargador_proba = DataLoader(proba, batch_size=args.tamano_batch, num_workers=args.procesos, pin_memory=True, persistent_workers=True)


    canles_entrada = 1 if len(adestramento[0][0].shape) == 3 else adestramento[0][0].shape[3]

    modelo = UNet(canles_entrada=canles_entrada, canles_base=args.canles_base, profundidade=args.profundidade)
    modelo.to(args.dispositivo)

    optimizador = optim.AdamW(modelo.parameters(), lr=args.paso, weight_decay=1e-2) 
    planificador_paso = ReduceLROnPlateau(optimizador, mode='min', factor=0.5, patience=args.paciencia_paso)
    #funcion_perdida = BCEWithLogitsLoss() 
    funcion_perdida = BCEDiceLoss()

    perdidas, perdidas_validacion, dices = adestra(modelo=modelo, 
            cargador_adestramento=cargador_adestramento, 
            cargador_validacion=cargador_validacion, 
            funcion_perdida=funcion_perdida, 
            optimizador=optimizador, 
            dispositivo=args.dispositivo, 
            epocas=args.epocas,
            paciencia=args.paciencia,
            planificador_paso=planificador_paso,
            verboso=args.verboso)
    
    plot_losses_and_dice(perdidas, perdidas_validacion, dices)
    plot_images_and_predictions(modelo, cargador_proba, dispositivo=args.dispositivo, n_rows=5)
