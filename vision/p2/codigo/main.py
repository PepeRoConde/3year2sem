from RedeC import UNet
from ConxuntoDatos import ConxuntoDatosOCT
from adestramento import adestra, valida
from utilidades import BCEWithLogitsLoss, BCEDiceLoss, plot_losses_and_dice, plot_images_and_predictions, redondea_a_tamano_valido 


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
    parser.add_argument('--factor_paso', type=float, default=0.7, help='Cando o paso reduzacase; paso(t+1) := paso(t) * factor ')
    parser.add_argument('--probabilidade_dropout', type=float, default=0.05)
    parser.add_argument('--paciencia', type=int, default=10, help='Paciencia de adestramento.')
    parser.add_argument('--paciencia_paso', type=int, default=5, help='Paciencia de paso de aprendizaxe.')
    parser.add_argument("--novo_tamano", type=int, nargs=2, default=(416,624), metavar=('ancho', 'alto'), help="Tamaño do recorte.")

    parser.add_argument('--dispositivo', type=str, choices=['mps', 'cuda', 'cpu', 'gpu'], default='mps', help=f'Dispositivo, por defecto mps.')
    parser.add_argument('--aumento_datos', action='store_true')
    parser.add_argument('--verboso', action='store_true')
    parser.add_argument('--mostra', action='store_true')
    parser.add_argument('--redondea_antes', action='store_true')
    parser.add_argument('--anade_canny', action='store_true')
    parser.add_argument('--anade_sobel', action='store_true')
    parser.add_argument('--anade_laplacian', action='store_true')
    parser.add_argument('--anade_frangi', action='store_true')

    return parser.parse_args()

if __name__ == "__main__":
    args = parsea_argumentos()

    nome = (
        f"batch{args.tamano_batch}_"
        f"proc{args.procesos}_"
        f"canles{args.canles_base}_"
        f"prof{args.profundidade}_"
        f"lr{args.paso}_"
        f"factor{args.factor_paso}_"
        f"dropout{args.probabilidade_dropout}_"
        f"pac{args.paciencia}_"
        f"paclr{args.paciencia_paso}_"
        f"tam{args.novo_tamano[0]}x{args.novo_tamano[1]}"
    )

    ruta  = '../OCT-dataset' 
    semilla = np.random.randint(1e6)

    if args.redondea_antes:
        args.novo_tamano =  redondea_a_tamano_valido(args.novo_tamano, args.profundidade)
        print(f'Tamaño redondeado {args.novo_tamano}')

    adestramento = ConxuntoDatosOCT(ruta=ruta, 
                                    aumento_datos=args.aumento_datos, 
                                    particion='adestramento', 
                                    novo_tamano=args.novo_tamano, 
                                    razon=args.razon, 
                                    semilla=semilla, 
                                    anade_canny=args.anade_canny, 
                                    anade_sobel=args.anade_sobel, 
                                    anade_laplacian=args.anade_laplacian,
                                    anade_frangi=args.anade_frangi)
    validacion = ConxuntoDatosOCT(ruta=ruta, 
                                  particion='validacion', 
                                  novo_tamano=args.novo_tamano, 
                                  razon=args.razon, 
                                  semilla=semilla, 
                                  anade_canny=args.anade_canny, 
                                  anade_sobel=args.anade_sobel, 
                                  anade_laplacian=args.anade_laplacian,
                                  anade_frangi=args.anade_frangi)
    proba = ConxuntoDatosOCT(ruta=ruta, 
                             particion='proba', 
                             novo_tamano=args.novo_tamano, 
                             razon=args.razon, 
                             semilla=semilla, 
                             anade_canny=args.anade_canny, 
                             anade_sobel=args.anade_sobel, 
                             anade_laplacian=args.anade_laplacian,
                             anade_frangi=args.anade_frangi)

    cargador_adestramento = DataLoader(adestramento, batch_size=args.tamano_batch, num_workers=args.procesos,pin_memory=True, persistent_workers=True, shuffle=True)
    cargador_validacion = DataLoader(validacion, batch_size=args.tamano_batch, num_workers=args.procesos, pin_memory=True, persistent_workers=True)
    cargador_proba = DataLoader(proba, batch_size=args.tamano_batch, num_workers=args.procesos, pin_memory=True, persistent_workers=True)


    canles_entrada = adestramento[0][0].shape[0]
    print(f'canles_entrada {canles_entrada}, adestramento[0][0].shape {adestramento[0][0].shape}')

    modelo = UNet(canles_entrada=canles_entrada, canles_base=args.canles_base, profundidade=args.profundidade, probabilidade_dropout=args.probabilidade_dropout)
    modelo.to(args.dispositivo)

    optimizador = optim.AdamW(modelo.parameters(), lr=args.paso, weight_decay=1e-2) 
    planificador_paso = ReduceLROnPlateau(optimizador, mode='min', factor=args.factor_paso, patience=args.paciencia_paso)
    #funcion_perdida = BCEWithLogitsLoss() 
    funcion_perdida = BCEDiceLoss()

    perdidas, perdidas_validacion, dices, dices_p, bces = adestra(modelo=modelo, 
            cargador_adestramento=cargador_adestramento, 
            cargador_validacion=cargador_validacion, 
            funcion_perdida=funcion_perdida, 
            optimizador=optimizador, 
            dispositivo=args.dispositivo, 
            epocas=args.epocas,
            paciencia=args.paciencia,
            planificador_paso=planificador_paso,
            verboso=args.verboso,
            nome=nome)
    

    _, dice_medio, _, _ = valida(modelo=modelo, cargador=cargador_proba, funcion_perdida=funcion_perdida, dispositivo=args.dispositivo, verboso=args.verboso, tipo='PROBA')
    
    nome = nome + f'diceProba{dice_medio:.3f}'
    
    plot_losses_and_dice(perdidas, perdidas_validacion, dices, dices_p, bces, nome=nome, mostra=args.mostra)
    plot_images_and_predictions(modelo, cargador_proba, dispositivo=args.dispositivo, n_rows=5, nome=nome, mostra=args.mostra)
