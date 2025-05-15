from Rede import UNet
from ConxuntoDatosC import ConxuntoDatosOCT
from adestramento import adestra, valida
from utilidades import redondea_a_tamano_valido
from graficas import grafica_curvas, grafica_prediccions
from Perdidas import Perdidas

import gc
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
    parser.add_argument('--probabilidade_dropout', type=float, default=0.01, help='Profundidade de Dropout')
    parser.add_argument('--peso_bce', type=float, default=0.7, help='Importancia do BCE na funcion de perdida')
    parser.add_argument('--peso_dice', type=float, default=0.4, help='Importancia do DICE na funcion de perdida')
    parser.add_argument('--peso_focal', type=float, default=0.2, help='Importancia do Focal na funcion de perdida')
    parser.add_argument('--peso_iou', type=float, default=0.7, help='Importancia do IOU na funcion de perdida')
    parser.add_argument('--paciencia', type=int, default=10, help='Paciencia de adestramento.')
    parser.add_argument('--paciencia_paso', type=int, default=5, help='Paciencia de paso de aprendizaxe.')
    parser.add_argument("--novo_tamano", type=int, nargs=2, default=(416,624), metavar=('ancho', 'alto'), help="Tamaño do recorte.")

    parser.add_argument('--dispositivo', type=str, choices=['mps', 'cuda', 'cpu', 'gpu'], default='mps', help=f'Dispositivo, por defecto mps.')
    parser.add_argument('--aumento_datos', action='store_true')
    parser.add_argument('--postprocesado', action='store_true')
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
        f"ad{args.aumento_datos}_"
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
    
    print(f'''
Executando ca seguinte configuracion: 

    -> Razon adestramento: {args.razon}
    -> Tamaño batch: {args.tamano_batch}
    -> Procesos na carga de datos: {args.procesos}
    -> Número máximo de épocas: {args.epocas}
    -> Número de canles de saída na primeira convolucion: {args.canles_base}
    -> Profundidade da UNet: {args.profundidade}
    -> Paso de aprendizaxe: {args.paso}
    -> Fctor de reducción do paso: {args.factor_paso}
    -> Probabilidade de Dropout: {args.probabilidade_dropout}
    -> Paciencia: {args.paciencia}
    -> Paciencia do paso: {args.paciencia_paso}
    -> Tamaño das imaxes: {args.novo_tamano}
    -> Dispositivo: {args.dispositivo}
    -> Aumento de datos? {args.aumento_datos}
    -> PostProcesado? {args.postprocesado}
    -> Verboso? {args.verboso}
    -> Mostra? {args.mostra}
    -> Redondea tamaño? {args.redondea_antes}
    -> Añade Canny? {args.anade_canny}
    -> Añade Sobel? {args.anade_sobel}
    -> Añade Laplacian? {args.anade_laplacian}
    -> Añade Frangi? {args.anade_frangi}

----------------------------------------------------
''')


    ruta  = '../OCT-dataset' 
    semilla = np.random.randint(1e6)
    
    print(f'Semilla: {semilla}')
    
    if args.redondea_antes:
        args.novo_tamano =  redondea_a_tamano_valido(args.novo_tamano, args.profundidade)
        if args.verboso: print(f'Tamaño redondeado {args.novo_tamano}')

    adestramento = ConxuntoDatosOCT(
        ruta=ruta, 
        aumento_datos=args.aumento_datos, 
        #
        apply_intelligent_crop=False,
        #
        particion='adestramento', 
        novo_tamano=args.novo_tamano, 
        razon=args.razon, 
        semilla=semilla, 
        anade_canny=args.anade_canny, 
        anade_sobel=args.anade_sobel, 
        anade_laplacian=args.anade_laplacian,
        anade_frangi=args.anade_frangi)
    
    validacion = ConxuntoDatosOCT(
        ruta=ruta, 
        #
        apply_intelligent_crop=False,
        #
        particion='validacion', 
        novo_tamano=args.novo_tamano, 
        razon=args.razon, 
        semilla=semilla, 
        anade_canny=args.anade_canny, 
        anade_sobel=args.anade_sobel, 
        anade_laplacian=args.anade_laplacian,
        anade_frangi=args.anade_frangi)

    proba = ConxuntoDatosOCT(
        ruta=ruta, 
        #
        apply_intelligent_crop=False,
        #
        particion='proba', 
        novo_tamano=args.novo_tamano, 
        razon=args.razon, 
        semilla=semilla, 
        anade_canny=args.anade_canny, 
        anade_sobel=args.anade_sobel, 
        anade_laplacian=args.anade_laplacian,
        anade_frangi=args.anade_frangi)

    cargador_adestramento = DataLoader(
        adestramento, 
        batch_size=args.tamano_batch, 
        num_workers=args.procesos,
        pin_memory=True, 
        persistent_workers=True, 
        shuffle=True)

    cargador_validacion = DataLoader(
        validacion, 
        batch_size=args.tamano_batch, 
        num_workers=args.procesos, 
        pin_memory=True, 
        persistent_workers=True)

    cargador_proba = DataLoader(
        proba, 
        batch_size=args.tamano_batch, 
        num_workers=args.procesos, 
        pin_memory=True, 
        persistent_workers=True)


    canles_entrada = adestramento[0][0].shape[0]
    if args.verboso: print(f'canles_entrada {canles_entrada}, adestramento[0][0].shape {adestramento[0][0].shape}')

    modelo = UNet(
        canles_entrada=canles_entrada, 
        canles_base=args.canles_base, 
        profundidade=args.profundidade, 
        probabilidade_dropout=args.probabilidade_dropout)

    modelo.to(args.dispositivo)

    optimizador = optim.AdamW(modelo.parameters(), lr=args.paso, weight_decay=1e-2) 
    planificador_paso = ReduceLROnPlateau(optimizador, mode='min', factor=args.factor_paso, patience=args.paciencia_paso)

    perdida_compuesta = lambda logits, mascaras: Perdidas.perdida_combinada(
        logits, 
        mascaras, 
        pesos={'bce': args.peso_bce,
               'dice': args.peso_dice,
               'focal': args.peso_focal,
               'iou': args.peso_iou}
    )
    
    historia_perdidas, historia_metricas = adestra(
        modelo=modelo, 
        cargador_adestramento=cargador_adestramento, 
        cargador_validacion=cargador_validacion, 
        perdida_compuesta=perdida_compuesta,
        optimizador=optimizador, 
        dispositivo=args.dispositivo, 
        epocas=args.epocas,
        paciencia=args.paciencia,
        planificador_paso=planificador_paso,
        verboso=args.verboso,
        nome=nome
    )

    perdidas_proba, metricas_proba = valida(
        modelo=modelo, 
        cargador=cargador_proba, 
        perdida_compuesta=perdida_compuesta,
        dispositivo=args.dispositivo, 
        verboso=True, 
        tipo='PROBA'
    )

    dice_proba = metricas_proba['dice']
    nome = nome + f'diceProba{dice_proba:.3f}' 

    grafica_curvas(
        perdidas=historia_perdidas,
        metricas=historia_metricas,
        nome=nome, 
        mostra=args.mostra
    )
    grafica_prediccions(
        modelo, 
        cargador_proba, 
        dispositivo=args.dispositivo, 
        filas=3, 
        nome=nome, 
        mostra=args.mostra, 
        postprocesado=args.postprocesado)
    
    del cargador_adestramento
    del cargador_validacion
    del cargador_proba

    gc.collect()
