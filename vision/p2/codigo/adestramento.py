from tqdm import tqdm
import torch
from typing import Dict, Tuple
from Metricas import Metricas

def adestra_unha_epoca(modelo, cargador, perdida_compuesta, optimizador, dispositivo, epoca):

    modelo.train(mode=True)
    acumulador_perdidas = {}
   
    bucle = tqdm(cargador, desc=f"Época {epoca} [Adestramento]", leave=False)
    for imaxes, mascaras in bucle:
        imaxes = imaxes.to(dispositivo)
        mascaras = mascaras.to(dispositivo)
        optimizador.zero_grad()
        saidas = modelo(imaxes)
        
        perdidas = perdida_compuesta(saidas, mascaras)
        perdida_total = perdidas['total']
        
        perdida_total.backward()
        optimizador.step()
        
        for key, value in perdidas.items():
            if key not in acumulador_perdidas:
                acumulador_perdidas[key] = []
            acumulador_perdidas[key].append(value.item())
        
        bucle.set_postfix(loss=perdida_total.item())
    return {key: sum(values)/len(values) for key, values in acumulador_perdidas.items()}

def valida(modelo, cargador, perdida_compuesta, dispositivo, epoca=1, verboso=True, tipo='VALIDACION'):
    modelo.train(mode=False)
    acumulador_perdidas = {}
    metricas = {}
    
    with torch.no_grad():
        acumulador_metricas = {}
        
        for imaxes, mascaras in tqdm(cargador, desc=f"Época {epoca} [Validacion]", leave=False):
            imaxes = imaxes.to(dispositivo)
            mascaras = mascaras.to(dispositivo)
            saidas = modelo(imaxes)
            
            perdidas = perdida_compuesta(saidas, mascaras)
            
            for key, value in perdidas.items():
                if key not in acumulador_perdidas:
                    acumulador_perdidas[key] = []
                acumulador_perdidas[key].append(value.item())
            
            metricas_tmp = Metricas.calcula_metricas(saidas, mascaras)
            
            for key, value in metricas_tmp.items():
                if key not in acumulador_metricas:
                    acumulador_metricas[key] = []
                acumulador_metricas[key].append(value.item())
    
    N = len(cargador)

    perdidas = {key: sum(values)/N for key, values in acumulador_perdidas.items()}
    metricas = {key: sum(values)/N for key, values in acumulador_metricas.items()}
    
    if verboso: 
        print(f'{tipo}: ' + 
              ' || '.join([f'{k}: {v:.4f}' for k, v in {**perdidas, **metricas}.items()]))
    
    return perdidas, metricas

def adestra(modelo, cargador_adestramento, cargador_validacion, perdida_compuesta, optimizador, dispositivo, epocas, paciencia, planificador_paso, verboso, nome):

    historia_perdidas = {}
    historia_metricas = {}
    
    mellor_perda_validacion = float('inf')
    conta_sen_mellora = 0
    paso = planificador_paso.get_last_lr()
    
    for epoca in range(1, epocas + 1):
        perdidas_epoca = adestra_unha_epoca(modelo, cargador_adestramento, perdida_compuesta, optimizador, dispositivo, epoca)
        perdidas, metricas = valida(modelo, cargador_validacion, perdida_compuesta, dispositivo, epoca, verboso)
        
        for key, value in {**perdidas_epoca, **perdidas}.items():
            if key not in historia_perdidas:
                historia_perdidas[key] = []
            historia_perdidas[key].append(value)
        
        for key, value in metricas.items():
            if key not in historia_metricas:
                historia_metricas[key] = []
            historia_metricas[key].append(value)
         
        if planificador_paso.get_last_lr() != paso and verboso:
            print(f'Novo paso de aprendizaxe: {planificador_paso.get_last_lr()[0]}')
            paso = planificador_paso.get_last_lr()
    
        perdida_validacion_actual = perdidas.get('total', float('inf'))
        if perdida_validacion_actual < mellor_perda_validacion: 
            mellor_perda_validacion = perdida_validacion_actual
            conta_sen_mellora = 0
        else: 
            conta_sen_mellora += 1
            if conta_sen_mellora == paciencia:
                print(f"Early stopping na época {epoca+1}. Mellor perda de validación: {mellor_perda_validacion:.4f}")
                break
        
        planificador_paso.step(perdida_validacion_actual)
   
    torch.save(modelo.state_dict(), f"../parametros/{nome}_ep{epoca}_perdVal{mellor_perda_validacion}.pth")
    return historia_perdidas, historia_metricas
