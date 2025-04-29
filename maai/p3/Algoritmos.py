from AlgoritmoAprendizaxeReforzo import AlgoritmoAprendizaxeReforzo
import numpy as np
from tqdm import tqdm
from typing import Tuple, Dict, List, Any

class MonteCarlo(AlgoritmoAprendizaxeReforzo):
    # implementacion de sobre-politica, primeira visista, para políticas epsilon-suaves
    def __init__(self, 
                 gamma: float = 0.99,
                 epsilon: float = 0.1,
                 epsilon_min: float = 0.0001,
                 discretizacion_estado: int | list[int] = 20,
                 discretizacion_accion: int = 20,
                 inicializacion_informada: bool = False,
                 perturbacion: bool = False,
                 determinista: bool = False,
                 primeira_visita: bool = True,
                 epsilon_decae: bool = True):

        super().__init__(gamma=gamma, 
                         epsilon=epsilon, 
                         epsilon_min=epsilon_min,
                         alpha=None,
                         discretizacion_estado=discretizacion_estado,
                         discretizacion_accion=discretizacion_accion,
                         inicializacion_informada=inicializacion_informada,
                         perturbacion=perturbacion,
                         determinista=determinista,
                         epsilon_decae=epsilon_decae)
        self.primeira_visita = primeira_visita

    def actualiza_q(self, historial_episodio: List[Tuple]):
        G = 0
        recompensas = {}
        estados_vistos = []

        for t in range(len(historial_episodio) -1, -1, -1):
            estado, accion, recompensa = historial_episodio[t]
            G = recompensa + self.gamma * G

            if self.primeira_visita:
                if (estado, accion) not in [(s, a) for s, a, _ in historial_episodio[:t]]: # esto es On2 :(
                    if (estado, accion) not in recompensas:
                        recompensas[(estado, accion)] = []
                    recompensas[(estado, accion)].append(G)
                    estados_vistos.append((estado, accion)) # hago esto para hacer las medias solo al final, como sugieres en el notebook
            else:        
                if (estado, accion) not in recompensas:
                    recompensas[(estado, accion)] = []
                recompensas[(estado, accion)].append(G)
                estados_vistos.append((estado, accion)) # hago esto para hacer las medias solo al final, como sugieres en el notebook

        for (estado, accion) in estados_vistos:
            # print(f'actualiza estado {estado} accion {accion} con {np.mean(recompensas[(estado, accion)])}')
            self.q[*estado, accion] = np.mean(recompensas[(estado, accion)])

    def adestra(self, num_episodios, num_maximo_pasos, verboso):
        self.tasa_decaemento = (self.epsilon_0 - self.epsilon_min) / num_episodios 

        barra_progreso = tqdm(range(num_episodios), desc=f"{self.__class__.__name__} Training")
        for episodio in barra_progreso:
            
            if self.epsilon_decae:
                self.actualiza_epsilon(episodio)
            
            estado, _ = self.env.reset()
            historial_episodio = []
            recompensa_total = 0

            for paso in range(num_maximo_pasos):

                estado_dis = self.discretiza_estado(estado)
                accion_dis = self.politica(estado_dis)
                accion = self.continua_accion(accion_dis)
                estado, recompensa, feito, truncado, _ = self.env.step(accion)
                recompensa_total += recompensa
                historial_episodio.append((estado_dis, accion_dis, recompensa))

                if feito or truncado: break
            
            self.actualiza_q(historial_episodio)
            self.recompensas_episodios.append(recompensa_total)
            
            if episodio % verboso == 0:
                barra_progreso.set_postfix({
                    "reward": f"{recompensa_total:.2f}",
                    "epsilon": f"{self.epsilon:.4f}"
                })

        self.garda_q()

class DiferenciaTemporal(AlgoritmoAprendizaxeReforzo):
    def __init__(self, 
                 gamma: float = 0.99,
                 epsilon: float = 0.1,
                 epsilon_min: float = 0.0001,
                 alpha: float = 0.1,
                 beta: float = 0.6,
                 discretizacion_estado: int | list[int] = 20,
                 discretizacion_accion: int = 20,
                 inicializacion_informada: bool = False,
                 perturbacion: bool = False,
                 determinista: bool = False,
                 alpha_decae: bool = True,
                 epsilon_decae: bool = True):

        super().__init__(gamma=gamma, 
                         epsilon=epsilon, 
                         epsilon_min=epsilon_min,
                         alpha=alpha,
                         discretizacion_estado=discretizacion_estado,
                         discretizacion_accion=discretizacion_accion,
                         inicializacion_informada=inicializacion_informada,
                         perturbacion=perturbacion,
                         determinista=determinista,
                         epsilon_decae=epsilon_decae)

        self.alpha_0 = alpha
        self.beta = beta
        self.alpha_decae = alpha_decae

    def seguinte_q(self, estado_seguinte, accion_seguinte):
        pass

    def actualiza_q(self, estado, accion, recompensa, estado_seguinte, accion_seguinte):
        valor_actual = self.q[*estado, accion] 
        seguinte_q = self.seguinte_q(estado_seguinte, accion_seguinte)
        DT_obxectivo = recompensa + self.gamma * seguinte_q
        DT_error = DT_obxectivo - valor_actual

        self.q[*estado, accion] = valor_actual + self.alpha * DT_error

    def actualiza_alpha(self, paso):
        self.alpha = self.alpha_0/((1+paso)**self.beta)

    def adestra(self, num_episodios, num_maximo_pasos, verboso):
        self.tasa_decaemento = (self.epsilon_0 - self.epsilon_min) / num_episodios 


        barra_progreso = tqdm(range(num_episodios), desc=f"{self.__class__.__name__} Training")
        for episodio in barra_progreso:

            if self.epsilon_decae:
                self.actualiza_epsilon(episodio)

            if self.alpha_decae:
                self.actualiza_alpha(episodio)

            estado, _ = self.env.reset()
            historial_episodio = []
            recompensa_total = 0
            estado_dis = self.discretiza_estado(estado)

            for paso in range(num_maximo_pasos):

                accion_dis = self.politica(estado_dis)
                accion = self.continua_accion(accion_dis)
                estado_seguinte, recompensa, feito, truncado, _ = self.env.step(accion)
                recompensa_total += recompensa
                estado_seguinte_dis = self.discretiza_estado(estado_seguinte)
                accion_seguinte_dis = self.politica(estado_seguinte_dis)
                self.actualiza_q(estado_dis, accion_dis, recompensa, estado_seguinte_dis, accion_seguinte_dis)
                estado_dis = estado_seguinte_dis 

                if feito or truncado:
                    break

            self.recompensas_episodios.append(recompensa_total)

            if episodio % verboso == 0:
                barra_progreso.set_postfix({
                    "reward": f"{recompensa_total:.2f}",
                    "epsilon": f"{self.epsilon:.4f}",
                    "alpha": f"{self.alpha:.4f}"
                })

        self.garda_q()

class Sarsa(DiferenciaTemporal):
    def seguinte_q(self, estado_seguinte, accion_seguinte):
        return self.q[*estado_seguinte, accion_seguinte] 

class Q_Aprendizaxe(DiferenciaTemporal):
    def seguinte_q(self, estado_seguinte, accion_seguinte):
        return np.max(self.q[*estado_seguinte])
