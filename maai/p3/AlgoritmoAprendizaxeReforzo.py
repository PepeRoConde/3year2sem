from abc import ABC, abstractmethod
import gymnasium as gym
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np
from scipy.ndimage import gaussian_filter1d

class AlgoritmoAprendizaxeReforzo(ABC):
    def __init__(self, 
                    gamma: float = 0.99,
                    epsilon: float = 0.1,
                    epsilon_min: float = 0.001,
                    alpha: float = 0.1,
                    discretizacion_estado: int | list[int] = 20, # jorge, esto e apartires de python 3.10 se non te vai e por iso
                    discretizacion_accion: int = 20,
                    inicializacion_informada: bool = False,
                    perturbacion: bool = False,
                    determinista: bool = False,
                    epsilon_decae: bool = False):
        
        if isinstance(discretizacion_estado, int):
            discretizacion_estado = [discretizacion_estado]*3 # preciso que sexa unha lista de 3 elementos (coseno, seno e velocidade angular), pero permito que se especifique como enteiro
        elif len(discretizacion_estado) != 3:
            print('A discretizacion_estado debe ser unha lista de tres elementos (coseno, seno, velocidade angular)')
    
        self.env = gym.make('Pendulum-v1') 
        self.gamma = gamma
        self.epsilon, self.epsilon_0 = epsilon, epsilon # epsilon_0 y alpha_0 (definido na clase DT) almacenaran o valor inicial en caso de que decaiga
        self.epsilon_min = epsilon_min
        self.alpha = alpha
        self.discretizacion_estado = discretizacion_estado
        self.discretizacion_accion = discretizacion_accion
        self.epsilon_decae = epsilon_decae
        self.perturbacion = perturbacion
        self.determinista = determinista
    
        # estos arreglos almacenaran flotantes que aproximen a realidade. veñen a ser unha lookup table para ter metodos tabulares con realidades continuas 
        self.estados = self.inicializa_estados()
        self.accions = self.inicializa_accions() # todas as accions estan permitidas en todos os estados
        self.q = self.inicializa_q(inicializacion_informada)
        self.recompensas_episodios = []
    
    def inicializa_estados(self):
        cota_inf, cota_sup = [-1.0, -1.0, -8.0], [1.0, 1.0, 8.0] # esta hardcodeado porque e relativo a natureza do entorno
        return np.array([np.linspace(inf, sup, cantos) for inf, sup, cantos in zip(cota_inf, cota_sup, self.discretizacion_estado)])
    
    def inicializa_accions(self):
        cota_inf = self.env.action_space.low[0]
        cota_sup = self.env.action_space.high[0]
        return np.linspace(cota_inf, cota_sup, self.discretizacion_accion)
 

    def inicializa_q(self, inicializacion_informada):
        if inicializacion_informada:
            q = np.zeros(tuple([*self.discretizacion_estado, self.discretizacion_accion]))
            
            for i in range(self.discretizacion_estado[0]):  # coseno
                for j in range(self.discretizacion_estado[1]):  # seno
                    for k in range(self.discretizacion_estado[2]):  # velocidad angular
                        for a in range(self.discretizacion_accion):  # acción
                            
                            cos_theta = self.estados[0][i]
                            sin_theta = self.estados[1][j]
                            vel_ang = self.estados[2][k]
                            accion = self.accions[a]
                            
                            posicion_bonus = -5.0 * (cos_theta + 1.0)  # Highest when cos(θ) = -1 (upright)
                            posicion_bonus = 0
                            velocidad_penalizacion = -0.1 * abs(vel_ang) 
                            accion_bonus = -0.2 * vel_ang * accion  # se oponse en signo, sera positivo 
                            
                            q[i,j,k,a] = posicion_bonus + velocidad_penalizacion + accion_bonus
            
            return q
        else:
            return np.zeros(tuple([*self.discretizacion_estado, self.discretizacion_accion]))

    def garda_q(self):
        ruta = self.ruta_segura_gardado('parametros', self.nome_figura(), extension='.npy')
        np.save(ruta, self.q)

    def carga_q(self,nome_arquivo):
        self.q = np.load(nome_arquivo)
    
    def discretiza_estado(self, estado):
        # pasar de flotantes a indices da taboa
        indices = []
        for cantos, valor, opcions in zip(self.discretizacion_estado, estado, self.estados):
            indice = np.digitize(valor, opcions) - 1 # o menos un e porque digitize solo devolve 0 se esta fora (que raro)
            indice = np.clip(indice, 0, cantos - 1) # agora, se cadra fora metemolo dentro
            indices.append(indice)
        return tuple(indices) # non e que non nos gustara usar namedtuple pero como sabemos que indice e cada cousa, non vimos a necesidade
        
    def discretiza_accion(self, accion):
        # pasar de flotantes a indices da taboa
        if True:
            indice = np.digitize(accion[0], self.accions) - 1
        return int(np.clip(indice, 0, len(self.accions) - 1))
    
    def continua_accion(self, accion):
        # pasar de indices da taboa a flotantes
        return np.array([self.accions[accion]])
    
    # NOTE: estas duas funcions son inversas (unha da outra) baixo esta representacion tabular do mundo. non o son, estrictamente falando, porque o discretizar perdese informacion
    
    def politica(self, estado):
        # chamada a politica epsilon avara para que dado un estado nos diga a seguinte accion. 
        # NOTE: como podese ver, non temos estructura de datos exlicita definida para a politica (nin para v) se non que, como a politica inferencia a apartires de Q, e cambiar a politica ten que ver sempre con cambiar Q, simplemente usamos Q.  xeito e computacionalmente mais eficiente
        if self.perturbacion and not self.determinista:
            if np.random.randint(20) == 0: # 1 entre 20 e un 5% de probablilidade
                if np.random.randint(1) == 0: return 0 # 50% de unha perturbacion ou outra
                else: return -1
        if (np.random.random() > self.epsilon) or self.determinista:
            return np.argmax(self.q[estado]) # explotacion
        else:
            return np.random.randint(0, len(self.accions)) # exploracion
    
    def nome_figura(self):
        try: # para os DT
            return f"{self.__class__.__name__}__g_{self.gamma}_e_{self.epsilon_0}_alpha_{self.alpha_0}".replace(".", "%")
        except: # para MC
            return f"{self.__class__.__name__}__g_{self.gamma}_e_{self.epsilon_0}".replace(".", "%")
    
    
    def ruta_segura_gardado(self, cartafol, nome_base, extension=".png"):
        if cartafol:
            os.makedirs(cartafol, exist_ok=True)
            nome_ficheiro = nome_base + extension
            ruta_completa = os.path.join(cartafol, nome_ficheiro)
    
            if os.path.exists(ruta_completa):
                agora = datetime.now().strftime("_%Y%m%d_%H%M%S")
                nome_ficheiro = nome_base + agora + extension
                ruta_completa = os.path.join(cartafol, nome_ficheiro)
        else:
            nome_ficheiro = nome_base + extension
            ruta_completa = nome_ficheiro
    
        return ruta_completa
    
    
    def curva_aprendizaxe(self, cartafol=None, mostra=True, sigma=None):
        if not sigma: sigma = int(len(self.recompensas_episodios)/50)
        
        plt.figure(figsize=(10, 6))
        plt.plot(gaussian_filter1d(self.recompensas_episodios,sigma))
        plt.xlabel("Episodios")
        plt.ylabel("Recompensa total")
        plt.grid(True)
        if mostra:
            plt.show()
        elif cartafol:
            nome_base = 'CA_' + self.nome_figura()
            ruta = self.ruta_segura_gardado(cartafol, nome_base)
            plt.savefig(ruta)
        
    
    def mostra_q(self, cartafol=None, mostra=True):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
        cos_indices = np.arange(self.discretizacion_estado[0])
        sin_indices = np.arange(self.discretizacion_estado[1])
        X, Y = np.meshgrid(cos_indices, sin_indices)
    
        torque_indices = [0, self.discretizacion_estado[2] // 2, -1]
        titulos = ['torque = -2', 'torque = 0', 'torque = 2']
    
        for ax, indice, titulo in zip(axes, torque_indices, titulos):
            Z = np.zeros_like(X, dtype=float)
            
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = np.argmax(self.q[X[i, j], Y[i, j], indice])
            
            pcm = ax.pcolormesh(X, Y, Z, cmap='viridis', shading='auto')
            fig.colorbar(pcm, ax=ax, label='Action')
            ax.set_title(f"Value Function ({titulo})", fontsize=12)
            ax.set_xlabel("cos(θ)", fontsize=10)
            ax.set_ylabel("sin(θ) ", fontsize=10)
            ax.grid(True, alpha=0.3)
    
        plt.tight_layout()
    
        if mostra:
            plt.show()
        elif cartafol:
            nome_base = 'Q_' + self.nome_figura()
            ruta = self.ruta_segura_gardado(cartafol, nome_base)
            plt.savefig(ruta)
    
    
    def actualiza_epsilon(self, paso):
        self.epsilon = max(self.epsilon_min, self.epsilon_0 - self.tasa_decaemento * paso)
    
    def proba(self, num_episodios=1, render=True):
        
        if render:
            self.env = gym.make('Pendulum-v1', render_mode='human') 
    
        antes_determinista = self.determinista
        self.determinista = True
    
        recompensas_totales = []
        for episodio in range(num_episodios):
            estado, _ = self.env.reset()
            recompensas_episodios = 0
            rematado, truncado = False, False
            
            while not (rematado or truncado):
                estado_discreto = self.discretiza_estado(estado)
                accion_discreta = self.politica(estado_discreto)
                accion = self.continua_accion(accion_discreta)
    
                estado, recompensa, rematado, truncado, _ = self.env.step(accion)
                recompensas_episodios += recompensa
    
            recompensas_totales.append(recompensas_episodios)
            print(f'Episodio de proba {episodio+1}/{num_episodios}. Recompensa: {recompensas_episodios:.2f}')
        recompensa_media = np.mean(recompensas_totales)
        print(f'\nRecompensa media {recompensa_media:.2f}\n\n')
    
        self.determinista = antes_determinista
    
        self.env.close()

        return recompensa_media
    
    @abstractmethod
    def adestra(self, num_episodios=1000, num_maximo_pasos=1000, verboso=10):
        pass
    
    @abstractmethod
    def actualiza_q(self, *args):
        pass
