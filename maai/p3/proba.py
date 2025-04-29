from Algoritmos import MonteCarlo, Sarsa, Q_Aprendizaxe
import numpy as np

mc = MonteCarlo(discretizacion_estado=15)
mc.mostra_q()

mc.q = np.load('MonteCarlo__g_0%99_e_0%3.npy')
mc.mostra_q()

mc.proba()