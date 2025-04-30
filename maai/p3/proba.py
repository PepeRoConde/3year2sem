from Algoritmos import MonteCarlo, Sarsa, Q_Aprendizaxe
import numpy as np

print('<<mc>>')
mc = Sarsa(discretizacion_estado=20,discretizacion_accion=30)
mc.q = np.load('parametros/MonteCarlo__de_30_pert_False_da_30_g_0%99_e_0%5_pv_False.npy')

mc.proba()




#print('<<sarsa>>')
#sarsa = Sarsa(discretizacion_estado=20,discretizacion_accion=30)
#sarsa.q = np.load('parametros/Sarsa__g_0%99_e_0%25_alpha_0%3.npy')

#sarsa.proba()


#print('<<aprendizaxeQ>>')
#q = Q_Aprendizaxe(discretizacion_estado=25,discretizacion_accion=25)
#q.q = np.load('parametros/Q_Aprendizaxe__de_25_da_25_g_0%99_e_0%4_alpha_0%5.npy')

#q.proba()
