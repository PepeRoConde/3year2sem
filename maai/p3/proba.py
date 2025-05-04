from Algoritmos import MonteCarlo, Sarsa, Q_Aprendizaxe, SarsaPromedio
import numpy as np

# variables gloabais, cambiar aquí
num_episodios, render = 1000, False



print('<<mc>>')
mc = MonteCarlo(discretizacion_estado=30,discretizacion_accion=30)
mc.q = np.load('parametros/MonteCarlo__de_30_pert_True_da_30_g_0%99_e_0%5_pv_True.npy')

mc.proba(num_episodios=num_episodios, render=render)




print('<<sarsa>>')
sarsa = Sarsa(discretizacion_estado=30,discretizacion_accion=30)
sarsa.q = np.load('parametros/Sarsa__de_30_pert_True_da_30_g_0%99_e_0%5_alpha_0%6.npy')

sarsa.proba(num_episodios=num_episodios, render=render)



print('<<sarsa promedio>>')
sarsa_p = SarsaPromedio(discretizacion_estado=30,discretizacion_accion=30)
sarsa_p.q = np.load('parametros/SarsaPromedio__de_30_pert_True_da_30_g_0%99_e_0%5_alpha_0%6.npy')

sarsa_p.proba(num_episodios=num_episodios, render=render)


print('<<aprendizaxeQ>>')
q = Q_Aprendizaxe(discretizacion_estado=30,discretizacion_accion=30)
q.q = np.load('parametros/Q_Aprendizaxe__de_30_pert_True_da_30_g_0%99_e_0%5_alpha_0%6.npy')

q.proba(num_episodios=num_episodios, render=render)
