from Algoritmos import MonteCarlo, Sarsa, Q_Aprendizaxe
import argparse 

def parsea_argumentos():
    parser = argparse.ArgumentParser(description='Uso de aprendizaxe por reforzo para o problema do pendulo invertido')
    
    parser.add_argument('--num_episodios', type=int, default=5000)
    parser.add_argument('--num_episodios_proba', type=int, default=5)
    parser.add_argument('--num_maximo_pasos', type=int, default=1000)
    parser.add_argument('--verboso', type=int, default=50, help='Cada cantos episodios imprimese por pantalla.')
    parser.add_argument('--discretizacion_estado', type=int, default=20, help='Canto redondea a discretizacion tabular. Máis numero é menos redondeo.')
    parser.add_argument('--discretizacion_accion', type=int, default=20, help='En cantas celdas dividese o espacio de actuación')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--epsilon_min', type=float, default=0.001)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.6)
    parser.add_argument('--cartafol_figuras', type=str, default='figuras',help='Onde gardarasen as figuras. Non ten efecto se usase --mostra')
    parser.add_argument('--inicializacion_informada', action='store_true') # ainda non o implentei
    parser.add_argument('--primeira_visita', action='store_true', help='Se especificase, o algoritmo de MonteCarlo soamente tera en conta a primeira visita dun estado nun episodio')
    parser.add_argument('--alpha_decae', action='store_true')
    parser.add_argument('--epsilon_decae', action='store_true')
    parser.add_argument('--mostra', action='store_true', help='Se especificase as figuras saldran por pantalla, en vez de gardarse')
    parser.add_argument('--render', action='store_true', help='Se especificase se vera a tempo real nas probas (non mentras adestra)')
    parser.add_argument('--perturbacion', action='store_true', help='Se especificase aplicaranse perturbacions como definidas no enunciado da practica')
    parser.add_argument('--determinista', action='store_true', help='Se especificase, nin sera epsilon-politica nin habera perturbacions')
    parser.add_argument('--algoritmos',type=list[str], default=['mc','s','q'])

    return parser.parse_args() 

if __name__ == "__main__":
    arg = parsea_argumentos()

    if 'mc' in arg.algoritmos:
        mc = MonteCarlo(gamma=arg.gamma, 
                    epsilon=arg.epsilon,
                    epsilon_min=arg.epsilon_min,
                    epsilon_decae=arg.epsilon_decae,
                    discretizacion_estado=arg.discretizacion_estado,
                    discretizacion_accion=arg.discretizacion_accion,
                    inicializacion_informada=arg.inicializacion_informada,
                    primeira_visita=arg.primeira_visita,
                    perturbacion=arg.perturbacion,
                    determinista=arg.determinista)
        mc.adestra(num_episodios=arg.num_episodios,
                    num_maximo_pasos=arg.num_maximo_pasos,
                    verboso=arg.verboso)
        mc.curva_aprendizaxe(cartafol=arg.cartafol_figuras,mostra=arg.mostra)
        mc.mostra_q(cartafol=arg.cartafol_figuras,mostra=arg.mostra)
        mc.proba(num_episodios=arg.num_episodios_proba,render=arg.render)

    if 's' in arg.algoritmos:
        sarsa = Sarsa(gamma=arg.gamma, 
                    epsilon=arg.epsilon,
                    epsilon_min=arg.epsilon_min,
                    alpha=arg.alpha,
                    alpha_decae=arg.alpha_decae,
                    beta=arg.beta,
                    epsilon_decae=arg.epsilon_decae,
                    discretizacion_estado=arg.discretizacion_estado,
                    discretizacion_accion=arg.discretizacion_accion,
                    inicializacion_informada=arg.inicializacion_informada,
                    perturbacion=arg.perturbacion,
                    determinista=arg.determinista)
        sarsa.adestra(num_episodios=arg.num_episodios,
                    num_maximo_pasos=arg.num_maximo_pasos,
                    verboso=arg.verboso)
        sarsa.curva_aprendizaxe(cartafol=arg.cartafol_figuras,mostra=arg.mostra)
        sarsa.mostra_q(cartafol=arg.cartafol_figuras,mostra=arg.mostra)
        sarsa.proba(num_episodios=arg.num_episodios_proba,render=arg.render)
        
    if 'q' in arg.algoritmos:
        q_aprendizaxe = Q_Aprendizaxe(gamma=arg.gamma, 
                    epsilon=arg.epsilon,
                    epsilon_min=arg.epsilon_min,
                    alpha=arg.alpha,
                    beta=arg.beta,
                    alpha_decae=arg.alpha_decae,
                    epsilon_decae=arg.epsilon_decae,
                    discretizacion_estado=arg.discretizacion_estado,
                    discretizacion_accion=arg.discretizacion_accion,
                    inicializacion_informada=arg.inicializacion_informada,
                    perturbacion=arg.perturbacion,
                    determinista=arg.determinista)
        q_aprendizaxe.adestra(num_episodios=arg.num_episodios,                      
                    num_maximo_pasos=arg.num_maximo_pasos,
                    verboso=arg.verboso)
        q_aprendizaxe.curva_aprendizaxe(cartafol=arg.cartafol_figuras,mostra=arg.mostra)
        q_aprendizaxe.mostra_q(cartafol=arg.cartafol_figuras,mostra=arg.mostra)
        q_aprendizaxe.proba(num_episodios=arg.num_episodios_proba,render=arg.render)
