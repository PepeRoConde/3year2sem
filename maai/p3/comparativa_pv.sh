#!/bin/bash

DISCRETIZACION_ESTADO=30
DISCRETIZACION_ACCION=30
GAMMA=0.99
VERBOSO=200
NUM_EPISODIOS=500
CARTAFOL=comparativa

: > resultados.txt

# Monte Carlo (primera visita y no), sen perturbacion
python main.py --algoritmos mc --num_episodios $NUM_EPISODIOS --verboso $VERBOSO \
  --discretizacion_estado $DISCRETIZACION_ESTADO --discretizacion_accion $DISCRETIZACION_ACCION \
  --gamma $GAMMA --alpha 0.6 --alpha_min 0.0005 --alpha_decae \
  --epsilon 0.5 --epsilon_min 0.0005 --epsilon_decae \
  --inicializacion_informada --cartafol_figuras $CARTAFOL >> resultados.txt 

python main.py --algoritmos mc --num_episodios $NUM_EPISODIOS --verboso $VERBOSO \
  --discretizacion_estado $DISCRETIZACION_ESTADO --discretizacion_accion $DISCRETIZACION_ACCION \
  --gamma $GAMMA --alpha 0.6 --alpha_min 0.0005 --alpha_decae \
  --epsilon 0.5 --epsilon_min 0.0005 --epsilon_decae \
  --inicializacion_informada --primeira_visita --cartafol_figuras $CARTAFOL >> resultados.txt 

