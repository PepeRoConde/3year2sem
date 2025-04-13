#!/bin/bash

# ex1
# no aumento no pretrained
python main.py   --batch_size 256 --num_epochs 1000 --patience 30 --lr_patience 10 --learning_rate 0.0001 --l2_lambda 0.000001 --mlp_head --figure_path ultimas_figuras "$@" >> ejecucion.txt
# aumento no pretrained
python main.py --data_augmentation  --batch_size 256 --num_epochs 1000 --patience 30 --lr_patience 10 --learning_rate 0.0001 --l2_lambda 0.000001 --mlp_head  --figure_path ultimas_figuras "$@" >> ejecucion.txt
# no aumento pretrained
python main.py --figure_path ultimas_figuras --save --model_save_path ex1params  --pretrained  --batch_size 256 --num_epochs 1000 --patience 30 --lr_patience 10 --learning_rate 0.0001 --l2_lambda 0.000001 --mlp_head "$@" >> ejecucion.txt
# aumento pretrained 
python main.py --figure_path ultimas_figuras --save --model_save_path ex1paramsDA --data_augmentation --pretrained  --batch_size 256 --num_epochs 1000 --patience 30 --lr_patience 10 --learning_rate 0.0001 --l2_lambda 0.000001 --mlp_head "$@" >> ejecucion.txt


# ex2
# no aumento no pretrained
python main.py --figure_path ultimas_figuras --docked --batch_size 256 --num_epochs 1000 --patience 30 --lr_patience 10 --learning_rate 0.0001 --l2_lambda 0.000001 --mlp_head "$@" >> ejecucion.txt
# aumento no pretrained
python main.py --figure_path ultimas_figuras --data_augmentation  --docked  --batch_size 256 --num_epochs 1000 --patience 30 --lr_patience 10 --learning_rate 0.0001 --l2_lambda 0.000001 --mlp_head "$@" >> ejecucion.txt
# no aumento pretrained
python main.py --figure_path ultimas_figuras --save --model_save_path ex2params --model_path ex1params --load_model --pretrained --docked --batch_size 256 --num_epochs 1000 --patience 30 --lr_patience 10 --learning_rate 0.0001 --l2_lambda 0.000001 --mlp_head "$@" >> ejecucion.txt
# aumento pretrained 
python main.py --figure_path ultimas_figuras --save --model_save_path ex2paramsDA --model_path ex1params --data_augmentation --load_model --pretrained --docked --batch_size 256 --num_epochs 1000 --patience 30 --lr_patience 10 --learning_rate 0.0001 --l2_lambda 0.000001 --mlp_head "$@" >> ejecucion.txt
