#!/bin/bash
python main.py   --batch_size 256 --num_epochs 500 --patience 20 --lr_patience 8 --learning_rate 0.0001 --l2_lambda 0.000001 --mlp_head "$@" >> ejecucion.txt
python main.py --data_augmentation  --batch_size 256 --num_epochs 500 --patience 20 --lr_patience 8 --learning_rate 0.0001 --l2_lambda 0.000001 --mlp_head "$@" >> ejecucion.txt
python main.py --pretrained  --batch_size 256 --num_epochs 500 --patience 20 --lr_patience 8 --learning_rate 0.0001 --l2_lambda 0.000001 --mlp_head "$@" >> ejecucion.txt
python main.py --data_augmentation --pretrained --save --batch_size 256 --num_epochs 500 --patience 20 --lr_patience 8 --learning_rate 0.0001 --l2_lambda 0.000001 --mlp_head "$@" >> ejecucion.txt
python main.py --docked --batch_size 256 --num_epochs 500 --patience 20 --lr_patience 8 --learning_rate 0.0001 --l2_lambda 0.000001 --mlp_head "$@" >> ejecucion.txt
python main.py --data_augmentation  --docked  --batch_size 256 --num_epochs 500 --patience 20 --lr_patience 8 --learning_rate 0.0001 --l2_lambda 0.000001 --mlp_head "$@" >> ejecucion.txt
python main.py  --docked --load_model --pretrained --batch_size 256 --num_epochs 500 --patience 20 --lr_patience 8 --learning_rate 0.0001 --l2_lambda 0.000001 --mlp_head "$@" >> ejecucion.txt
python main.py --data_augmentation --load_model --pretrained --docked --batch_size 256 --num_epochs 500 --patience 20 --lr_patience 8 --learning_rate 0.0001 --l2_lambda 0.000001 --mlp_head "$@" >> ejecucion.txt
