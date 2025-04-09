#!/bin/bash
python main.py --save --model_save_path ex1params --data_augmentation --pretrained  --batch_size 256 --num_epochs 500 --patience 20 --lr_patience 8 --learning_rate 0.0001 --l2_lambda 0.000001 --mlp_head "$@" >> ejecucion.txt
python main.py --save --model_save_path ex2params --model_path ex1params--data_augmentation --load_model --pretrained --docked --batch_size 256 --num_epochs 500 --patience 20 --lr_patience 8 --learning_rate 0.0001 --l2_lambda 0.000001 --mlp_head "$@" >> ejecucion.txt
