python main.py   --batch_size 24 --num_epochs 1 --patience 8 --lr_patience 4 --learning_rate 0.0001 --l2_lambda 0.000001 --mlp_head
python main.py --data_augmentation  --batch_size 24 --num_epochs 1 --patience 8 --lr_patience 4 --learning_rate 0.0001 --l2_lambda 0.000001 --mlp_head
python main.py  --pretrained  --batch_size 24 --num_epochs 1 --patience 8 --lr_patience 4 --learning_rate 0.0001 --l2_lambda 0.000001 --mlp_head
python main.py --data_augmentation --pretrained --batch_size 24 --num_epochs 1 --patience 8 --lr_patience 4 --learning_rate 0.0001 --l2_lambda 0.000001 --mlp_head
python main.py  --docked --batch_size 24 --num_epochs 1 --patience 8 --lr_patience 4 --learning_rate 0.0001 --l2_lambda 0.000001 --mlp_head
python main.py --data_augmentation  --docked --batch_size 24 --num_epochs 1 --patience 8 --lr_patience 4 --learning_rate 0.0001 --l2_lambda 0.000001 --mlp_head
python main.py  --pretrained --docked --batch_size 24 --num_epochs 1 --patience 8 --lr_patience 4 --learning_rate 0.0001 --l2_lambda 0.000001 --mlp_head
python main.py --data_augmentation --pretrained --docked --batch_size 24 --num_epochs 1 --patience 8 --lr_patience 4 --learning_rate 0.0001 --l2_lambda 0.000001 --mlp_head
