#python train.py --dataset ARC --num_epochs 10 --lr 5e-6
python test.py --dataset ARC --batch_size 256 --temperature 0.7 --max_new_tokens 300

#python train.py --dataset ECQA --num_epochs 10 --lr 5e-6
python test.py --dataset ECQA --batch_size 256 --temperature 0.7 --max_new_tokens 300

#python train.py --dataset GSM8K --num_epochs 10 --lr 5e-6
python test.py --dataset GSM8K --batch_size 256 --temperature 0.7 --max_new_tokens 300

#python train.py --dataset MATH --num_epochs 10 --lr 5e-6
python test.py --dataset MATH --batch_size 256 --temperature 0.7 --max_new_tokens 300
