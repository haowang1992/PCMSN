import os
import argparse

parser = argparse.ArgumentParser('baseline loss weight grid search')
parser.add_argument('--dataset', type=str)
parser.add_argument('--test', action='store_true', default=False, help='test')
opt = parser.parse_args()

if opt.dataset == 'Sketchy':
    if not opt.test:
        cmd = f'PYTHONPATH=`pwd` /home/xxx/anaconda3/bin/python train_tip.py --dataset Sketchy_extended ' \
            f'--dim_out 64 --semantic_models word2vec-google-news hieremb-jcn ' \
            f'--dataset_root ./ZS-SBIR ' \
            f'--epochs 30 --early_stop 10 --lr 0.0001 --gpu_id 0 --seed 0 ' \
            f'--lambda_gen_adv {1.0} --lambda_ret_cls {0.4} --lambda_mm_euc {0.4} --lambda_rec {1.0} --drop {0.5}'
        os.system(cmd)
    else:
        cmd = f'PYTHONPATH=`pwd` /home/xxx/anaconda3/bin/python train_tip.py --dataset Sketchy_extended ' \
              f'--dim_out 64 --semantic_models word2vec-google-news hieremb-jcn ' \
              f'--dataset_root ./ZS-SBIR --test ' \
              f'--epochs 30 --early_stop 10 --lr 0.0001 --gpu_id 0 --seed 0 ' \
              f'--lambda_gen_adv {1.0} --lambda_ret_cls {0.4} --lambda_mm_euc {0.4} --lambda_rec {1.0} --drop {0.5}'
        os.system(cmd)

elif opt.dataset == 'TU-Berlin':
    if not opt.test:
        cmd = f'PYTHONPATH=`pwd` /home/xxx/anaconda3/bin/python train_tip.py --dataset TU-Berlin ' \
            f'--dim_out 64 --semantic_models hieremb-path word2vec-google-news hieremb-path ' \
            f'--dataset_root ./ZS-SBIR ' \
            f'--epochs 30 --early_stop 10 --lr 0.0001 --gpu_id 0 --seed 0 ' \
            f'--lambda_gen_adv {1.0} --lambda_ret_cls {0.1} --lambda_mm_euc {0.4} --lambda_rec {0.5} --drop {0.5}'
        os.system(cmd)
    else:
        cmd = f'PYTHONPATH=`pwd` /home/xxx/anaconda3/bin/python train_tip.py --dataset TU-Berlin ' \
              f'--dim_out 64 --semantic_models hieremb-path word2vec-google-news hieremb-path ' \
              f'--dataset_root ./ZS-SBIR ' \
              f'--epochs 30 --early_stop 10 --lr 0.0001 --gpu_id 0 --seed 0 ' \
              f'--lambda_gen_adv {1.0} --lambda_ret_cls {0.1} --lambda_mm_euc {0.4} --lambda_rec {0.5} --drop {0.5}'
        os.system(cmd)
