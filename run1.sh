#!/bin/bash

#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 5697 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT --output_dir \
#results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":130,"wd_classifier": 1e-5, "prompt_dim":10}' --exp 'epvt_c4_l4_d'  --clustering True \
# --use_domain_labels False --loss-disc-weight --num-clustering 4






#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 5697 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT --output_dir \
#results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":130,"wd_classifier": 1e-4, "prompt_dim":10}' --exp 'epvt_c4_l4_d'  --clustering True \
# --use_domain_labels False --loss-disc-weight --num-clustering 2
#
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 5697 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT --output_dir \
#results/exp --hparams '{"lr": 5e-7, "lr_classifier": 5e-7,"batch_size":130,"wd_classifier": 1e-5, "prompt_dim":10}' --exp 'epvt_c4_l4_d'  --clustering True \
# --use_domain_labels False --loss-disc-weight --num-clustering 2
#
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 5697 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT --output_dir \
#results/exp --hparams '{"lr": 5e-7, "lr_classifier": 5e-6,"batch_size":130,"wd_classifier": 1e-5, "prompt_dim":4}' --exp 'epvt_c4_l4_d'  --clustering True \
# --use_domain_labels False --loss-disc-weight --num-clustering 3
#
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.test_prompt10 --model_name 'epvt_c4_l4_d.pkl'

CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 5697 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT --output_dir \
results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":130,"wd_classifier": 1e-2, "prompt_dim":4}' --exp 'l2_v1'  --clustering True \
 --use_domain_labels False --loss-disc-weight --num-clustering 4

CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 5697 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT --output_dir \
results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":130,"wd_classifier": 1e-2, "prompt_dim":4}' --exp 'l2_v2'  --clustering True \
 --use_domain_labels False --loss-disc-weight --num-clustering 2
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.test_prompt --model_name 'l2.pkl'



