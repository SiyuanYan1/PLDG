#!/bin/bash
##local3
CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_dr_latent --data_dir=./domainbed/data/ --steps 1000 --dataset Latent_DR_Dataset --test_envs 0 --val_envs 1 \
--algorithm Latent_EPVT --output_dir results/exp --hparams '{"lr":5e-6, "lr_classifier": 5e-5,"batch_size":130,"wd_classifier":1e-5,"prompt_dim":10,"seed":1}' --exp 'latent_dr_eyepacs_p10' --clustering True
 CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train_dr_latent --data_dir=./domainbed/data/ --steps 5697 --dataset Latent_DR_Dataset --test_envs 2 --val_envs 1 \
--algorithm Latent_EPVT --output_dir results/exp --hparams '{"lr":5e-6, "lr_classifier": 5e-5,"batch_size":130,"wd_classifier":1e-5}' --exp 'latent_dr_m2' --clustering True
CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_dr_latent --data_dir=./domainbed/data/ --steps 5697 --dataset Latent_DR_Dataset --test_envs 3 --val_envs 1 \
--algorithm Latent_EPVT --output_dir results/exp --hparams '{"lr":5e-7, "lr_classifier": 5e-6,"batch_size":130,"wd_classifier":1e-5}' --exp 'latent_dr_aptos_5e-7' --clustering True

#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_dr_prompt --data_dir=./domainbed/data/ --steps 1501 --dataset DR --test_env 1 --val_envs 3 --algorithm DoPrompt_group_decompose \
#--output_dir results/exp --hparams '{"lr":5e-7, "lr_classifier": 5e-6,"batch_size":26,"wd_classifier":1e-5,"prompt_dim":10,"seed":1}' --exp 'epvt_m1_5e-7' 
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_dr_prompt --data_dir=./domainbed/data/ --steps 1501 --dataset DR --test_env 1 --val_envs 3 --algorithm DoPrompt_group_decompose \
#--output_dir results/exp --hparams '{"lr":5e-5, "lr_classifier": 5e-4,"batch_size":26,"wd_classifier":1e-5,"prompt_dim":10,"seed":1}' --exp 'epvt_m1_5e-5' 

#test other methods on m1. local3

#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_dr --data_dir=./domainbed/data/ --steps 1001 --dataset DR --test_env 1 --val_envs 3 --algorithm DANN \
#--output_dir results/exp --hparams '{"lr":5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier":1e-5}' --exp 'dann_dr_m1'
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_dr --data_dir=./domainbed/data/ --steps 1001 --dataset DR --test_env 1 --val_envs 3 --algorithm Fishr \
#--output_dir results/exp --hparams '{"lr":5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier":1e-5}' --exp 'fishr_dr_m1'
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_dr_prompt --data_dir=./domainbed/data/ --steps 1001 --dataset DR --test_env 1 --val_envs 3 --algorithm DoPrompt_group_decompose \
#--output_dir results/exp --hparams '{"lr":5e-7, "lr_classifier": 5e-6,"batch_size":26,"wd_classifier":1e-5,"prompt_dim":10,"seed":1}' --exp 'epvt_m1_5e-7' 
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_dr --data_dir=./domainbed/data/ --steps 1001 --dataset DR --test_env 1 --val_envs 3 --algorithm CORAL \
#--output_dir results/exp --hparams '{"lr":5e-7, "lr_classifier": 5e-6,"batch_size":26,"wd_classifier":1e-5}' --exp 'coral_dr_m15e-7'
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_dr --data_dir=./domainbed/data/ --steps 1001 --dataset DR --test_env 1 --val_envs 3 --algorithm CORAL \
#--output_dir results/exp --hparams '{"lr":5e-5, "lr_classifier": 5e-4,"batch_size":26,"wd_classifier":1e-5}' --exp 'coral_dr_m15e-5'
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_dr --data_dir=./domainbed/data/ --steps 1001 --dataset DR --test_env 1 --val_envs 3 --algorithm SelfReg \
#--output_dir results/exp --hparams '{"lr":5e-7, "lr_classifier": 5e-6,"batch_size":26,"wd_classifier":1e-5}' --exp 'selfreg_dr_m1_5e-7'
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_dr --data_dir=./domainbed/data/ --steps 1001 --dataset DR --test_env 1 --val_envs 3 --algorithm SelfReg \
#--output_dir results/exp --hparams '{"lr":5e-5, "lr_classifier": 5e-4,"batch_size":26,"wd_classifier":1e-5}' --exp 'selfreg_dr_m1_5e-5'
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_dr --data_dir=./domainbed/data/ --steps 1501 --dataset DR --test_env 1 --val_envs 3 --algorithm SelfReg \
#--output_dir results/exp --hparams '{"lr":5e-7, "lr_classifier": 5e-6,"batch_size":26,"wd_classifier":1e-5}' --exp 'selfreg_dr_m1_iter1501'

