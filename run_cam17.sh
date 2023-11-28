#!/bin/bash



#ours
CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train_cam17 --data_dir=./domainbed/data/ --steps 5900 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT --output_dir \
results/exp --hparams '{"lr": 5e-7, "lr_classifier": 5e-7,"batch_size":130,"wd_classifier": 1e-4, "prompt_dim":4}' --exp 'epvt_5e-7_5e-7_1e-4_sne_exp'  --clustering True \
 --use_domain_labels False --loss-disc-weight --num-clustering 4

#
#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train_cam17_erm --data_dir=./domainbed/data/ --steps 1501 --dataset WILDSCamelyon --test_env 0 --algorithm ERM \
#--output_dir results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier":1e-5}'
#
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_cam17_erm --data_dir=./domainbed/data/ --steps 1501 --dataset WILDSCamelyon --test_env 0 --algorithm CORAL \
#--output_dir results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier":1e-5}'

#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_cam17_erm --data_dir=./domainbed/data/ --steps 1501 --dataset WILDSCamelyon --test_env 0 --algorithm DoPrompt_group_decompose \
#--output_dir results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier":1e-5}' --exp 'epvt_baseline'


#local3 3
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_cam17_erm --data_dir=./domainbed/data/ --steps 301 --dataset WILDSCamelyon --test_env 4 --algorithm ERM \
#--output_dir results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier":1e-5}'
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_cam17_erm --data_dir=./domainbed/data/ --steps 301 --dataset WILDSCamelyon --test_env 4 --algorithm IRM \
#--output_dir results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier":1e-5}'
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_cam17_erm --data_dir=./domainbed/data/ --steps 301 --dataset WILDSCamelyon --test_env 4 --algorithm SelfReg \
#--output_dir results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier":1e-5}'