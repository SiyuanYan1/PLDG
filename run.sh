#!/bin/bash


#training
#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train --data_dir=./domainbed/data/ --steps 1501 --dataset SKIN --test_env 0 --algorithm DoPrompt_group_decompose --output_dir \
#results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier": 1e-5, "prompt_dim":10}' --exp 'prompt_mixup_group_P10_5e-5' 

#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train --data_dir=./domainbed/data/ --steps 1501 --dataset SKIN --test_env 0 --algorithm DoPromptMixup --output_dir \
#results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier": 1e-5, "prompt_dim":10}' --exp 'prompt_mixup' 




#
#
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train --data_dir=./domainbed/data/ --steps 1501 --dataset SKIN --test_env 0 --algorithm DoPrompt --output_dir \
#results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier": 1e-5, "prompt_dim":10}' --exp 'prompt' 
#
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_erm --data_dir=./domainbed/data/ --steps 1501 --dataset SKIN --test_env 0 --algorithm DoPrompt_equal --output_dir \
#results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier": 1e-5, "prompt_dim":10}' --exp 'prompt_mixupequal' 

#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train_erm --data_dir=./domainbed/data/ --steps 1501 --dataset SKIN --test_env 0 --algorithm ERM \
#--output_dir results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier":1e-5}' --exp 'erm_baseline'
#
#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train_erm --data_dir=./domainbed/data/ --steps 1501 --dataset SKIN --test_env 0 --algorithm ERM \
#--output_dir results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier":1e-5}' --exp 'erm_baseline_5e-6' 

#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_erm --data_dir=./domainbed/data/ --steps 1501 --dataset SKIN --test_env 0 --algorithm Latent_ERM \
#--output_dir results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier":1e-5}' --exp 'erm_baseline_5e-6_hook' 

#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.test_prompt10 --model_name 'prompt_mixup_group_P10_5e-6.pkl'
#testing
#python -m domainbed.scripts.test_erm --model_name 'erm_baseline.pkl'
#python -m domainbed.scripts.test_erm --model_name 'erm_baseline_1e-5.pkl'
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.test_prompt10 --model_name 'prompt_mixup_group_P10_5e-5.pkl'
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.test_prompt10 --model_name 'prompt_mixup.pkl'
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.test_prompt --model_name 'epvt_latent.pkl'


###multiple runs'erm_baseline.pkl'

#python -m domainbed.scripts.test_erm --model_name 'selfreg.pkl' --model 'SelfReg'

#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train --data_dir=./domainbed/data/ --steps 1501 --dataset SKIN --test_env 0 --algorithm DoPrompt_group_decompose --output_dir \
#results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier": 1e-5, "prompt_dim":10,"seed":1}' --exp 'final_seed1' 
#
#!!!!
#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train --data_dir=./domainbed/data/ --steps 1501 --dataset SKIN --test_env 0 --algorithm DoPrompt_group_decompose --output_dir \
#results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier": 1e-5, "prompt_dim":10}' --exp 'final_seed2' 

#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train --data_dir=./domainbed/data/ --steps 1501 --dataset SKIN --test_env 0 --algorithm DoPrompt_group_decompose --output_dir \
#results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier": 1e-5, "prompt_dim":10}' --exp 'epvt_base' 

#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 1501 --dataset SKIN --test_env 0 --algorithm DoPrompt_group_decompose --output_dir \
#results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier": 1e-5, "prompt_dim":10}' --exp 'epvt_latent' 
#
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 5697 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT --output_dir \
#results/exp --hparams '{"lr": 1e-5, "lr_classifier": 3e-4,"batch_size":130,"wd_classifier": 1e-5, "prompt_dim":10}' --exp 'epvt_latent_1e-5'  --clustering False --use_domain_labels False

#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 5697 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT --output_dir \
#results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":130,"wd_classifier": 1e-5, "prompt_dim":10}' --exp 'epvt_latent_c'  --clustering True --use_domain_labels False --loss-disc-weight
#
#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 5697 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT --output_dir \
#results/exp --hparams '{"lr": 5e-5, "lr_classifier": 5e-4,"batch_size":130,"wd_classifier": 1e-5, "prompt_dim":10}' --exp 'epvt_latent_5e-5'  --clustering True --use_domain_labels False

#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.test_prompt10 --model_name 'final_seed1.pkl'
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.test_prompt10 --model_name 'final_seed2.pkl'
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.test_prompt10 --model_name 'prompt5e-6.pkl'
#baselines

#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.test_prompt10 --model_name 'epvt_latent_c.pkl'
#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train_erm --data_dir=./domainbed/data/ --steps 1501 --dataset SKIN --test_env 0 --algorithm MMD \
#--output_dir results/exp --hparams '{"lr":5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier":1e-5}' --exp 'mmd' 
#
#
#
#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train_erm --data_dir=./domainbed/data/ --steps 2501 --dataset SKIN --test_env 0 --algorithm CAD \
#--output_dir results/exp --hparams '{"lr":1e-5, "lr_classifier": 3e-4,"batch_size":26,"wd_classifier":1e-2}' --exp 'cad' 
#
#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train_erm --data_dir=./domainbed/data/ --steps 1501 --dataset SKIN --test_env 0 --algorithm MLDG \
#--output_dir results/exp --hparams '{"lr":5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier":1e-5}' --exp 'mldg' 
#
#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train_erm --data_dir=./domainbed/data/ --steps 1501 --dataset SKIN --test_env 0 --algorithm DANN \
#--output_dir results/exp --hparams '{"lr":5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier":1e-5}' --exp 'dann' 
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_erm --data_dir=./domainbed/data/ --steps 1501 --dataset SKIN --test_env 0 --algorithm IRM \
#--output_dir results/exp --hparams '{"lr":5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier":1e-5}' --exp 'irm' 
#
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_erm --data_dir=./domainbed/data/ --steps 1501 --dataset SKIN --test_env 0 --algorithm CORAL \
#--output_dir results/exp --hparams '{"lr":5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier":1e-5}' --exp 'coral' 
#
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_erm --data_dir=./domainbed/data/ --steps 1501 --dataset SKIN --test_env 0 --algorithm SelfReg \
#--output_dir results/exp --hparams '{"lr":5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier":1e-5}' --exp 'selfreg' 

#
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_erm --data_dir=./domainbed/data/ --steps 1501 --dataset SKIN --test_env 0 --algorithm SagNet \
#--output_dir results/exp --hparams '{"lr":5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier":1e-5}' --exp 'sagnet' 

###
#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train --data_dir=./domainbed/data/ --steps 1501 --dataset SKIN --test_env 0 --algorithm DoPrompt --output_dir \
#results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier": 1e-5}' --exp 'prompt5e-6' 

#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 5697 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT --output_dir \
#results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":130,"wd_classifier": 1e-5, "prompt_dim":10}' --exp 'epvt_latent_clustering_3'  --clustering True \
# --use_domain_labels False --loss-disc-weight --num-clustering 3

#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 5697 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT --output_dir \
#results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":130,"wd_classifier": 1e-5, "prompt_dim":10}' --exp 'epvt_c4_l1'  --clustering True \
# --use_domain_labels False --loss-disc-weight --num-clustering 4
#
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 5697 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT --output_dir \
#results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":130,"wd_classifier": 1e-5, "prompt_dim":10}' --exp 'epvt_c4_l0'  --clustering True \
# --use_domain_labels False --loss-disc-weight --num-clustering 4

#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 5697 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT --output_dir \
#results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":130,"wd_classifier": 1e-5, "prompt_dim":10}' --exp 'epvt_c4_l0_d'  --clustering True \
# --use_domain_labels False --loss-disc-weight --num-clustering 4
# CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 5697 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT --output_dir \
#results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":130,"wd_classifier": 1e-5, "prompt_dim":10}' --exp 'epvt_c2_l0_d'  --clustering True \
# --use_domain_labels False --loss-disc-weight --num-clustering 2

#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 5697 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT --output_dir \
#results/exp --hparams '{"lr": 5e-5, "lr_classifier": 5e-4,"batch_size":130,"wd_classifier": 1e-5, "prompt_dim":10}' --exp 'epvt_c4_l0_5e-5'  --clustering True \
# --use_domain_labels False --loss-disc-weight --num-clustering 4

#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 5900 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT --output_dir \
#results/exp --hparams '{"lr": 5e-7, "lr_classifier": 5e-7,"batch_size":130,"wd_classifier": 1e-4, "prompt_dim":4}' --exp 'epvt_5e-7_5e-7_1e-4_sne_exp'  --clustering True \
# --use_domain_labels False --loss-disc-weight --num-clustering 4

#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 5900 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT --output_dir \
#results/exp --hparams '{"lr": 5e-7, "lr_classifier": 5e-7,"batch_size":130,"wd_classifier": 1e-4, "prompt_dim":8}' --exp 'epvt_p8_5e-7_5e-7_1e-4_c3'  --clustering True \
# --use_domain_labels False --loss-disc-weight --num-clustering 3

#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 5900 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT --output_dir \
#results/exp --hparams '{"lr": 5e-7, "lr_classifier": 5e-7,"batch_size":130,"wd_classifier": 1e-2, "prompt_dim":4}' --exp 'epvt_p6_5e-7_5e-7_1e-2'  --clustering True \
# --use_domain_labels False --loss-disc-weight --num-clustering 4

#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 5900 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT --output_dir \
#results/exp --hparams '{"lr": 5e-7, "lr_classifier": 5e-6,"batch_size":130,"wd_classifier": 1e-4, "prompt_dim":6}' --exp 'epvt_p6_5e-7_5e-6_1e-4'  --clustering True \
# --use_domain_labels False --loss-disc-weight --num-clustering 4

#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 6200 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT --output_dir \
#results/exp --hparams '{"lr": 5e-7, "lr_classifier": 5e-5,"batch_size":130,"wd_classifier": 1e-5, "prompt_dim":4}' --exp 'epvt_p4_5e-7_5e-5'  --clustering True \
# --use_domain_labels False --loss-disc-weight --num-clustering 4

#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 5900 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT --output_dir \
#results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":120,"wd_classifier": 1e-5, "prompt_dim":10}' --exp 'epvt_c4_l0_p10'  --clustering True \
# --use_domain_labels False --loss-disc-weight --num-clustering 4

#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train --data_dir=./domainbed/data/ --steps 1501 --dataset SKIN --test_env 0 --algorithm DoPromptMixup --output_dir \
#results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":26,"wd_classifier": 1e-5, "prompt_dim":10}' --exp 'prompt_mixup' 



#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 5697 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT --output_dir \
#results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":130,"wd_classifier": 1e-5, "prompt_dim":10}' --exp 'epvt_c4_l7_d'  --clustering True \
# --use_domain_labels False --loss-disc-weight --num-clustering 4

#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 5697 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT --output_dir \
#results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":130,"wd_classifier": 1e-5, "prompt_dim":10}' --exp 'epvt_c4_l1_d'  --clustering True \
# --use_domain_labels False --loss-disc-weight --num-clustering 4

#local2_skin
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 5900 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT_NOG --output_dir \
#results/exp --hparams '{"lr": 5e-7, "lr_classifier": 5e-7,"batch_size":130,"wd_classifier": 1e-4, "prompt_dim":4}' --exp 'latyent_epvt_nog'  --clustering True \
# --use_domain_labels False --loss-disc-weight --num-clustering 4
# CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 5900 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT_NOMG --output_dir \
#results/exp --hparams '{"lr": 5e-7, "lr_classifier": 5e-7,"batch_size":130,"wd_classifier": 1e-4, "prompt_dim":4}' --exp 'latyent_epvt_nomg'  --clustering True \
# --use_domain_labels False --loss-disc-weight --num-clustering 4
# CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 5900 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT_N0_AMG --output_dir \
#results/exp --hparams '{"lr": 5e-7, "lr_classifier": 5e-7,"batch_size":130,"wd_classifier": 1e-4, "prompt_dim":4}' --exp 'latyent_epvt_noamg'  --clustering True \
# --use_domain_labels False --loss-disc-weight --num-clustering 4
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_dr_latent --data_dir=./domainbed/data/ --steps 1000 --dataset Latent_DR_Dataset --test_envs 2 --val_envs 1 \
#--algorithm Latent_EPVT_NOG --output_dir results/exp --hparams '{"lr":5e-6, "lr_classifier": 5e-5,"batch_size":130,"wd_classifier":1e-5}' --exp 'latent_dr_m2_nog' --clustering True
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_dr_latent --data_dir=./domainbed/data/ --steps 1000 --dataset Latent_DR_Dataset --test_envs 2 --val_envs 1 \
#--algorithm Latent_EPVT_NOMG --output_dir results/exp --hparams '{"lr":5e-6, "lr_classifier": 5e-5,"batch_size":130,"wd_classifier":1e-5}' --exp 'latent_dr_m2_nomg' --clustering True
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_dr_latent --data_dir=./domainbed/data/ --steps 1000 --dataset Latent_DR_Dataset --test_envs 2 --val_envs 1 \
#--algorithm Latent_EPVT_N0_AMG --output_dir results/exp --hparams '{"lr":5e-6, "lr_classifier": 5e-5,"batch_size":130,"wd_classifier":1e-5}' --exp 'latent_dr_m2_noamg' --clustering True

#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 5900 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT --output_dir \
#results/exp --hparams '{"lr": 5e-7, "lr_classifier": 5e-7,"batch_size":130,"wd_classifier": 1e-4, "prompt_dim":4}' --exp 'l1'  --clustering True \
# --use_domain_labels False --loss-disc-weight --num-clustering 2
#
#
#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 5697 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT --output_dir \
#results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":130,"wd_classifier": 1e-4, "prompt_dim":4}' --exp 'epvt_c4_l4_d'  --clustering True \
# --use_domain_labels False --loss-disc-weight --num-clustering 2

#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.test_prompt --model_name 'epvt_c4_l4_d.pkl'
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 5697 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT --output_dir \
#results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":130,"wd_classifier": 1e-4, "prompt_dim":10}' --exp 'epvt_c4_l4_d'  --clustering True \
# --use_domain_labels False --loss-disc-weight --num-clustering 2


CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 5697 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT --output_dir \
results/exp --hparams '{"lr": 5e-7, "lr_classifier": 5e-7,"batch_size":130,"wd_classifier": 1e-4, "prompt_dim":4}' --exp 'l1_v1'  --clustering True \
 --use_domain_labels False --loss-disc-weight --num-clustering 2
CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 5697 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT --output_dir \
results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":130,"wd_classifier": 1e-5, "prompt_dim":10}' --exp 'l1_v2'  --clustering True \
 --use_domain_labels False --loss-disc-weight --num-clustering 2
CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 5697 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT --output_dir \
results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":130,"wd_classifier": 1e-5, "prompt_dim":10}' --exp 'l1_v3'  --clustering True \
 --use_domain_labels False --loss-disc-weight --num-clustering 4