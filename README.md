# Prompt-driven Latent Domain Generalization for Medical Image Classification

WARNING: this repo is still under construction, welcome to raise an issue if you run into any bugs.

## Introduction
**[abstract]** *Deep learning models for medical image analysis easily suffer from distribution shifts caused by dataset
artifacts bias, camera variations, differences in the imaging station, etc., leading to unreliable diagnoses in realworld clinical settings. Domain generalization (DG) methods, which aim to train models on multiple domains to perform well on unseen domains, offer a promising direction to
solve the problem. However, existing DG methods assume
domain labels of each image are available and accurate,
which is typically feasible for only a limited number of
medical datasets. To address these challenges, we propose
a novel DG framework for medical image classification
without relying on domain labels, called Prompt-driven
Latent Domain Generalization (PLDG). PLDG consists of
unsupervised domain discovery and prompt learning. This
framework first discovers pseudo domain labels by clustering the bias-associated style features, then leverages collaborative domain prompts to guide a Vision Transformer
to learn knowledge from discovered diverse domains. To
facilitate cross-domain knowledge learning between different prompts, we introduce a domain prompt generator that
enables knowledge sharing between domain prompts and
a shared prompt. A domain mixup strategy is additionally
employed for more flexible decision margins and mitigates
the risk of incorrect domain assignments. Extensive experiments on three medical image classification tasks and one
debiasing task demonstrate that our method can achieve
comparable or even superior performance than conventional DG algorithms without relying on domain labels.*


## Installation
Create the environment and install packages
```
conda create -n env_name python=3.8 -y
conda activate env_name
pip install -r requirements.txt
```

## Preparing datasets

**ISIC2019**: download ISIC2019 training dataset from [here](https://challenge.isic-archive.com/data/#2019)

**Derm7pt**: download Derm7pt Clinical and Derm7pt Dermoscopic dataset from [here](https://derm.cs.sfu.ca/Welcome.html)

**PH2**: download the PH2 dataset from [here](https://www.fc.up.pt/addi/ph2%20database.html)

**PAD**: download the PAD-UFES-20 dataset from [here](https://paperswithcode.com/dataset/pad-ufes-20)

## Pre-processing datasets

Pre-processing the ISIC2019 dataset to construct the artifacts-based domain generalization training dataset, you need to modify path names in the pre-processing file accordingly.
```
python data_proc/grouping.py
```

## Directly accessing all datasets via GoogleDrive

The processed ISIC2019 dataset  and 4 OOD testing datasets are in [GoogleDrive](https://drive.google.com/file/d/12SoMs_44jD4mRT6JEyIfdjBa4Fw07i2m/view?usp=sharing).
Please refer to our paper and its supplementary material for more details about these datasets.

## Training

Our benchmark is modified based on DomainBed, please refer to [DomainBed Readme](https://github.com/facebookresearch/DomainBed) for more details on commands running jobs. Here are some examples to train and test on the three medical datasets.

```sh
# Training PLDG on skin classification
CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train_latent_epvt --data_dir=./domainbed/data/ --steps 5900 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT --output_dir \
results/exp --hparams '{"lr": 5e-6, "lr_classifier": 5e-5,"batch_size":130,"wd_classifier": 1e-2, "prompt_dim":4}' --exp 'epvt_5e-6-5e-5_p4' --ood_vis True --clustering True \
 --use_domain_labels False --loss-disc-weight --num-clustering 4

#Test PLDG on four OOD skin datasets with pompt=10
CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.test_prompt10 --model_name 'model_name.pkl'

#Test PLDG on four OOD skin datasets with pompt=4
CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.test_prompt --model_name 'model_name.pkl'

# Training PLDG on three out of four DR classification datasets and testing on the remaining one
CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_dr_latent --data_dir=./domainbed/data/ --steps 1000 --dataset Latent_DR_Dataset --test_envs 0 --val_envs 1 \
--algorithm Latent_EPVT --output_dir results/exp --hparams '{"lr":5e-6, "lr_classifier": 5e-5,"batch_size":130,"wd_classifier":1e-5,"prompt_dim":10,"seed":1}' --exp 'latent_dr_eyepacs_p10' --clustering True
 CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_dr_latent --data_dir=./domainbed/data/ --steps 5697 --dataset Latent_DR_Dataset --test_envs 2 --val_envs 1 \
--algorithm Latent_EPVT --output_dir results/exp --hparams '{"lr":5e-6, "lr_classifier": 5e-5,"batch_size":130,"wd_classifier":1e-5}' --exp 'latent_dr_m2' --clustering True 
CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_dr_latent --data_dir=./domainbed/data/ --steps 5697 --dataset Latent_DR_Dataset --test_envs 3 --val_envs 1 \
--algorithm Latent_EPVT --output_dir results/exp --hparams '{"lr":5e-7, "lr_classifier": 5e-6,"batch_size":130,"wd_classifier":1e-5}' --exp 'latent_dr_aptos_5e-7' --clustering True

# training PLDG on Camelyon17-wilds dataset
CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train_cam17 --data_dir=./domainbed/data/ --steps 5900 --dataset DG_Dataset --test_env 0 --algorithm Latent_EPVT --output_dir \
results/exp --hparams '{"lr": 5e-7, "lr_classifier": 5e-7,"batch_size":130,"wd_classifier": 1e-4, "prompt_dim":4}' --exp 'epvt_5e-7_5e-7_1e-4_sne_exp' --ood_vis True --clustering True \
 --use_domain_labels False --loss-disc-weight --num-clustering 4

```



## Citation

```bibtex
@inproceedings{yan2023epvt,
  title={EPVT: Environment-Aware Prompt Vision Transformer for Domain Generalization in Skin Lesion Recognition},
  author={Yan, Siyuan and Liu, Chi and Yu, Zhen and Ju, Lie and Mahapatra, Dwarikanath and Mar, Victoria and Janda, Monika and Soyer, Peter and Ge, Zongyuan},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={249--259},
  year={2023},
  organization={Springer}
}
```

## Acknowlegdement

This code is built on [DomainBed](https://github.com/facebookresearch/DomainBed), [DoPrompt](https://github.com/zhengzangw/DoPrompt), [DG_SKIN](https://github.com/alceubissoto/artifact-generalization-skin), and [dg_mmld](https://github.com/mil-tokyo/dg_mmld). We thank the authors for sharing their codes.

