import os
import types
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from domainbed import algorithms, datasets, hparams_registry
from domainbed.lib.fast_data_loader import FastDataLoader, InfiniteDataLoader
import random
import numpy as np
from tqdm import tqdm
import pandas as pd
# from utils import AverageMeter, accuracy
# from loss import LossComputer
# from dataset_loader import CSVDatasetWithName, CSVDataset
# #from pytorch_transformers import AdamW, WarmupLinearSchedule
# from sklearn.metrics import confusion_matrix, roc_auc_score
# from data.skin_dataset import get_transform_skin
import pickle

from domainbed.lib import misc
from torchvision import transforms
class AugmentOnTest:
    def __init__(self, dataset, n):
        self.dataset = dataset
        self.n = n

    def __len__(self):
        return self.n * len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i // self.n]
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Domain generalization Testing')
    parser.add_argument('--model_name', type=str, default='model.pkl')
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    args = parser.parse_args()


    def set_random_seed(seed, deterministic=False):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


    set_random_seed(0)
    root = '/mount/neuron/Lamborghini/dir/pythonProject/TMI/'
    BEST_MODEL_PATH=root+'PLDG/results/exp/'+args.model_name
    # RUN TESTS
    print("Performing tests. Loading model at {}".format(BEST_MODEL_PATH))
    hparams = hparams_registry.default_hparams('Latent_EPVT', 'SKIN')
    print('HHHHHHHH',hparams)
    hparams['prompt_dim']=10
    algorithm_class = algorithms.get_algorithm_class('Latent_EPVT')
    # (3,224,224), num class, 3 domain
    algorithm = algorithm_class((3,224,224), 2,
                                5, hparams)


    algorithm.load_state_dict(torch.load(BEST_MODEL_PATH)['model_dict'])
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model=algorithm
    model.to(device)
    root = '/mount/neuron/Lamborghini/dir/pythonProject/MICCAI/'
    # Test for Skin
    # val_transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    val_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomRotation(45),
        transforms.ColorJitter(hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_ds_atlas_clin = datasets.CSVDataset(root + '/artifact-generalization-skin/datasets/edraAtlas',
                                    root + '/artifact-generalization-skin/datasets/edraAtlas/atlas-clinical-all.csv',
                                    'image', 'label', transform=val_transform, add_extension='.jpg')
    test_ds_atlas_derm = datasets.CSVDataset(root + '/artifact-generalization-skin/datasets/edraAtlas',
                                    root + '/artifact-generalization-skin/datasets/edraAtlas/atlas-dermato-all.csv',
                                    'image', 'label', transform=val_transform, add_extension='.jpg')
    test_ds_ph2 = datasets.CSVDataset(root + '/artifact-generalization-skin/datasets/ph2images/',
                             root + '/artifact-generalization-skin/datasets/ph2images/ph2.csv', 'image', 'label',
                             transform=val_transform, add_extension='.png')
    test_ds_padufes = datasets.CSVDataset(root + '/artifact-generalization-skin/datasets/pad-ufes/',
                                 root + '/artifact-generalization-skin/datasets/pad-ufes/padufes-test-wocarc.csv',
                                 'img_id', 'label', transform=val_transform, add_extension=None)

    shuffle = False
    data_sampler = None
    num_workers = 16
    REPLICAS = 128

    # def collate_fn(batch):
    #     batch = list(filter(lambda x: x is not None, batch))
    #     return torch.utils.data.dataloader.default_collate(batch)

    dataloaders_atlas_dermato = {
        'val': DataLoader(test_ds_atlas_derm, batch_size=REPLICAS,
                          shuffle=shuffle, num_workers=num_workers,
                          sampler=data_sampler, pin_memory=True)
    }
    dataloaders_atlas_clin = {
        'val': DataLoader(test_ds_atlas_clin, batch_size=REPLICAS,
                          shuffle=shuffle, num_workers=num_workers,
                          sampler=data_sampler, pin_memory=True)
    }
    dataloaders_ph2 = {
        'val': DataLoader(test_ds_ph2, batch_size=REPLICAS,
                          shuffle=shuffle, num_workers=num_workers,
                          sampler=data_sampler, pin_memory=True)
    }
    dataloaders_padufes = {
        'val': DataLoader(test_ds_padufes, batch_size=REPLICAS,
                          shuffle=shuffle, num_workers=num_workers,
                          sampler=data_sampler, pin_memory=True)
    }

    weights = None
    """evaluating performance of eval dataset"""
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print('OOD evaluation')

    ACC, BACC, Prec, Rec, F1, AUC, SPEC, kappa, val_loss = misc.eval_indomain(model, loader=dataloaders_atlas_dermato['val'], valid_df=test_ds_atlas_derm,
                                                                                  batch_size=REPLICAS,
                                                                                  weights=weights, device=device, name=None,
                                                                                  domain=None)



    print('---------------------derm7pt_derm---------------------')
    print(' Test acc: {:.3f} | Test bacc: {:.3f} |Test roc_auc: {:.6f} | F1 : {:.3f}:'.format(
            ACC,
            BACC,
            AUC,
            F1))
    ACC, BACC, Prec, Rec, F1, AUC, SPEC, kappa, val_loss = misc.eval_indomain(model, loader=dataloaders_atlas_clin['val'], valid_df=test_ds_atlas_clin,
                                                                                  batch_size=REPLICAS,
                                                                                  weights=weights, device=device, name=None,
                                                                                  domain=None)



    print('---------------------derm7pt_clinic---------------------')
    print(' Test acc: {:.3f} | Test bacc: {:.3f} |Test roc_auc: {:.6f} | F1 : {:.3f}:'.format(
            ACC,
            BACC,
            AUC,
            F1))

    ACC, BACC, Prec, Rec, F1, AUC, SPEC, kappa, val_loss = misc.eval_indomain(model, loader=dataloaders_ph2['val'], valid_df=test_ds_ph2,
                                                                                  batch_size=REPLICAS,
                                                                                  weights=weights, device=device, name=None,
                                                                                  domain=None)



    print('---------------------ph2---------------------')
    print(' Test acc: {:.3f} | Test bacc: {:.3f} |Test roc_auc: {:.6f} | F1 : {:.3f}:'.format(
            ACC,
            BACC,
            AUC,
            F1))
    ACC, BACC, Prec, Rec, F1, AUC, SPEC, kappa, val_loss = misc.eval_indomain(model, loader=dataloaders_padufes['val'], valid_df=test_ds_padufes,
                                                                                  batch_size=REPLICAS,
                                                                                  weights=weights, device=device, name=None,
                                                                                  domain=None)



    print('---------------------pad---------------------')
    print(' Test acc: {:.3f} | Test bacc: {:.3f} |Test roc_auc: {:.6f} | F1 : {:.3f}:'.format(
            ACC,
            BACC,
            AUC,
            F1))



    #
    # print('Indomain Evaluation')
    #
    #
    # eval_class = FastDataLoader
    # eval_root = '/mount/neuron/Lamborghini/dir/pythonProject/MICCAI/prompt_derm/data_proc/'
    # test_df = pd.read_csv(eval_root + 'test_bias_0_1.csv')
    # imfolder = '/mount/neuron/Lamborghini/dir/pythonProject/CVPR/data/ISIC_2019_Training/'
    # test = datasets.MelanomaDataset(df=test_df,
    #                                 imfolder=imfolder)
    # test_loaders = eval_class(
    #     dataset=test,
    #     batch_size=REPLICAS,
    #     num_workers=num_workers)
    #
    # ACC, BACC, Prec, Rec, F1, AUC, SPEC, kappa, val_loss = misc.eval_indomain(model, loader=test_loaders, valid_df=test_df,
    #                                                                               batch_size=REPLICAS,
    #                                                                               weights=weights, device=device, name=None,
    #                                                                               domain=None)
    #
    #
    # print('---------------------ISIC2019 Testing---------------------')
    # print(' Test acc: {:.3f} | Test bacc: {:.3f} |Test roc_auc: {:.6f} | F1 : {:.3f}:'.format(
    #         ACC,
    #         BACC,
    #         AUC,
    #         F1))
    print('done')

