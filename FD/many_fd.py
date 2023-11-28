import os
import sys

sys.path.append(".")
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
from scipy import linalg
from torchvision import transforms
from tqdm import trange

import os
import types
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from domainbed import algorithms, datasets, hparams_registry
from domainbed.lib.fast_data_loader import FastDataLoader, InfiniteDataLoader
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
import argparse
from domainbed.lib import misc
from torchvision import transforms

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                        description='PyTorch MNIST FD-Metaset')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Batch size to use')
parser.add_argument('--dims', type=int, default=768,
                    choices=list([128, 9216]),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('-c', '--gpu', default='1', type=str,
                    help='GPU to use (leave blank for CPU only)')
#
# import torchsnooper
# @torchsnooper.snoop()
def get_activations(files, model, batch_size=50, dims=2048,
                    cuda=False, overlap=False, verbose=False,test=None):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FD score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    num_workers = 16
    REPLICAS = 128
    if overlap:
        shuffle = False
        data_sampler = None
        num_workers = 16
        REPLICAS = 128

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
                                          root + '/artifact-generalization-skin/datasets/ph2images/ph2.csv', 'image',
                                          'label',
                                          transform=val_transform, add_extension='.png')
        test_ds_padufes = datasets.CSVDataset(root + '/artifact-generalization-skin/datasets/pad-ufes/',
                                              root + '/artifact-generalization-skin/datasets/pad-ufes/padufes-test-wocarc.csv',
                                              'img_id', 'label', transform=val_transform, add_extension=None)
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
        if test==0:
            test_loader=dataloaders_atlas_dermato['val']
            test_df=test_ds_atlas_derm
        elif test==1:
            test_loader=dataloaders_atlas_clin['val']
            test_df=test_ds_atlas_clin
        elif test==2:
            test_loader=dataloaders_ph2['val']
            test_df=test_ds_ph2
        elif test==3:
            test_loader=dataloaders_padufes['val']
            test_df=test_ds_padufes
        else:
            test_loader,test_df=None,None

    else:
        eval_class = FastDataLoader
        eval_root = '/mount/neuron/Lamborghini/dir/pythonProject/MICCAI/prompt_derm/data_proc/'
        test_df = pd.read_csv(eval_root + 'test_bias_0_1.csv')
        imfolder = '/mount/neuron/Lamborghini/dir/pythonProject/CVPR/data/ISIC_2019_Training/'
        test = datasets.MelanomaDataset(df=test_df,
                                        imfolder=imfolder)
        test_loader = eval_class(
            dataset=test,
            batch_size=REPLICAS,
            num_workers=num_workers)
    n_batches = len(test_df) // REPLICAS
    n_used_imgs = n_batches * REPLICAS

    pred_arr = np.empty((n_used_imgs, dims))
    for i, data in enumerate(test_loader):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)
        start = i * REPLICAS
        end = start + REPLICAS

        batch, _ = data

        if cuda:
            batch = batch.cuda()

        pred = model.get_act(batch, domain=None)
        if pred.size(0)!=REPLICAS:
            break
        pred_arr[start:end] = pred.cpu().data.numpy().reshape(REPLICAS, -1)

    if verbose:
        print(' done')
    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=50,
                                    dims=2048, cuda=False, overlap=False,test=None):
    """Calculation of the statistics used by the FD.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, cuda, overlap,test=test)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma, act


def _compute_statistics_of_path(path, model, batch_size, dims, cuda, overlap,test=None):
    m, s, act = calculate_activation_statistics(path, model, batch_size,
                                                dims, cuda, overlap,test=test)

    return m, s, act


def calculate_fid_given_paths(path, batch_size, cuda, dims,test=None):
    """Calculates the FD of two paths"""
    print()
    m2, s2, act2 = _compute_statistics_of_path(path, model, batch_size,
                                               dims, cuda, overlap=True,test=test)

    return m2, s2, act2


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_cuda = args.gpu and torch.cuda.is_available()

    root = '/mount/neuron/Lamborghini/dir/pythonProject/MICCAI/'
    # BEST_MODEL_PATH=root+'prompt_derm/results/exp/erm_1e-5_3e-4_weight.pkl'
    # # RUN TESTS
    # print("Performing tests. Loading model at {}".format(BEST_MODEL_PATH))
    # hparams = hparams_registry.default_hparams('ERM', 'SKIN')
    #
    # algorithm_class = algorithms.get_algorithm_class('ERM')

    BEST_MODEL_PATH=root+'prompt_derm/results/exp/prompt_mixup_group.pkl'
    # RUN TESTS
    print("Performing tests. Loading model at {}".format(BEST_MODEL_PATH))
    hparams = hparams_registry.default_hparams('DoPrompt_group', 'SKIN')

    algorithm_class = algorithms.get_algorithm_class('DoPrompt_group')

    hparams.update(json.loads('{"lr": 1e-5, "lr_classifier": 3e-4,"batch_size":26}'))

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


    model.eval()

    # test_dirs = sorted(os.listdir('./dataset_bg'))
    feat_path = './FD/dataset_feature/'
    try:
        os.makedirs(feat_path)
    except:
        None

    fd_bg = []

    with torch.no_grad():
        '''
        training dataset (overlap=False--> source dataset)
        test dataset (overlap=True--> sample set)
        '''
        # training dataset (overlap=False--> source dataset)
        m1, s1, act1 = _compute_statistics_of_path('', model, 128,
                                                   768, args.gpu != '', overlap=False)
        #768,(768,768)
        print('isic done')
        # # saving features of training set
        np.save(feat_path + 'train_mean', m1)
        np.save(feat_path + 'train_variance', s1)
        np.save(feat_path + 'train_feature', act1)
        #
        print('---------domain distance----------')
        data=['derm7pt_derm','derm7pt_clin','ph2','pad']
        for i in range(4):
            # test dataset (overlap=True--> sample set)

            m2, s2, act2 = calculate_fid_given_paths(None,128,args.gpu != '',768,test=i)

            fd_value = calculate_frechet_distance(m1, s1, m2, s2)


            print('---',data[i],'---')
            print('FD: ', fd_value)
        #     fd_bg.append(fd_value)
        #
        #     # saving features for nn regression
        #     np.save(feat_path + '_%s_mean' % (path), m2)
        #     np.save(feat_path + '_%s_variance' % (path), s2)
        #     np.save(feat_path + '_%s_feature' % (path), act2)
        #
        # np.save('./FD/fd_mnist.npy', fd_bg)
