

import argparse
import collections
import json
import os
import random
import sys
import uuid
import wandb
import numpy as np
import PIL
import torch
import torch.utils.data
import torchvision
from tqdm import tqdm
from domainbed import algorithms, datasets, hparams_registry
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import FastDataLoader, InfiniteDataLoader
from domainbed.lib.torchmisc import dataloader
import pandas as pd
import datetime
import time
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--val_envs', type=int, nargs='+', default=[1])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--restore', type=str, default=None)
    parser.add_argument('--exp', type=str, default='miccai_project')

    args = parser.parse_args()


    wandb.init(name=args.exp,
               project="TMI",
               notes="prompt",
               tags=["DR"],
               config=args
               )
    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None
    domain_folders = ['EyePACS', 'Messidor-1', 'Messidor-2', 'aptos2019-blindness-detection']

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))
    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError
    print('----------------',dataset)

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.

    """give each enviornment dataset a weight"""
    in_splits = []
    # out_splits = []
    uda_splits = []
    #dataset: [d1 d2 d3 d4 d5]
    for env_i, env in enumerate(dataset):
        #for each dataset [d1,d2,d3,d4]
        uda = []
        #in weights for isic2019 training
        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(env)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((env, in_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))


    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")
    #[data2,3,4]

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs or args.val_envs]
    # for i, (env, env_weights) in enumerate(in_splits):
    #     if i not in args.test_envsL
    # uda_loaders = [InfiniteDataLoader(
    #     dataset=env,
    #     weights=env_weights,
    #     batch_size=hparams['batch_size'],
    #     num_workers=dataset.N_WORKERS)
    #     for i, (env, env_weights) in enumerate(uda_splits)
    #     if i in args.test_envs]

    num_workers_eval = 8 if args.dataset != "DomainNet" else 8
    batch_size_eval = 128 if args.dataset != "DomainNet" else 128
    eval_class = FastDataLoader if args.dataset != "DomainNet" else dataloader
    ###eval dataset and test dataset
    eval_root='/mount/neuron/Lamborghini/dir/pythonProject/TMI/latent_prompt/domainbed/data/DG_DR_Classification/Messidor-1/'
    test_root='/mount/neuron/Lamborghini/dir/pythonProject/TMI/latent_prompt/domainbed/data/DG_DR_Classification/'

    # test_df = pd.read_csv(eval_root+'test_bias_0_1.csv')
    # val_df=pd.read_csv(eval_root+'val_bias_0_1.csv')
    # # val = datasets.MelanomaDataset(df=val_df,
    # #                       imfolder=imfolder)

    for i, (env, env_weights) in enumerate(in_splits):
        if i in args.val_envs:
            eval_root='/mount/neuron/Lamborghini/dir/pythonProject/TMI/latent_prompt/domainbed/data/DG_DR_Classification/'+domain_folders[i]+'/'
            print('Val env:',domain_folders[i])
        if i in args.test_envs:
            test_root = '/mount/neuron/Lamborghini/dir/pythonProject/TMI/latent_prompt/domainbed/data/DG_DR_Classification/'+domain_folders[i]+'/'
            print('Test env:', domain_folders[i])

    val_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomRotation(45),
        transforms.ColorJitter(hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val=datasets.FolderImageDataset(root_dir=eval_root,transform=val_transform)
    test=datasets.FolderImageDataset(root_dir=test_root,transform=val_transform)


    print('Val Size:',len(val), '，','Test Size:',len(test))

    eval_loaders = eval_class(
        dataset=val,
        batch_size=batch_size_eval,
        num_workers=num_workers_eval)
    test_loaders = eval_class(
        dataset=test,
        batch_size=batch_size_eval,
        num_workers=num_workers_eval)


    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    #(3,224,224), num class, 3 domain
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) , hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)
    
    if args.restore:
        ckpt = torch.load(args.restore)
        missing_keys, unexpected_keys = algorithm.load_state_dict(ckpt["model_dict"], strict=False)
        print("restored from {}".format(args.restore))
        print("missing keys: {}".format(missing_keys))
        print("unexpected keys: {}".format(unexpected_keys))
        
    # domain mapping
    cnt = 0
    #domain mapping: {0: None, 1: 0, 2: 1, 3: 2}
    domain_mapping = {x: None for x in args.test_envs}
    for i in range(len(in_splits)):
        if i not in args.test_envs:
            domain_mapping[i] = cnt
            cnt += 1
    train_minibatches_iterator = zip(*train_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])
    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ
    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    last_results_keys = None
    es_patience = 200  # Early Stopping patience - for how many epochs with no improvements to wait
    best_val = 0  # Best validation score within this fold
    patience = es_patience  # Current patience counter
    for step in range(start_step, n_steps):
        steps_per_epoch = int(steps_per_epoch)
        epoch = step // steps_per_epoch
        if step % steps_per_epoch == 0 or (step == n_steps - 1):
            step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
            for x,y in next(train_minibatches_iterator)]
        uda_device = None
        step_vals = algorithm.update(minibatches_device, uda_device)
        loss_train =0

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if step%steps_per_epoch == 0 or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)
            evals = eval_loaders
            weights=None
            """evaluating performance of eval dataset"""
            val_acc = misc.accuracy(algorithm, eval_loaders, weights, device, name=None, domain=None)

            print(
                'Epoch {:03}:  | Loss_train: {:.3f}  | VAL ACC: {:.3f} | Training time: {}'.format(
                    epoch,
                    loss_train,
                    val_acc,
                    str(datetime.timedelta(seconds=time.time() - step_start_time))[:7]))

            wandb.log({'train loss': loss_train,
                       'val acc': val_acc})

            if epoch == 50 or epoch == 100 or epoch == 150 or epoch == 200 or epoch == 250:
                save_checkpoint(args.exp + str(epoch) + '.pkl')
            if val_acc >= best_val:
                best_val = val_acc
                patience = es_patience  # Resetting patience since we have new best validation accuracy
                save_checkpoint(args.exp+'.pkl')
            else:
                patience -= 1
                if patience == 0:
                    print('Early stopping. Best Val acc: {:.3f}'.format(best_val))
                    break
            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

    print('----------------testing----------------')
    weights = None
    algorithm.load_state_dict(torch.load('/mount/neuron/Lamborghini/dir/pythonProject/TMI/latent_prompt/results/exp/'+args.exp+'.pkl')['model_dict'])
    test_acc = misc.accuracy(algorithm, test_loaders, weights, device, name=None, domain=None)
    print('Test Acc:',test_acc)
    wandb.finish()
