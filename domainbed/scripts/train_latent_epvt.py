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
from domainbed.dataloader import my_dataloader
from domainbed.lib.fast_data_loader import FastDataLoader, InfiniteDataLoader
from domainbed.lib.torchmisc import dataloader
from domainbed.clustering.domain_split import domain_split
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
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--restore', type=str, default=None)
    parser.add_argument('--exp', type=str, default='miccai_project')
    parser.add_argument('--use_domain_labels', type=bool, default=True)
    parser.add_argument('--clustering', type=bool, default=False)
    parser.add_argument('--clustering-step', type=int, default=1)
    parser.add_argument('--clustering-method', type=str, default='Kmeans')
    parser.add_argument('--num-clustering', type=int, default=5)
    parser.add_argument('--instance-stat', type=bool, default=True)
    parser.add_argument('--loss-disc-weight', action='store_true')


    args = parser.parse_args()

    wandb.init(name=args.exp,
               project="TMI",
               notes="prompt",
               tags=["isic-19"],
               config=args
               )
    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

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


    domain=['clean','dark_corner','gel_bubble','hair','ruler']

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](root_dir=args.data_dir+'ISIC2019_train/',domain=domain,split='val',
                                               get_domain_label=args.use_domain_labels, get_cluster=args.clustering, color_jitter=True)
    else:
        raise NotImplementedError

    """return (img,label,domain labels,pseudo domain labels)"""

    num_workers_eval = 8 if args.dataset != "DomainNet" else 8
    batch_size_eval = 128 if args.dataset != "DomainNet" else 128
    eval_class = FastDataLoader if args.dataset != "DomainNet" else dataloader
    ###eval dataset and test dataset
    eval_root='/mount/neuron/Lamborghini/dir/pythonProject/TMI/PLDG/data_proc/'
    imfolder='/mount/neuron/Lamborghini/dir/pythonProject/TMI/PLDG/domainbed/data/ISIC2019_train/'

    test_df = pd.read_csv(eval_root+'test_bias_0_1.csv')
    val_df=pd.read_csv(eval_root+'val_bias_0_1.csv')
    val = datasets.MelanomaDataset(df=val_df,
                          imfolder=imfolder)

    test = datasets.MelanomaDataset(df=test_df,
                           imfolder=imfolder)

    print('Val Size:',len(val), '，','Test Size:',len(test))

    eval_loaders = eval_class(
        dataset=val,
        batch_size=batch_size_eval,
        num_workers=num_workers_eval)
    test_loaders = eval_class(
        dataset=val,
        batch_size=batch_size_eval,
        num_workers=num_workers_eval)


    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    #(3,224,224), num class, 3 domain
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,dataset.num_domain, hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)
    
    if args.restore:
        ckpt = torch.load(args.restore)
        missing_keys, unexpected_keys = algorithm.load_state_dict(ckpt["model_dict"], strict=False)
        print("restored from {}".format(args.restore))
        print("missing keys: {}".format(missing_keys))
        print("unexpected keys: {}".format(unexpected_keys))

    # train_minibatches_iterator = zip(*train_loaders)

    checkpoint_vals = collections.defaultdict(lambda: [])
    # steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])
    steps_per_epoch=len(dataset)/hparams['batch_size']
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


    exp_file='results/' + args.exp
    if not os.path.exists(exp_file):
        # create the folder if it does not exist
        os.makedirs(exp_file)
    last_results_keys = None
    es_patience = 200  # Early Stopping patience - for how many epochs with no improvements to wait
    best_val = 0  # Best validation score within this fold
    patience = es_patience  # Current patience counter
    for step in range(start_step, n_steps):
        steps_per_epoch = int(steps_per_epoch)
        epoch = step // steps_per_epoch
        # print(epoch)
        if step % steps_per_epoch == 0 or (step == n_steps - 1):
            step_start_time = time.time()

        if args.clustering:
            if epoch % args.clustering_step == 0 and step % steps_per_epoch==0:
                if epoch==0 or epoch==10:
                    print('-----------pseudo label clustering----------')
                    pseudo_domain_label = domain_split(dataset, algorithm, device=device,
                                                cluster_before=dataset.clusters,
                                                filename= 'results/'+args.exp+'/nmi.txt', epoch=epoch,
                                                nmb_cluster=args.num_clustering, method=args.clustering_method,
                                                pca_dim=256, whitening=False, L2norm=False, instance_stat=args.instance_stat,exp_name=args.exp)
                    dataset.set_cluster(np.array(pseudo_domain_label))
                    train_loaders = my_dataloader(dataset=dataset, batch_size=hparams['batch_size'], num_workers=8,
                                                  get_domain_label=args.use_domain_labels, get_cluster=args.clustering)
                    train_minibatches_iterator = iter(train_loaders)
                    # Define the path to the directory where you want to save the images
                    save_dir = '/mount/neuron/Lamborghini/dir/pythonProject/TMI/latent_prompt/results/' + args.exp+str(epoch)

                    # Loop through each item in the dataset
                    # for i in range(len(dataset)):
                    #     # Get the image, label, and cluster label
                    #     image, label, _, cluster_label = dataset[i]
                    #
                    #     # Create a folder for the cluster label if it doesn't exist
                    #     cluster_dir = os.path.join(save_dir, f"cluster_{cluster_label}")
                    #     if not os.path.exists(cluster_dir):
                    #         os.makedirs(cluster_dir)
                    #
                    #     # Get the filename of the image and save it to the corresponding folder
                    #     filename = os.path.basename(dataset.images[i])
                    #     save_path = os.path.join(cluster_dir, filename)
                    #     copyfile(dataset.images[i], save_path)

        #torch.Size([130, 3, 224, 224]) torch.Size([130]) torch.Size([130]) torch.Size([130])
        minibatches_device=[ele.to(device) for ele in next(train_minibatches_iterator)]
        uda_device = None
        step_vals = algorithm.update(minibatches_device, uda_device)

        loss_dp,loss_a,loss_w=step_vals['loss_dp'],step_vals['loss_a'],step_vals['loss_w']


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

            ACC, BACC, Prec, Rec, F1, AUC_ovo, SPEC, kappa, val_loss=misc.eval_indomain(algorithm, loader=evals, valid_df=val_df,batch_size=batch_size_eval,weights=weights, device=device, name=None, domain=None)
            val_acc = ACC
            val_roc = AUC_ovo

            print(
                'Epoch {:03}:  | Loss_dp: {:.3f} | Loss_a: {:.3f}  | Loss_w: {:.3f}  Val Loss: {:.3f} | Val acc: {:.3f} | Val bacc: {:.3f} |Val roc_auc: {:.6f} | F1 : {:.3f}| Training time: {}'.format(
                    epoch,
                    loss_dp,
                    loss_a,
                    loss_w,
                    val_loss,
                    val_acc,
                    BACC,
                    val_roc,
                    F1,
                    str(datetime.timedelta(seconds=time.time() - step_start_time))[:7]))



            wandb.log({'train ce loss': loss_dp, 'train_second_ce_loss': loss_a, 'train_loss_reg': loss_w,
                       'val loss': val_loss,
                       'val auc': val_roc, 'val acc': val_acc, 'val balanced acc': BACC, 'F1 score': F1})



            if epoch%50==0:
                save_checkpoint(args.exp + str(epoch) + '.pkl')
            if val_roc >= best_val:
                best_val = val_roc
                patience = es_patience  # Resetting patience since we have new best validation accuracy
                save_checkpoint(args.exp+'.pkl')
            else:
                patience -= 1
                if patience == 0:
                    print('Early stopping. Best Val roc_auc: {:.3f}'.format(best_val))
                    break
            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
    wandb.finish()
