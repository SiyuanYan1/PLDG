

"""
Things that don't belong anywhere else
"""

import hashlib
import json
import operator
import os
import sys
from collections import Counter, OrderedDict, defaultdict
from numbers import Number
from shutil import copyfile
import datetime
import time
import copy
from enum import Enum
import numpy as np
import torch
import tqdm
from domainbed.lib import metrics

def l2_between_dicts(dict_1, dict_2):
    assert len(dict_1) == len(dict_2)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    return (
        torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
        torch.cat(tuple([t.view(-1) for t in dict_2_values]))
    ).pow(2).mean()

class MovingAverage:

    def __init__(self, ema, oneminusema_correction=True):
        self.ema = ema
        self.ema_data = {}
        self._updates = 0
        self._oneminusema_correction = oneminusema_correction

    def update(self, dict_data):
        ema_dict_data = {}
        for name, data in dict_data.items():
            data = data.view(1, -1)
            if self._updates == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self.ema_data[name]

            ema_data = self.ema * previous_data + (1 - self.ema) * data
            if self._oneminusema_correction:
                # correction by 1/(1 - self.ema)
                # so that the gradients amplitude backpropagated in data is independent of self.ema
                ema_dict_data[name] = ema_data / (1 - self.ema)
            else:
                ema_dict_data[name] = ema_data
            self.ema_data[name] = ema_data.clone().detach()

        self._updates += 1
        return ema_dict_data
class MovingAvg:
    def __init__(self, network, ema=False, sma_start_iter=100):
        self.network = network
        self.network_sma = copy.deepcopy(network)
        self.sma_start_iter = sma_start_iter
        self.global_iter = 0
        self.sma_count = 0
        self.ema = ema

    def update_sma(self):
        self.global_iter += 1
        if self.global_iter >= self.sma_start_iter and self.ema:
            # if False:
            self.sma_count += 1
            for param_q, param_k in zip(
                self.network.parameters(), self.network_sma.parameters()
            ):
                param_k.data = (param_k.data * self.sma_count + param_q.data) / (
                    1.0 + self.sma_count
                )
        else:
            for param_q, param_k in zip(
                self.network.parameters(), self.network_sma.parameters()
            ):
                param_k.data = param_q.data



def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights

def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()

def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def print_separator():
    print("="*80)

def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)

class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]
    def __len__(self):
        return len(self.keys)
def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)
def random_pairs_of_minibatches(minibatches):
    #minibatches = [(tensor<(26, 3, 224, 224), float32, cuda:0>, te... float32, cuda:0>, tensor<(26,), int64, cuda:0>)]
    #peerm[2,1,4,0,3]
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0
        #pick up from ith domain
        # xi = tensor<(26, 3, 224, 224), float32, cuda:0>
        # yi = tensor<(26,), int64, cuda:0>
        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]
        #26
        min_n = min(len(xi), len(xj))
        #pairs = [((tensor<(26, 3, 224, 224), float32, cuda:0>, t...float32, cuda:0>, tensor<(26,), int64, cuda:0>))]
        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs
# import torchsnooper
# @torchsnooper.snoop()
def my_random_pairs_of_minibatches(minibatches):
    #minibatches = [(tensor<(26, 3, 224, 224), float32, cuda:0>, te... float32, cuda:0>, tensor<(26,), int64, cuda:0>)]
    #peerm[2,1,4,0,3]
    all_x = minibatches[0]
    all_y = minibatches[1]
    disc_labels = minibatches[2]
    all_pd = minibatches[3]
    # (130, 3, 224, 224)
    num_domain=all_pd.max().item()+1
    perm = torch.randperm(num_domain).tolist()
    pairs = []
    for i in range(num_domain):
        j = i + 1 if i < (num_domain - 1) else 0

        index_i=torch.nonzero(all_pd == perm[i])
        index_j=torch.nonzero(all_pd==perm[j])
        xi,yi=all_x[index_i],all_y[index_i]
        xj,yj=all_x[index_j],all_y[index_j]
        min_n=min(xi.size(0),xj.size(0))
        min_n = min(len(xi), len(xj))
        # if xi[min_n:].size(0)==0:
        x_remain=xi[min_n:]
        y_remain=yi[min_n:]
        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n]),(x_remain,y_remain)))
    # print('done')
    return pairs

from sklearn.metrics import precision_score, recall_score

import torch
from sklearn.metrics import precision_score, recall_score,jaccard_score


def eval_dr(network, loader, weights, device, name=None, domain=None):
    true_labels = []
    predicted_labels = []
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x, domain=domain)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset: weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                predicted_labels += (p.gt(0).float() * batch_weights.view(-1, 1)).tolist()
            else:
                predicted_labels += (p.argmax(1).float() * batch_weights).tolist()
            true_labels += (y.float() * batch_weights).tolist()

    network.train()

    accuracy = torch.tensor(predicted_labels).eq(torch.tensor(true_labels)).sum().item() / len(predicted_labels)
    weighted_precision = precision_score(true_labels, predicted_labels, average='weighted')
    weighted_recall = recall_score(true_labels, predicted_labels, average='weighted')
    weighted_jaccard = jaccard_score(true_labels, predicted_labels, average='weighted')

    return accuracy, weighted_precision, weighted_recall,weighted_jaccard


def accuracy(network, loader, weights, device, name=None, domain=None):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()

    with torch.no_grad():
        for x, y in loader:

            x = x.to(device)
            y = y.to(device)
            p = network.predict(x, domain=domain)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct / total




import torch.nn as nn
# import torchsnooper
# @torchsnooper.snoop()
def eval_indomain(model, loader,valid_df,batch_size,weights, device,name,domain):

    model.eval()  # switch model to the evaluation mode
    val_preds = torch.zeros((len(valid_df), 2), dtype=torch.float32, device=device)
    val_preds_1d = torch.zeros((len(valid_df), 1), dtype=torch.float32, device=device)
    val_loader=loader
    criterion = nn.CrossEntropyLoss()
    val_labels = []
    weights_array=[]
    with torch.no_grad():  # Do not calculate gradient since we are only predicting
        # Predicting on validation set
        for j, batch in enumerate(val_loader):
            x_val, y_val=batch
            # x_val = torch.tensor(x_val, device=device, dtype=torch.float32)
            # y_val = torch.tensor(y_val, device=device, dtype=torch.float32)
            x_val=x_val.to(device)
            y_val=y_val.to(device)
            y_val = y_val.long()
            val_labels += (y_val.tolist())
            z_val= model.predict(x_val, domain=domain)
            # weights=torch.sum(all_bias,2)
            # weights=torch.sum(weights,0)
            # weights_array.append((weights))
            # print('all_bias',all_bias.size())
            # print('bias', all_bias)
            val_loss = criterion(z_val, y_val)
            _, val_pred_1d = torch.max(z_val.data, 1)
            val_pred_1d = torch.unsqueeze(val_pred_1d, 1)

            val_preds_1d[j * batch_size:j * batch_size + x_val.size(0)] = val_pred_1d
            val_preds[j * batch_size:j * batch_size + x_val.size(0)] = z_val
        # sum_weights=torch.zeros_like(weights)
        m=nn.Softmax(dim=-1)
        # for weights in weights_array:
        #     sum_weights=torch.add(sum_weights,weights)
        # # sum_weights=m(sum_weights)
        # sum_weights=sum_weights/torch.sum(sum_weights)
        val_labels=torch.tensor(val_labels)
        ACC, BACC, Prec, Rec, F1, AUC_ovo, SPEC, kappa =metrics.compute_isic_metrics_binary(val_labels, val_preds)
        model.train()

        return ACC, BACC, Prec, Rec, F1, AUC_ovo, SPEC, kappa,val_loss

class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)
