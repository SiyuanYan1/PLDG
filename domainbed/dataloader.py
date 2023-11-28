from torch.utils.data import DataLoader, random_split
import numpy as np
from copy import deepcopy
from domainbed.datasets import DG_Dataset
import torchsnooper
from domainbed.lib.fast_data_loader import InfiniteDataLoader

# @torchsnooper.snoop()
def my_dataloader(dataset,  batch_size,num_workers=8,get_domain_label=True,get_cluster=False):

    # source = (img,label,domain label,pseudo). initialize using 0
    source_train=dataset
    source_train = deepcopy(source_train)

    source_train.split = 'train'
    source_train.set_transform('train')
    source_train.get_domain_label = get_domain_label
    source_train.get_cluster = get_cluster

    # (img,labwel,domain label)
    source_train=InfiniteDataLoader(dataset=source_train,weights=None,batch_size=batch_size,num_workers=num_workers)
    return source_train