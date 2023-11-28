import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
from torch.utils.data import DataLoader
from domainbed.clustering import clustering
from scipy.optimize import linear_sum_assignment
import os
import numpy as np
import pickle
# import gzip, cPickle
from tsne import bh_sne
import matplotlib.pyplot as plt
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C)
    return feat_mean, feat_std

def reassign(y_before, y_pred):
    assert y_before.size == y_pred.size
    D = max(y_before.max(), y_pred.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_before.size):
        w[y_before[i], y_pred[i]] += 1
    row_ind, col_ind= linear_sum_assignment(w.max() - w)
    return col_ind

def compute_features(dataloader, model, N, device):
    model.eval()
    # discard the label information in the dataloader
    for i, (input_tensor, _, _) in enumerate(dataloader):
        with torch.no_grad():
            input_var = input_tensor.to(device)
            aux = model.domain_features(input_var).data.cpu().numpy()
            if i == 0:
                features = np.zeros((N, aux.shape[1])).astype('float32')

            if i < len(dataloader) - 1:
                features[i * dataloader.batch_size: (i + 1) * dataloader.batch_size] = aux.astype('float32')
            else:
                # special treatment for final batch
                features[i * dataloader.batch_size:] = aux.astype('float32')
    return features
# import torchsnooper
# @torchsnooper.snoop()
def compute_instance_stat(dataloader, model, N, device):
    model.network.eval()

    for i, (input_tensor, _, _,_) in enumerate(dataloader):
        with torch.no_grad():
            #(128, 3, 224, 224)
            input_var = input_tensor.to(device)
            #(12,128,197,768) 12 encoder blocks
            feature_map = model.featurizer.get_feature_from_input(input_var)
            cls = feature_map[1][:, 0, :]  # (128,768)  #the clas token from the last encoder block of VIT
            aux = cls.data.cpu().numpy()
            if i == 0:
                #(128, 128, 28, 28),
                features = np.zeros((N, aux.shape[1])).astype('float32')
            if i < len(dataloader) - 1:
                features[i * dataloader.batch_size: (i + 1) * dataloader.batch_size] = aux.astype('float32')
            else:
                # special treatment for final batch
                features[i * dataloader.batch_size:] = aux.astype('float32')
    #(8321,384)
    #(12343, 768)
    return features
# import torchsnooper
# @torchsnooper.snoop()
def arrange_clustering(images_lists):
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    indexes = np.argsort(image_indexes)
    return np.asarray(pseudolabels)[indexes]
#

# import torchsnooper
# @torchsnooper.snoop()
def domain_split_dr(dataset, model, device, cluster_before, filename, epoch, nmb_cluster=3, method='Kmeans', pca_dim=256, batchsize=128, num_workers=4, whitening=False, L2norm=False, instance_stat=True,exp_name=None):
    #kmeans,
    cluster_method = clustering.__dict__[method](nmb_cluster, pca_dim, whitening, L2norm)

    dataset.set_transform('val')
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=num_workers)

    if instance_stat:
        # (12343, 768)
        features = compute_instance_stat(dataloader, model, len(dataset), device)

    else:
        # (12343, 768)
        features = compute_features(dataloader, model, len(dataset), device)
    #none
    clustering_loss = cluster_method.cluster(features, verbose=False)
    #cluster_method.images_lists: image index 1-8261
    #(8321)
    cluster_list = arrange_clustering(cluster_method.images_lists)
    #cluster_list:same length, index of clustering
    save_tsne=False
    if save_tsne==True and (epoch==5 or epoch==20):
        assigned_target=np.array(cluster_list).reshape(-1,1) #8321
        out_features=np.array(features)  #8321,384
        true_target=np.array(dataloader.dataset.domains)
        print(true_target)

        print('epoch is {}, saving generated feature'.format(epoch))
        output_2d = bh_sne(out_features.astype(float))
        plt.rcParams['figure.figsize'] = 20, 20
        plt.scatter(output_2d[:, 0], output_2d[:, 1],c=assigned_target)
        # sns.scatterplot(output_2d[:, 0], output_2d[:, 1], hue=y, legend='full', palette=palette)
        fig='/mount/neuron/Lamborghini/dir/pythonProject/TMI/latent_prompt/results/'+exp_name+'/'+str(epoch)+'.png'
        plt.savefig(fig, bbox_inches='tight')
        plt.cla()

    class_nmi = normalized_mutual_info_score(
        cluster_list, dataloader.dataset.labels, average_method='geometric')
    domain_nmi = normalized_mutual_info_score(
        cluster_list, dataloader.dataset.domains, average_method='geometric')
    before_nmi = normalized_mutual_info_score(
        cluster_list, cluster_before, average_method='arithmetic')
    
    log = 'Epoch: {}, NMI against class labels: {:.3f}, domain labels: {:.3f}, previous assignment: {:.3f}'.format(epoch, class_nmi, domain_nmi, before_nmi)

    if filename:
        with open(filename, 'a') as f:
            f.write(log + '\n')
        
    mapping = reassign(cluster_before, cluster_list)
    cluster_reassign = [cluster_method.images_lists[mapp] for mapp in mapping]
    dataset.set_transform(dataset.split)
    return arrange_clustering(cluster_reassign)
