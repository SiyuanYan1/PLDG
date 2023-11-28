import numpy as np
import sklearn.metrics as metrics
from imblearn.metrics import sensitivity_score, specificity_score
import pdb
# from sklearn.metrics.ranking import roc_auc_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import torchsnooper
import torch.nn.functional as F
SCORE = specificity_score


def compute_isic_metrics(gt, pred):
    """
    :param gt: (batch,) torch tensor
    :param pred: (batch,class) torch tnesor
    :return:
    """
    # y = label_binarize(y, classes=[0, 1, 2])

    gt_np = gt.cpu().detach().numpy() #(602,1)
    gt_np = label_binarize(gt_np, classes=[0, 1, 2])
    pred_np = pred.cpu().detach().numpy() #(602,3)

    gt_class = np.argmax(gt_np, axis=1)
    # gt_class=gt_np
    pred_class = np.argmax(pred_np, axis=1)

    ACC = accuracy_score(gt_class, pred_class)
    BACC = balanced_accuracy_score(gt_class, pred_class) # balanced accuracy
    Prec = precision_score(gt_class, pred_class, average='macro')
    Rec = recall_score(gt_class, pred_class, average='macro')
    F1 = f1_score(gt_class, pred_class, average='macro')
    #(batch,class),(batch,class) one hot label vs prob
    AUC_ovo = metrics.roc_auc_score(gt_np, pred_np, average='macro', multi_class='ovo')
    # AUC_macro = metrics.roc_auc_score(gt_class, pred_np, average='macro', multi_class='ovo')

    # SPEC = SCORE(gt_class, pred_class, average='macro')
    SPEC = SCORE(gt_class, pred_class, average=None)

    kappa = cohen_kappa_score(gt_class, pred_class, weights='quadratic')

    print(confusion_matrix(gt_class, pred_class))

    return ACC, BACC, Prec, Rec, F1, AUC_ovo, SPEC, kappa
# import torchsnooper
# @torchsnooper.snoop()
def compute_isic_metrics_binary(gt, pred):
    """
    :param gt: (batch,) torch tensor
    :param pred: (batch,class) torch tnesor
    :return:
    """
    # y = label_binarize(y, classes=[0, 1, 2])

    gt_np = gt.cpu().detach().numpy() #(602,1)
    gt_np = label_binarize(gt_np, classes=[0, 1, 2])
    pred_np = pred.cpu().detach().numpy() #(602,3)

    gt_class = np.argmax(gt_np, axis=1)
    # gt_class=gt_np
    pred_class = np.argmax(pred_np, axis=1)

    ACC = accuracy_score(gt_class, pred_class)
    BACC = balanced_accuracy_score(gt_class, pred_class) # balanced accuracy
    Prec = precision_score(gt_class, pred_class, average='macro')
    Rec = recall_score(gt_class, pred_class, average='macro')
    F1 = f1_score(gt_class, pred_class, average='macro')

    AUC=metrics.roc_auc_score(gt_class,F.softmax(pred, dim=1).cpu().data.numpy()[:,1])
    #(batch,class),(batch,class) one hot label vs prob
    # AUC = metrics.roc_auc_score(gt_np, pred_np, average='macro', multi_class='ovo')
    # AUC_ovo = metrics.roc_auc_score(gt_np, pred_np, average='macro', multi_class='ovo')
    # AUC_macro = metrics.roc_auc_score(gt_class, pred_np, average='macro', multi_class='ovo')
    # SPEC = SCORE(gt_class, pred_class, average='macro')
    SPEC = SCORE(gt_class, pred_class, average=None)

    kappa = cohen_kappa_score(gt_class, pred_class, weights='quadratic')

    # print(confusion_matrix(gt_class, pred_class))

    return ACC, BACC, Prec, Rec, F1, AUC, SPEC, kappa