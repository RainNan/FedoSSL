import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment
import os
import os.path
import torch.nn.functional as F


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


class AverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target):
    num_correct = np.sum(output == target)
    res = num_correct / len(target)

    return res


def cluster_acc(y_pred, y_true):

    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """

    if y_pred.size == 0 or y_true.size == 0:
        return 0.0  # 如果任何一个数组为空，返回0.0作为准确率

    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    return w[row_ind, col_ind].sum() / y_pred.size


# def cluster_acc_w(y_pred, y_true):
#     """
#     Calculate clustering accuracy. Require scikit-learn installed
#     # Arguments
#         y: true labels, numpy.array with shape `(n_samples,)`
#         y_pred: predicted labels, numpy.array with shape `(n_samples,)`
#     # Return
#         accuracy, in [0,1]
#     """
#
#     if y_pred.size == 0 or y_true.size == 0:
#         return 0.0, w  # 如果任何一个数组为空，返回0.0作为准确率
#
#     y_true = y_true.astype(np.int64)
#     assert y_pred.size == y_true.size
#     D = max(y_pred.max(), y_true.max()) + 1
#     # print("y_pred.max(): ", y_pred.max())
#     w = np.zeros((D, D), dtype=np.int64)
#
#
#
#     for i in range(y_pred.size):
#         w[y_pred[i], y_true[i]] += 1
#     row_ind, col_ind = linear_sum_assignment(w.max() - w)
#
#     return w[row_ind, col_ind].sum() / y_pred.size, w

def cluster_acc_w(y_pred, y_true):
    # 使用 线性分配算法 来计算聚类的加权准确率
    """
    Calculate clustering accuracy. Require scikit-learn installed.
    # Arguments
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        y_true: true labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    # 检查输入数组是否为空
    if len(y_pred) == 0 or len(y_true) == 0:
        return 0.0, None

    # 尝试将标签转换为整数
    y_pred_int = np.array(y_pred, dtype=int)
    y_true_int = np.array(y_true, dtype=int)

    # 确保转换后的数组与原始数组长度相同
    if len(y_pred_int) != len(y_pred) or len(y_true_int) != len(y_true):
        raise ValueError("All elements in y_pred and y_true must be convertible to integers")

    # 计算混淆矩阵的大小
    D = max(max(y_pred_int), max(y_true_int)) + 1
    w = np.zeros((D, D), dtype=np.int64)

    # 填充混淆矩阵
    for i in range(len(y_pred)):
        w[y_pred_int[i], y_true_int[i]] += 1

    # 计算最佳匹配
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    # 计算加权准确率
    correct_matches = w[row_ind, col_ind].sum()
    total_matches = np.sum(w)
    accuracy = correct_matches / total_matches if total_matches > 0 else 0.0

    return accuracy, w


def entropy(x):
    """ 
    Helper function to compute the entropy over the batch 
    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """
    EPS = 1e-8
    x_ = torch.clamp(x, min=EPS)
    b = x_ * torch.log(x_)

    if len(b.size()) == 2:  # Sample-wise entropy
        return - b.sum(dim=1).mean()
    elif len(b.size()) == 1:  # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' % (len(b.size())))


class MarginLoss(nn.Module):

    def __init__(self, m=0.2, weight=None, s=10):
        super(MarginLoss, self).__init__()
        self.m = m
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        x_m = x - self.m * self.s

        output = torch.where(index, x_m, x)
        return F.cross_entropy(output, target, weight=self.weight)
