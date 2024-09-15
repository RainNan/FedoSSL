from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torchvision
import torch.utils.data as data
from torchvision import transforms
import itertools
from torch.utils.data.sampler import Sampler
from torchvision.datasets import ImageFolder

from pytorch_cinic.dataset import CINIC10

from typing import Optional, Callable, Tuple, Any


class OPENWORLDCIFAR100(torchvision.datasets.CIFAR100):

    def __init__(self, root, labeled=True, labeled_num=50, labeled_ratio=0.5, rand_number=0, transform=None,
                 target_transform=None,
                 download=False, unlabeled_idxs=None):
        super(OPENWORLDCIFAR100, self).__init__(root, True, transform, target_transform, download)

        downloaded_list = self.train_list
        self.data = []
        self.targets = []
        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        labeled_classes = range(labeled_num)
        np.random.seed(rand_number)

        if labeled:
            self.labeled_idxs, self.unlabeled_idxs = self.get_labeled_index(labeled_classes, labeled_ratio)
            self.shrink_data(self.labeled_idxs)
        else:
            self.shrink_data(unlabeled_idxs)

    def get_labeled_index(self, labeled_classes, labeled_ratio):
        labeled_idxs = []
        unlabeled_idxs = []
        for idx, label in enumerate(self.targets):
            if label in labeled_classes and np.random.rand() < labeled_ratio:
                labeled_idxs.append(idx)
            else:
                unlabeled_idxs.append(idx)
        return labeled_idxs, unlabeled_idxs

    def shrink_data(self, idxs):
        targets = np.array(self.targets)
        self.targets = targets[idxs].tolist()
        self.data = self.data[idxs, ...]


# 1.对cifar10进行划分，有标签的分5个类，剩下5个类为无标签
# 2.按比例划分有无标签数据
# 继承自 PyTorch 中 torchvision.datasets.CIFAR10
class OPENWORLDCIFAR10(torchvision.datasets.CIFAR10):

    def __init__(self,
                 root,     # root: 数据存储的路径
                 labeled=True,     # labeled: 是否使用标注数据
                 labeled_num=5,     # labeled_num：多少类数据被认为是“标记的”。例如，设置为 5，则意味着使用前 5 类作为标记数据
                 labeled_ratio=0.5,     # labeled_ratio: 标记数据的比例
                 rand_number=0,     # rand_number: 随机数种子
                 transform=None,    # transform 和 target_transform: 数据变换，通常用于数据增强等任务
                 target_transform=None,
                 download=False,    # download: 是否需要下载数据集
                 unlabeled_idxs=None):    # unlabeled_idxs: 如果 labeled 为 False，则使用这个索引列表来选择未标记的数据

        # 调用父类的 __init__ 方法，初始化 CIFAR10 数据集的基本参数。
        super(OPENWORLDCIFAR10, self).__init__(root,
                                               True,
                                               transform,
                                               target_transform,
                                               download)

        # 训练集 train_list
        downloaded_list = self.train_list
        self.data = []  # 数据
        self.targets = []  # 标签
        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        # print("BEFORE cifar10 self.data[0].shape: ", self.data[0].shape)

        # 将加载的图像数据调整为适合处理的形状
        # CIFAR-10 的原始形状是 (num_samples, 3, 32, 32)这是 PyTorch 的默认形状（通道优先）
        # 这里将其转换为 (num_samples, 32, 32, 3)，这是常用的图像格式（高度、宽度、通道）
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        # print("AFTER cifar10 self.data[0].shape: ", self.data[0].shape)

        # 将要标记的类别定义为前 labeled_num 个类别
        labeled_classes = range(labeled_num)
        # 使用 rand_number 设置随机种子，以保证每次运行时数据划分的一致性
        np.random.seed(rand_number)

        # shrink_data 根据传入的索引，缩小 self.data 和 self.targets 的范围。只保留指定索引的样本，用于后续的训练。
        if labeled:
            self.labeled_idxs, self.unlabeled_idxs = self.get_labeled_index(labeled_classes, labeled_ratio)
            self.shrink_data(self.labeled_idxs)
        else:
            self.shrink_data(unlabeled_idxs)

    # 遍历所有标签，对于属于标记类别的样本，按照设定的标记比例随机选择是否标记
    def get_labeled_index(self, labeled_classes, labeled_ratio):
        labeled_idxs = []
        unlabeled_idxs = []
        for idx, label in enumerate(self.targets):
            if label in labeled_classes and np.random.rand() < labeled_ratio:
                labeled_idxs.append(idx)
            else:
                unlabeled_idxs.append(idx)
        return labeled_idxs, unlabeled_idxs

    # 根据传入的索引，缩小 self.data 和 self.targets 的范围。只保留指定索引的样本，用于后续的训练。
    def shrink_data(self, idxs):
        targets = np.array(self.targets)
        self.targets = targets[idxs].tolist()
        self.data = self.data[idxs, ...]


class OPENWORLDCINIC10(CINIC10):

    def __init__(self, root, labeled=True, labeled_num=5, labeled_ratio=0.5, rand_number=0, transform=None,
                 target_transform=None,
                 download=False, unlabeled_idxs=None):
        super(OPENWORLDCINIC10, self).__init__(root, "train", transform, target_transform, download)

        self.partition = "train"
        # self.root = root

        # if download:
        #     self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self.origin_data = ImageFolder(os.path.join(root, self.partition))
        self.data = []
        self.targets = []
        for data_target in self.origin_data:
            self.data.append(np.array(data_target[0]))
            self.targets.append(data_target[1])

        self.data = np.array(self.data)

        labeled_classes = range(labeled_num)
        np.random.seed(rand_number)

        if labeled:
            self.labeled_idxs, self.unlabeled_idxs = self.get_labeled_index(labeled_classes, labeled_ratio)
            self.shrink_data(self.labeled_idxs)
        else:
            self.shrink_data(unlabeled_idxs)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def get_labeled_index(self, labeled_classes, labeled_ratio):
        labeled_idxs = []
        unlabeled_idxs = []
        for idx, label in enumerate(self.targets):
            if label in labeled_classes and np.random.rand() < labeled_ratio:
                labeled_idxs.append(idx)
            else:
                unlabeled_idxs.append(idx)
        return labeled_idxs, unlabeled_idxs

    def shrink_data(self, idxs):
        targets = np.array(self.targets)
        self.targets = targets[idxs].tolist()
        # print("image size: ", self.data[0].size)
        # print("channel size: ", len(self.data[0].split()))
        tmp = np.array(self.data[0])
        # print("image to array shape: ", tmp.shape)
        self.data = self.data[idxs]
        # self.data = self.data[idxs, ...]


# Dictionary of transforms
# 数据集预处理
dict_transform = {
    # 训练集
    # 经过随机裁剪和水平翻转来进行数据增强，这些操作有助于生成更多的样本变体，提升模型的泛化能力
    'cifar_train': transforms.Compose([
        # 在图片周围添加 4 像素的填充，然后从中随机裁剪 32x32 的图像。这种操作可以增强模型的鲁棒性，模拟不同的图像边缘。
        transforms.RandomCrop(32, padding=4),
        # 随机水平翻转图像，进一步增加数据的多样性，有助于避免模型的【过拟合】
        transforms.RandomHorizontalFlip(),
        # 将图像从 PIL 格式或 numpy 数组转换为 PyTorch 的张量格式，同时将像素值归一化到 [0, 1] 范围
        transforms.ToTensor(),
        # 对图像进行标准化，分别使用每个通道（RGB）的均值 (0.5071, 0.4867, 0.4408) 和标准差 (0.2675, 0.2565, 0.2761)
        # 标准化是常见的图像预处理操作，有助于提升模型的训练稳定性和效果
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]),
    # 测试集
    # 没有数据增强操作，仅做了标准的张量转换和归一化，以保证测试集的分布一致性
    'cifar_test': transforms.Compose([
        # 将图像转换为张量
        transforms.ToTensor(),
        # 标准化
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
}
