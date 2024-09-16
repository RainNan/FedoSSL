"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import copy
from models.criterion import AngularPenaltySMLoss
# from criterion import AngularPenaltySMLoss
import sys

__all__ = ['resnet18']
# Sinkhorn Knopp
# 聚类算法
def sknopp(cZ, lamd=25, max_iters=100):
    with torch.no_grad():
        N_samples, N_centroids = cZ.shape # cZ is [N_samples, N_centroids]
        probs = F.softmax(cZ * lamd, dim=1).T # probs should be [N_centroids, N_samples]

        r = torch.ones((N_centroids, 1), device=probs.device) / N_centroids # desired row sum vector
        c = torch.ones((N_samples, 1), device=probs.device) / N_samples # desired col sum vector

        inv_N_centroids = 1. / N_centroids
        inv_N_samples = 1. / N_samples

        err = 1e3
        for it in range(max_iters):
            r = inv_N_centroids / (probs @ c)  # (N_centroids x N_samples) @ (N_samples, 1) = N_centroids x 1
            c_new = inv_N_samples / (r.T @ probs).T  # ((1, N_centroids) @ (N_centroids x N_samples)).t() = N_samples x 1
            if it % 10 == 0:
                err = torch.nansum(torch.abs(c / c_new - 1))
            c = c_new
            if (err < 1e-2):
                break

        # inplace calculations.
        probs *= c.squeeze()
        probs = probs.T # [N_samples, N_centroids]
        probs *= r.squeeze()

        return probs * N_samples # Soft assignments


class BasicBlock(nn.Module):
    expansion = 1

    # in_planes 输入的特征图的通道数
    # planes 通常表示 残差块（Residual Block）中卷积操作后输出的基础通道数
    # planes 代表了当前残差块中希望得到的输出特征图的通道数，但并不是最终的通道数，最终的输出通道数还需要乘以 expansion（扩展系数）来得到
    # stride 步幅
    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last

        # 两个 3x3 的卷积层，每层都接上批量归一化 Batch Normalization

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # 【跳跃连接（shortcut  connection）】是 ResNet 中的核心概念之一
        # 允许输入直接绕过若干层网络，并加到输出上，从而形成残差学习
        # Sequential 恒等映射，即输入直接加到输出上，不做任何改变
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        # 对输入 x 进行一次卷积操作 (self.conv1(x))
        # 然后将结果通过批量归一化（self.bn1）处理
        # 最后通过 ReLU 激活函数。激活函数用于引入非线性
        out = F.relu(self.bn1(self.conv1(x)))

        # 对上一步的输出 out 进行第二次卷积操作（self.conv2）
        # 随后再进行批量归一化（self.bn2）
        # 此时，out 已经经过两次卷积和一次 ReLU 激活
        out = self.bn2(self.conv2(out))

        # 跳跃连接（shortcut  connection）
        # 将输入 x 加到当前的输出 out 上，这就是所谓的 【残差连接】或【恒等映射】
        # 这一步的意义是让网络能够直接学习 输入与输出的差异，而不是让网络学习整个输入到输出的映射，这样可以缓解深层网络的训练问题
        out += self.shortcut(x)
        preact = out

        # 第二次 ReLU 激活
        out = F.relu(out)

        # 判断是否为最后一个块
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last

        # Bottleneck 包含三个卷积层
        # 一个 1x1 卷积层，用于减少维度
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 一个 3x3 卷积层，用于在低维度空间中进行卷积操作
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 一个 1x1 卷积层，用于恢复维度
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out
########################################################################################################################
'''
FedoSSL 的 ResNet 结构：
一个3x3卷积层（输入通道数64，输出64，）
一个批量归一化层
'''
# ResNet = BasicBlock + Bottleneck
# ResNet18 就用一个 BasicBlock
class ResNet(nn.Module):

    # block 残差块类型
    # num_blocks 列表，表示每个层中残差块的数量
    # in_channel 输入图片的通道数，默认为 3（RGB 图像）
    # zero_init_residual 是否将残差块中的最后一个批量归一化层初始化为 0
    def __init__(self, block, num_blocks, num_classes=100, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # 【卷积层】
        # 输入图片首先经过一个 3x3 的卷积层（self.conv1），输入通道数为 in_channel，输出通道数为 64
        # 不使用偏置，因为批量归一化层已经提供了偏移功能
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # 【残差层】
        # ResNet 的四个主要卷积层，每个层由多个残差块组成。每一层的输出通道数依次为 64、128、256 和 512
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # 【自适应平均池化层】
        # 将特征图的尺寸缩放到 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 【线性分类层】
        # 全连接层，用于根据类别数进行分类
        self.linear = NormedLinear(512 * block.expansion, num_classes)

        ################################################################################################################
        '''
        质心（centroids）部分，包括 全局质心 和 局部质心，局部质心又分成了 有标签数据质心 和 无标签数据质心
        有标签数据质心的计算：基于每个数据类别的特征表示进行的聚类操作
        无标签数据质心的计算：使用了无监督的聚类方法 Sinkhorn Knopp
        '''
        # 局部质心的数量
        self.N_local = 32
        # 特征投影
        # 降低特征的维度 1024 变成 512
        self.mem_projections = nn.Linear(1024, 512, bias=False) # para1: Memory size per client
        #self.centroids = NormedLinear(512 * block.expansion, num_classes) # global cluster centroids

        # 全局质心
        # 输出的维度是类别数量（cifar是10）
        self.centroids = nn.Linear(512 * block.expansion, num_classes, bias=False)  # global cluster centroids
        # 局部质心
        # 每个客户端会生成 32 个簇
        self.local_centroids = nn.Linear(512 * block.expansion, self.N_local, bias=False)  # must be defined last
        # self.global_labeled_centroids = nn.Linear(512 * block.expansion, 10, bias=False)  # labeled data feature centroids

        # 局部标记数据的质心
        self.local_labeled_centroids = nn.Linear(512 * block.expansion, num_classes, bias=False) # labeled data feature centroids
        # T 温度参数
        # 通常用于自监督学习中的对比学习（contrastive learning）任务
        # 温度参数用于缩放样本间的相似度度量，它控制了模型在计算相似度时的敏感性
        self.T = 0.1
        # 带标签的数据样本数为 6（cifar10）
        self.labeled_num = 6
        ################################################################################################################

        # 初始化
        # 遍历所有的子模块，包括卷积层、批量归一化层、线性层等
        for m in self.modules():
            # 当前模块是否是 nn.Conv2d 卷积层
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # 权重
                nn.init.constant_(m.weight, 1)
                # 偏置
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677

        # zero_init_residual 是否将残差块中的最后一个批量归一化层初始化为 0
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    # 构造函数结束

    # @torch.no_grad() 是一个装饰器
    # 表示这个方法在执行过程中不会计算梯度，也不会影响模型的自动求导机制。这通常用于不需要反向传播的部分，比如模型的推理或参数更新
    @torch.no_grad()
    # 更新 【特征记忆】
    def update_memory(self, F):
        # F.shape[0] 表示特征张量的第一个维度大小，也就是批量大小（N），即当前输入数据所生成的特征数
        N = F.shape[0]
        # Shift memory [D, m_size]
        self.mem_projections.weight.data[:, :-N] = self.mem_projections.weight.data[:, N:].detach().clone()
        # Transpose LHS [D, bsize]
        self.mem_projections.weight.data[:, -N:] = F.T.detach().clone()

    # Local clustering (happens at the client after every training round; clusters are made equally sized via Sinkhorn-Knopp, satisfying K-anonymity)
    # 局部聚类(发生在每一轮训练后的客户端;通过满足k -匿名性的Sinkhorn-Knopp，簇的大小相等）
    # 局部聚类是基于 Sinkhorn-Knopp 算法的迭代过程
    # 主要通过计算簇中心和样本之间的相似性，逐步更新簇中心
    # 这种聚类仅在客户端本地执行，并更新局部簇中心以反映客户端的数据分布
    def local_clustering(self, device=torch.device("cuda")):
        # Local centroids: [# of centroids, D]; local clustering input (mem_projections.T): [m_size, D]
        with torch.no_grad():
            Z = self.mem_projections.weight.data.T.detach().clone()
            centroids = Z[np.random.choice(Z.shape[0], self.N_local, replace=False)]
            local_iters = 5
            # clustering
            for it in range(local_iters):
                assigns = sknopp(Z @ centroids.T, max_iters=10)
                choice_cluster = torch.argmax(assigns, dim=1)
                for index in range(self.N_local):
                    selected = torch.nonzero(choice_cluster == index).squeeze()
                    selected = torch.index_select(Z, 0, selected)
                    if selected.shape[0] == 0:
                        selected = Z[torch.randint(len(Z), (1,))]
                    centroids[index] = F.normalize(selected.mean(dim=0), dim=0)

        # Save local centroids
        self.local_centroids.weight.data.copy_(centroids.to(device))

    # Global clustering (happens only on the server; see Genevay et al. for full details on the algorithm)
    # def global_clustering(self, Z1, nG=1., nL=1.):
    #     N = Z1.shape[0] # Z has dimensions [m_size * n_clients, D]
    #     # Optimizer setup
    #     optimizer = torch.optim.SGD(self.centroids.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    #     train_loss = 0.
    #     total_rounds = 500
    #     for round_idx in range(total_rounds):
    #         with torch.no_grad():
    #             # Cluster assignments from Sinkhorn Knopp
    #             SK_assigns = sknopp(self.centroids(Z1))
    #         # Zero grad
    #         optimizer.zero_grad()
    #         # Predicted cluster assignments [N, N_centroids] = local centroids [N, D] x global centroids [D, N_centroids]
    #         probs1 = F.softmax(self.centroids(F.normalize(Z1, dim=1)) / self.T, dim=1)
    #         # Match predicted assignments with SK assignments
    #         loss = - F.cosine_similarity(SK_assigns, probs1, dim=-1).mean()
    #         # Train
    #         loss.backward()
    #         optimizer.step()
    #         with torch.no_grad():
    #             #self.centroids.weight.copy_(self.centroids.weight.data.clone()) # Not Normalize centroids
    #             self.centroids.weight.copy_(F.normalize(self.centroids.weight.data.clone(), dim=1)) # Normalize centroids
    #             train_loss += loss.item()
    #     ######
    ###
    # Global clustering (happens only on the server; see Genevay et al. for full details on the algorithm)
    def global_clustering(self, Z1, nG=1., nL=1.):
        N = Z1.shape[0] # Z has dimensions [m_size * n_clients, D]
        # Optimizer setup
        optimizer = torch.optim.SGD(self.centroids.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        train_loss = 0.
        total_rounds = 500
        angular_criterion = AngularPenaltySMLoss()
        for round_idx in range(total_rounds):
            with torch.no_grad():
                # Cluster assignments from Sinkhorn Knopp
                SK_assigns = sknopp(self.centroids(Z1))
            # Zero grad
            optimizer.zero_grad()
            # Predicted cluster assignments [N, N_centroids] = local centroids [N, D] x global centroids [D, N_centroids]
            probs1 = F.softmax(self.centroids(F.normalize(Z1, dim=1)) / self.T, dim=1)
            ## 增加 Prototype距离 ##
            # cos_output = self.centroids(F.normalize(Z1, dim=1))
            # SK_target = np.argmax(SK_assigns.cpu().numpy(), axis=1)
            # angular_loss = angular_criterion(cos_output, SK_target)
            # print("angular_loss: ", angular_loss)
            ######################
            # Match predicted assignments with SK assignments
            cos_loss = F.cosine_similarity(SK_assigns, probs1, dim=-1).mean()
            loss = - cos_loss #+ angular_loss
            print("F.cosine_similarity: ", cos_loss)
            # Train
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                #self.centroids.weight.copy_(self.centroids.weight.data.clone()) # Not Normalize centroids
                self.centroids.weight.copy_(F.normalize(self.centroids.weight.data.clone(), dim=1)) # Normalize centroids
                train_loss += loss.item()
        #sys.exit(0)
        ######
    ###
    # 这段代码的功能是基于标记数据的特征中心来更新模型的全局簇中心。其目的是将标记数据的特征中心与全局簇中心进行【对齐】
    def set_labeled_feature_centroids(self, device=torch.device("cuda")):
        assignment = [999 for _ in range(self.labeled_num)]
        not_assign_list = [i for i in range(10)]
        # 让labeled feature中心点替换 最接近的 self.centroids参数 #
        C = self.centroids.weight.data.detach().clone() # [10,512]
        labeled_feature_centroids = self.local_labeled_centroids.weight.data[:self.labeled_num].detach().clone() # [labeled_num,512]
        copy_labeled_feature_centroids = copy.deepcopy(labeled_feature_centroids)
        copy_C = copy.deepcopy(C)
        #
        # C_norm = C / torch.norm(C, 2, 1, keepdim=True) #采用欧氏距离时，C此时没有归一化
        C_norm = C
        labeled_norm = labeled_feature_centroids / torch.norm(labeled_feature_centroids, 2, 1, keepdim=True)
        cosine_dist = torch.mm(labeled_norm, C_norm.t()) # [labeled_num, 10]
        vals, pos_idx = torch.topk(cosine_dist, 2, dim=1)
        pos_idx_1 = pos_idx[:, 0].cpu().numpy().flatten().tolist() # top1 [labeled_num]
        pos_idx_2 = pos_idx[:, 1].cpu().numpy().flatten().tolist() # top2 [labeled_num]
        print("cosine_dist: ", cosine_dist)
        print("pos_idx: ", pos_idx)
        print("pos_idx_1: ", pos_idx_1)
        print("pos_idx_2: ", pos_idx_2)
        #
        for idx in range(self.labeled_num):
            if pos_idx_1[idx] not in assignment:
                assignment[idx] = pos_idx_1[idx]
                not_assign_list.remove(assignment[idx])
                #C[assignment[idx]] = labeled_feature_centroids[idx]
            else:
                assignment[idx] = pos_idx_2[idx]
                not_assign_list.remove(assignment[idx])
                #C[assignment[idx]] = labeled_feature_centroids[idx]
        # set labeled centroids at first
        for idx in range(10):
            if idx < self.labeled_num:
                # C[idx] = copy_labeled_feature_centroids[idx]  ##### use avg label data feature centroids
                C[idx] = copy_C[assignment[idx]] ##### use cluster centroids #####
                # C[idx] = labeled_norm[idx] * torch.norm(copy_C[assignment[idx]], 2)  ##### 用label data中心 加上 归一化
                # C[idx] = copy_C[assignment[idx]] * torch.norm(copy_labeled_feature_centroids[idx] , 2)#采用欧氏距离时，global中心不用归一化
                # if idx == 0:
                #     avg_norm = torch.norm(copy_labeled_feature_centroids[idx] , 2)
                # else:
                #     avg_norm = avg_norm + torch.norm(copy_labeled_feature_centroids[idx] , 2)
            else:
                C[idx] = copy_C[not_assign_list[idx - self.labeled_num]]
                # C[idx] = copy_C[not_assign_list[idx - self.labeled_num]] * (avg_norm/self.labeled_num)#采用欧氏距离时，global中心不用归一化
        #
        self.centroids.weight.data.copy_(C.to(device))
        # self.centroids.weight.data.copy_(F.normalize(C.to(device), dim=1))
        # return -1
    ###

    # 在 ResNet 结构中构建一层包含多个残差块（residual block）的【卷积层】
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out_linear = self.linear(out)
        #
        tZ1 = F.normalize(out, dim=1)
        # Update target memory
        with torch.no_grad():
            self.update_memory(tZ1) # New features are [bsize, D]
        #
        return out_linear, out

# 可变参数 **kwargs 允许将任意数量的关键字参数传递给函数
def resnet18(**kwargs):
    # 使用了 BasicBlock 作为基本构建块，并且有特定的层数分布
    # [2, 2, 2, 2]：这是一个列表，定义了每个层级中残差块的数量。在 ResNet18 中，有 4 个层级，每个层级包含 2 个残差块。
    # 第一个 2 表示第一个层级（conv2_x）有 2 个 BasicBlock。
    # 第二个 2 表示第二个层级（conv3_x）有 2 个 BasicBlock。
    # 依次类推，最后一层（conv5_x）也有 2 个 BasicBlock。
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return 10 * out