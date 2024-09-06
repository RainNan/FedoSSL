import sys
import copy
import argparse
import warnings
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models
import open_world_cifar as datasets
import client_open_world_cifar as client_datasets
from utils import cluster_acc, AverageMeter, entropy, MarginLoss, accuracy, TransformTwice, cluster_acc_w
from sklearn import metrics
import numpy as np
import os
# from utils_cluster import
from torch.utils.tensorboard import SummaryWriter
from itertools import cycle


# 没用到
# def euclidean_dist(x, y):
#     m, n = x.size(0), y.size(0)
#     xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
#     yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
#     dist = xx + yy
#     dist.addmm_(1, -2, x, y.t())
#     dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
#     return dist

# 接收并处理客户端模型的参数
def receive_models(clients_model):
    global uploaded_weights
    global uploaded_models
    uploaded_weights = []
    uploaded_models = []
    for model in clients_model:
        uploaded_weights.append(1.0 / len(clients_model))
        # self.uploaded_models.append(copy.deepcopy(client.model.parameters()))
        uploaded_models.append(model.parameters())


# 按权重聚合 客户端模型的参数 到 全局模型 中
def add_parameters(w, client_model):
    for (name, server_param), client_param in zip(global_model.named_parameters(), client_model):
        if "centroids" not in name:
            server_param.data += client_param.data.clone() * w
        if "local_labeled_centroids" in name:
            server_param.data += client_param.data.clone() * w
            # print("Averaged layer name: ", name)


# 初始化全局模型的参数为零，然后通过调用 add_parameters() 函数聚合客户端的模型参数
def aggregate_parameters():
    for name, param in global_model.named_parameters():
        if "centroids" not in name:
            param.data = torch.zeros_like(param.data)
        if "local_labeled_centroids" in name:
            param.data = torch.zeros_like(param.data)
            # print("zeros_liked layer name: ", name)
    for w, client_model in zip(uploaded_weights, uploaded_models):
        add_parameters(w, client_model)


def train(args, model, device, train_label_loader, train_unlabel_loader, optimizer, m, epoch, tf_writer, client_id,
          global_round):
    # 初始化模型中的局部标记聚类中心为零。
    model.local_labeled_centroids.weight.data.zero_()  # model.local_labeled_centroids.weight.data: torch.Size([10, 512])

    # 记录【每个类别】的标记样本数
    # [0 for _ in range(10)] 等价于 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # 为什么是10个：因为 CIFAR-10 有 10 个类别
    labeled_samples_num = [0 for _ in range(10)]

    # 将模型设置为“训练模式”
    # 在训练模式下，模型中的一些特定层（例如 Dropout 和 BatchNorm）会以训练时的方式工作
    # 它们的行为与评估模式（即 model.eval()）不同
    model.train()

    # 使用二元交叉熵损失
    bce = nn.BCELoss()

    m = min(m, 0.5)
    # m = 0
    # 使用带有余量的 交叉熵损失函数
    ce = MarginLoss(m=-1 * m)

    # 未标记数据的交叉熵损失
    unlabel_ce = MarginLoss(m=0)  # (m=-1*m)
    # 未标记数据加载器，使用 cycle 生成一个无限循环的迭代器。
    unlabel_loader_iter = cycle(train_unlabel_loader)

    # 初始化一个 AverageMeter 对象
    # 在训练神经网络模型时，通常需要记录每个批次（batch）的损失值和准确率，并计算整个训练过程中这些指标的平均值
    # AverageMeter 提供了一种简便的方法来实现这一功能

    # 用于跟踪二元交叉熵损失（bce_loss）的平均值
    bce_losses = AverageMeter('bce_loss', ':.4e')
    # 跟踪交叉熵损失（ce_loss）的平均值
    ce_losses = AverageMeter('ce_loss', ':.4e')
    # 跟踪熵损失（entropy_loss）的平均值
    entropy_losses = AverageMeter('entropy_loss', ':.4e')
    #
    np_cluster_preds = np.array([])  # cluster_preds 聚类预测
    np_unlabel_targets = np.array([])
    #
    # enumerate()返回一个迭代器，每次迭代都会返回一个包含索引和元素的元组
    # x是原始输入数据
    # x2数据增强
    # target 标签
    for batch_idx, ((x, x2), target) in enumerate(train_label_loader):
        ## 各个类的不确定性权重（固定值）
        beta = 0.2
        Nk = [1600 * 5, 1600 * 5, 1600 * 5, 1600 * 5, 1600 * 5, 1600 * 5, 1600, 1600, 1600, 1600]
        Nmax = 1600 * 5
        p_weight = [beta ** (1 - Nk[i] / Nmax) for i in range(10)]

        # next()从迭代器中获取下一个元素
        # ux 无标签输入
        # ux2 无标签的数据增强输入
        # unlabel_target 【伪】标签
        ((ux, ux2), unlabel_target) = next(unlabel_loader_iter)

        # 拼接张量
        x = torch.cat([x, ux], 0)
        x2 = torch.cat([x2, ux2], 0)
        #
        labeled_len = len(target)
        # print("labeled_len: ", labeled_len)

        x, x2, target = x.to(device), x2.to(device), target.to(device)

        # 在进行反向传播之前，我们需要先调用 optimizer.zero_grad() 来清除这些累积的梯度，以免影响后续的梯度计算
        optimizer.zero_grad()

        # output, feat = model(x) 是在 PyTorch 中使用模型进行前向传播的一种方式
        # 在这行代码中，输入 x 通过模型 model 进行前向计算，返回两个结果：output 和 feat
        # model(x) 前向传播，输入x输出结果和特征
        output, feat = model(x)  # output: [batch size, 10]; feat: [batch size, 512]
        output2, feat2 = model(x2)

        # Softmax 操作，将输出的 logits 转换为概率分布
        # logits：神经网络最后一层的输出（原始得分）
        prob = F.softmax(output, dim=1)
        # output[labeled_len:] 从 output 中截取从索引 labeled_len 开始到最后的部分，也就是【无标签数据的输出】
        reg_prob = F.softmax(output[labeled_len:], dim=1)  # unlabel data's prob 无标签数据概率
        prob2 = F.softmax(output2, dim=1)
        reg_prob2 = F.softmax(output2[labeled_len:], dim=1)  # unlabel data's prob 无标签数据概率

        # update local_labeled_centroids
        # 更新模型中每个类别的本地有标签聚类中心
        # 使用有标签的数据样本的特征 feat 来更新对应类别的聚类中心

        # feat[:labeled_len] 代表只选择【有标签数据的特征】
        # detach() 分离张量：获取张量的值进行处理，但不希望这些操作影响梯度
        # zip(feat[:labeled_len].detach().clone(), target) 将每个样本的特征 feature 和对应的真实标签 true_label 一一配对
        for feature, true_label in zip(feat[:labeled_len].detach().clone(), target):
            labeled_samples_num[true_label] += 1

            # 取出对应 true_label 类别的聚类中心，并将当前样本的特征 feature 累加到这个聚类中心上
            # 这行代码的作用是将每个样本的特征加到其所属类别的聚类中心上，目的是逐步计算所有属于该类别样本的特征平均值，最终更新每个类别的中心位置
            model.local_labeled_centroids.weight.data[true_label] += feature
        # print("before model.local_labeled_centroids.weight.data: ", model.local_labeled_centroids.weight.data)

        # 对聚类质心进行【归一化处理】
        # 每个类别的【聚类质心除以该类别的样本数量】，计算出每个类别的平均特征向量
        # 这样，每个类别的聚类质心最终会是该类别所有样本特征的平均值
        for idx, (feature_centroid, num) in enumerate(
                zip(model.local_labeled_centroids.weight.data, labeled_samples_num)):
            if num > 0:
                model.local_labeled_centroids.weight.data[idx] = feature_centroid / num
        # print("model.local_labeled_centroids.weight.data size: ", model.local_labeled_centroids.weight.data.size())
        # print("model.local_labeled_centroids.weight.data: ", model.local_labeled_centroids.weight.data)
        # print("labeled_samples_num: ", labeled_samples_num)

        # L_reg:  reg_prob 中每一行的预测 label
        # 从模型预测的无标签数据概率中生成伪标签
        # 先复制 reg_prob reg_prob2 （无标签数据的概率分布）
        copy_reg_prob1 = copy.deepcopy(reg_prob.detach())
        copy_reg_prob2 = copy.deepcopy(reg_prob2.detach())
        # np.argmax(..., axis=1)：对每个样本的概率分布，在类别维度上（axis=1）找到最大概率值对应的索引
        # 这个索引即为模型对该样本预测的类别【伪标签】
        # 在 np.argmax 函数中，axis=1 表示沿着某个特定的维度进行操作
        # 在这里，axis=1 意味着在二维数组的每一行中寻找最大值对应的索引，而不是在整个数组中寻找
        reg_label1 = np.argmax(copy_reg_prob1.cpu().numpy(), axis=1)
        reg_label2 = np.argmax(copy_reg_prob2.cpu().numpy(), axis=1)
        ### 制作 target, target 除了 label=1 外与 reg_prob 一致
        # 将无标签数据的伪标签所对应的类别的预测概率设置为 1，以便后续的处理更加明确
        # 人为调整伪标签的置信度，可以更好地引导模型学习
        for idx, (label, oprob) in enumerate(zip(reg_label1, copy_reg_prob1)):
            copy_reg_prob1[idx][label] = 1
        for idx, (label, oprob) in enumerate(zip(reg_label2, copy_reg_prob2)):
            copy_reg_prob2[idx][label] = 1
        #
        # L1损失（L1 Loss），又称为 绝对误差损失
        # 用于衡量模型预测值与真实值之间的差异，计算方式是【取两者差值的绝对值并进行平均】
        L1_loss = nn.L1Loss()
        L_reg1 = 0.0
        L_reg2 = 0.0
        for idx, (ooutput, otarget, label) in enumerate(zip(reg_prob, copy_reg_prob1, reg_label1)):
            L_reg1 = L_reg1 + L1_loss(reg_prob[idx], copy_reg_prob1[idx]) * p_weight[label]
        for idx, (ooutput, otarget, label) in enumerate(zip(reg_prob2, copy_reg_prob2, reg_label2)):
            L_reg2 = L_reg2 + L1_loss(reg_prob2[idx], copy_reg_prob2[idx]) * p_weight[label]
        # 归一化
        L_reg1 = L_reg1 / len(reg_label1)
        L_reg2 = L_reg2 / len(reg_label2)
        #### Ours loss end
        ## 欧氏距离 ########################################################################
        # C = model.centroids.weight.data.detach().clone()
        # Z1 = feat.detach()
        # Z2 = feat2.detach()
        # cZ1 = euclidean_dist(Z1, C)
        # cZ2 = euclidean_dist(Z2, C)

        ## Cluster loss begin (Orchestra)
        # 聚类损失开始（Orchestra算法）
        # cos-similarity ###############################
        # 计算模型特征和聚类中心之间的【余弦相似度】
        # .T 是转置操作，它将张量的形状从 [num_classes, feature_dim] 变为 [feature_dim, num_classes]
        C = model.centroids.weight.data.detach().clone().T
        # 特征向量 feat 和 feat2 进行归一化，使得每个特征向量的长度为 1
        Z1 = F.normalize(feat, dim=1)
        Z2 = F.normalize(feat2, dim=1)

        # @ 是 Python 的矩阵乘法运算符。它执行的是矩阵【点积】操作
        # cZ1 是一个形状为 [batch_size, num_classes] 的矩阵，其中每一行表示某个样本与所有类别聚类中心的相似度
        # cZ1 【样本特征与聚类中心的相似度分数矩阵】
        # Z1 的形状通常是 [batch_size, feature_dim]
        # C 是模型中每个类别的聚类中心，是一个经过转置后的张量，形状是 [feature_dim, num_classes]
        cZ1 = Z1 @ C
        cZ2 = Z2 @ C
        ##
        # 对 cZ1 进行 softmax 归一化，将相似度分数转换为概率分布
        # model.T 通常是一个温度参数，用于缩放 cZ1 中的相似度分数
        tP1 = F.softmax(cZ1 / model.T, dim=1)

        # max(1) 表示在第 1 维上（即类别维度上）寻找最大值
        # 对于每个样本，tP1.max(1) 会找到该样本在所有类别中的最大概率值
        # confidence_cluster_pred 对每个样本的最大类别预测的置信度
        # cluster_pred 每个样本被分配到的类别标签索引，该索引表示模型认为该样本最有可能属于哪个类别
        confidence_cluster_pred, cluster_pred = tP1.max(1)  # cluster_pred: [512]; target: [170]

        # tP2 没有像 tP1 那么做：
        # tP1 是基于原始特征 Z1 计算的，用于生成最终的类别预测（cluster_pred）和置信度（confidence_cluster_pred）
        # 这些预测结果被用于后续的聚类操作和伪标签生成

        # tP2 是基于增强特征 Z2 计算的，可能是用来与 tP1 进行对比学习、聚类一致性或其他正则化目标，但它没有直接用于生成伪标签或置信度计算
        tP2 = F.softmax(cZ2 / model.T, dim=1)
        # logpZ2 = torch.log(F.softmax(cZ2 / model.T, dim=1))
        # Clustering loss
        # L_cluster = - torch.sum(tP1 * logpZ2, dim=1).mean()
        # print("L_cluster: ", L_cluster)
        ## Cluster loss end (Orchestra)
        # 聚类损失结束（Orchestra算法）

        ### 统计 cluster_pred (伪标签，cluster id) 置信度 ###
        # 记录每个类别的置信度
        confidence_list = [0 for _ in range(10)]
        # 记录每个类别中被分配的样本数量
        num_of_cluster = [0 for _ in range(10)]
        # mask_tmp = np.array([])
        for confidence, cluster_id in zip(confidence_cluster_pred[labeled_len:], cluster_pred[labeled_len:]):
            confidence_list[cluster_id] = confidence_list[cluster_id] + confidence
            num_of_cluster[cluster_id] = num_of_cluster[cluster_id] + 1
        for cluster_id, (sum_confidence, num) in enumerate(zip(confidence_list, num_of_cluster)):
            if num > 0:
                confidence_list[cluster_id] = confidence_list[cluster_id].cpu().detach().numpy() / num
                confidence_list[cluster_id] = np.around(confidence_list[cluster_id], 4)  # 保留小数点后4位
        # mask_tmp = np.append(mask_tmp, confidence_cluster_pred[labeled_len:].cpu().detach().numpy())
        # 设置一个阈值，用于筛选置信度高于 0.95 的预测结果
        threshold = 0.95
        # confidence_mask = mask_tmp > threshold
        # 根据置信度筛选【无标签数据】，并创建一个掩码（confidence_mask）
        # 标识出哪些无标签数据的置信度超过阈值（threshold）
        # 最终生成一个包含满足条件的样本索引的张量
        confidence_mask = (confidence_cluster_pred[labeled_len:] > threshold)
        confidence_mask = torch.nonzero(confidence_mask)
        confidence_mask = torch.squeeze(confidence_mask)
        if client_id == 0:
            print("confidence_mask: ", confidence_mask)
        # print("confidence_mask: ", confidence_mask)
        # sys.exit(0)
        # if (args.epochs * global_round + epoch) % 5 == 0:
        print("global round: ", global_round, ";   client_id: ", client_id, ";   confidence_list: ", confidence_list)

        # calculate distance
        # 计算样本之间的余弦相似度
        feat_detach = feat.detach()
        # torch.norm 计算张量的范数（即向量的长度），这里的 2 表示使用 L2 范数，也就是计算向量元素平方和的平方根
        feat_norm = feat_detach / torch.norm(feat_detach, 2, 1, keepdim=True)
        # torch.mm 是矩阵乘法操作，计算的是两个矩阵的点积

        # 余弦相似度矩阵
        cosine_dist = torch.mm(feat_norm, feat_norm.t())
        labeled_len = len(target)

        # 存储正样本对（positive pairs）
        # 正样本对指的是在任务中认为应该有较高相似度或应该被归为同一类的样本对
        pos_pairs = []
        # target 张量从 PyTorch 张量转换为 NumPy 数组
        target_np = target.cpu().numpy()

        ## label part ########################################################################
        # 为每个【有标签】的样本（即前 labeled_len 个样本）选择一个正样本对，并将其索引添加到 pos_pairs 列表中
        for i in range(labeled_len):
            target_i = target_np[i]
            idxs = np.where(target_np == target_i)[0]
            if len(idxs) == 1:
                pos_pairs.append(idxs[0])
            else:
                selec_idx = np.random.choice(idxs, 1)
                while selec_idx == i:
                    selec_idx = np.random.choice(idxs, 1)
                pos_pairs.append(int(selec_idx))

        ## unlabel part ########################################################################
        # 是提取 无标签数据 和 所有样本 之间的余弦相似度矩阵
        unlabel_cosine_dist = cosine_dist[labeled_len:, :]
        # 从 无标签样本与所有样本的余弦相似度矩阵 中，找出每个无标签样本与最相似的两个样本的【相似度值】和【索引】
        # torch.topk 是 PyTorch 中的一个函数，用于从张量中返回前 k 个最大的元素以及它们的索引
        vals, pos_idx = torch.topk(unlabel_cosine_dist, 2, dim=1)
        # print(pos_idx.size())
        # print(pos_idx)
        pos_idx = pos_idx[:, 1].cpu().numpy().flatten().tolist()
        pos_pairs.extend(pos_idx)  # pos_pairs size: [512,1]

        # bce + L_cluster
        cluster_pos_prob = tP2[pos_pairs, :]  # cluster_pos_prob size: [512,10]
        # bce
        # cluster_pos_sim = torch.bmm(tP1.view(args.batch_size, 1, -1), cluster_pos_prob.view(args.batch_size, -1, 1)).squeeze()
        # cluster_ones = torch.ones_like(cluster_pos_sim)
        # cluster_bce_loss = bce(cluster_pos_sim, cluster_ones)
        # cross-entropy
        logcluster_pos_prob = torch.log(cluster_pos_prob)
        L_cluster = - torch.sum(tP1 * logcluster_pos_prob, dim=1).mean()  # [170(label)/512-170(unlabel)]
        #
        pos_prob = prob2[pos_pairs, :]
        pos_sim = torch.bmm(prob.view(args.batch_size, 1, -1), pos_prob.view(args.batch_size, -1, 1)).squeeze()
        ones = torch.ones_like(pos_sim)
        bce_loss = bce(pos_sim, ones)
        ce_loss = ce(output[:labeled_len], target)
        # unlabel ce loss
        # unlabel_ce_loss = unlabel_ce(output[labeled_len:], cluster_pred[labeled_len:])
        ###
        # print("1:",output[labeled_len:].index_select(0,confidence_mask).size())
        # print("2:",cluster_pred[labeled_len:].index_select(0,confidence_mask).size())
        ####
        unlabel_ce_loss = unlabel_ce(output[labeled_len:].index_select(0, confidence_mask),
                                     cluster_pred[labeled_len:].index_select(0, confidence_mask))
        np_cluster_preds = np.append(np_cluster_preds, cluster_pred[labeled_len:].cpu().numpy())
        np_unlabel_targets = np.append(np_unlabel_targets, unlabel_target.cpu().numpy())
        #
        entropy_loss = entropy(torch.mean(prob, 0))

        # loss = - entropy_loss + ce_loss + bce_loss
        # loss = ce_loss
        if global_round > 4:  # 4
            if global_round > 6:  # 6
                loss = - entropy_loss + ce_loss + bce_loss + 0.5 * L_cluster + unlabel_ce_loss  # + 2 * L_reg1 + 2 * L_reg2  # + L_cluster # 调整L_reg倍率
            else:
                loss = - entropy_loss + ce_loss + bce_loss + 0.5 * L_cluster  # + 2 * L_reg1 + 2 * L_reg2 #+ L_cluster # 调整L_reg倍率
        else:
            loss = - entropy_loss + ce_loss + bce_loss  # + 2 * L_reg1 + 2 * L_reg2 # 调整L_reg倍率
        if client_id == 0:
            print("entropy_loss: ", entropy_loss)
            print("ce_loss: ", ce_loss)
            print("bce_loss: ", bce_loss)
            print("L_cluster: ", L_cluster)
            print("unlabel_ce_loss: ", unlabel_ce_loss)
        print("L_reg1: ", 2 * L_reg1)
        print("L_reg2: ", 2 * L_reg2)
        # sys.exit(0)

        bce_losses.update(bce_loss.item(), args.batch_size)
        ce_losses.update(ce_loss.item(), args.batch_size)
        entropy_losses.update(entropy_loss.item(), args.batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # if ((args.epochs * global_round + epoch) % 10 == 0) and (client_id == 0):

    # if client_id == 0:
    # unlabel_acc, w_unlabel_acc = cluster_acc_w(np.array(cluster_pred[labeled_len:].cpu().numpy()), np.array(unlabel_target.cpu().numpy()))
    np_cluster_preds = np_cluster_preds.astype(int)
    unlabel_acc, w_unlabel_acc = cluster_acc_w(np_cluster_preds, np_unlabel_targets)
    print("unlabel_acc: ", unlabel_acc)
    print("w_unlabel_acc: ", w_unlabel_acc)
    print("unlabel target: ", unlabel_target)
    print("unlabel cluster_pred: ", cluster_pred[labeled_len:])
    # sys.exit(0)

    tf_writer.add_scalar('client{}/loss/bce'.format(client_id), bce_losses.avg, args.epochs * global_round + epoch)
    tf_writer.add_scalar('client{}/loss/ce'.format(client_id), ce_losses.avg, args.epochs * global_round + epoch)
    tf_writer.add_scalar('client{}/loss/entropy'.format(client_id), entropy_losses.avg,
                         args.epochs * global_round + epoch)
    tf_writer.add_scalars('client{}/loss'.format(client_id), {"bce": bce_losses.avg},
                          args.epochs * global_round + epoch)
    tf_writer.add_scalars('client{}/loss'.format(client_id), {"ce": ce_losses.avg}, args.epochs * global_round + epoch)
    tf_writer.add_scalars('client{}/loss'.format(client_id), {"entropy": entropy_losses.avg},
                          args.epochs * global_round + epoch)


def test(args, model, labeled_num, device, test_loader, epoch, tf_writer, client_id, global_round):
    model.eval()
    preds = np.array([])
    cluster_preds = np.array([])  # cluster_preds
    targets = np.array([])
    confs = np.array([])
    with torch.no_grad():
        C = model.centroids.weight.data.detach().clone().T
        for batch_idx, (x, label) in enumerate(test_loader):
            x, label = x.to(device), label.to(device)
            output, feat = model(x)
            prob = F.softmax(output, dim=1)
            conf, pred = prob.max(1)
            # cluster pred
            Z1 = F.normalize(feat, dim=1)
            cZ1 = Z1 @ C
            tP1 = F.softmax(cZ1 / model.T, dim=1)
            _, cluster_pred = tP1.max(1)  # return #1: max data    #2: max data index
            #
            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())
            cluster_preds = np.append(cluster_preds, cluster_pred.cpu().numpy())
            confs = np.append(confs, conf.cpu().numpy())
    targets = targets.astype(int)
    preds = preds.astype(int)
    cluster_preds = cluster_preds.astype(int)

    seen_mask = targets < labeled_num
    unseen_mask = ~seen_mask
    ## preds <-> cluster_preds ##
    origin_preds = preds
    # preds = cluster_preds
    ## local_unseen_mask (4) ##
    local_unseen_mask_4 = targets == 4
    local_unseen_acc_4 = cluster_acc(preds[local_unseen_mask_4], targets[local_unseen_mask_4])
    ## local_unseen_mask (4) ##
    local_unseen_mask_5 = targets == 5
    local_unseen_acc_5 = cluster_acc(preds[local_unseen_mask_5], targets[local_unseen_mask_5])
    ## local_unseen_mask (4) ##
    local_unseen_mask_6 = targets == 6
    local_unseen_acc_6 = cluster_acc(preds[local_unseen_mask_6], targets[local_unseen_mask_6])
    ## local_unseen_mask (4) ##
    local_unseen_mask_7 = targets == 7
    local_unseen_acc_7 = cluster_acc(preds[local_unseen_mask_7], targets[local_unseen_mask_7])
    ## local_unseen_mask (4) ##
    local_unseen_mask_8 = targets == 8
    local_unseen_acc_8 = cluster_acc(preds[local_unseen_mask_8], targets[local_unseen_mask_8])
    ## local_unseen_mask (4) ##
    local_unseen_mask_9 = targets == 9
    local_unseen_acc_9 = cluster_acc(preds[local_unseen_mask_9], targets[local_unseen_mask_9])
    ## global_unseen_mask (5-9) ##
    global_unseen_mask = targets > labeled_num
    global_unseen_acc = cluster_acc(preds[global_unseen_mask], targets[global_unseen_mask])
    ##
    # overall_acc = cluster_acc(preds, targets)
    overall_acc, w_overall_acc = cluster_acc_w(origin_preds, targets)
    if ((args.epochs * global_round + epoch) % 10 == 0) and (client_id == 0):
        print("w_overall_acc: ", w_overall_acc)
    # cluster_acc
    overall_cluster_acc = cluster_acc(cluster_preds, targets)
    #
    # seen_acc = accuracy(preds[seen_mask], targets[seen_mask])
    seen_acc = accuracy(origin_preds[seen_mask], targets[seen_mask])
    #
    unseen_acc, w_unseen_acc = cluster_acc_w(preds[unseen_mask], targets[unseen_mask])
    if ((args.epochs * global_round + epoch) % 10 == 0) and (client_id == 0):
        print("w_unseen_acc: ", w_unseen_acc)
    unseen_nmi = metrics.normalized_mutual_info_score(targets[unseen_mask], preds[unseen_mask])
    mean_uncert = 1 - np.mean(confs)
    print(
        'epoch {}, Client id {}, Test overall acc {:.4f}, Test overall cluster acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}, local_unseen acc {:.4f}, global_unseen acc {:.4f}'.format(
            epoch, client_id, overall_acc, overall_cluster_acc, seen_acc, unseen_acc, local_unseen_acc_6,
            global_unseen_acc))
    # tf_writer.add_scalar('client{}/acc/overall'.format(client_id), overall_acc, args.epochs * global_round + epoch)
    # tf_writer.add_scalar('client{}/acc/seen'.format(client_id), seen_acc, args.epochs * global_round + epoch)
    # tf_writer.add_scalar('client{}/acc/unseen'.format(client_id), unseen_acc, args.epochs * global_round + epoch)
    tf_writer.add_scalars('client{}/acc'.format(client_id), {"overall": overall_acc},
                          args.epochs * global_round + epoch)
    tf_writer.add_scalars('client{}/acc'.format(client_id), {"seen": seen_acc}, args.epochs * global_round + epoch)
    tf_writer.add_scalars('client{}/acc'.format(client_id), {"unseen": unseen_acc}, args.epochs * global_round + epoch)
    ##
    tf_writer.add_scalars('client{}/acc'.format(client_id), {"local_unseen_4": local_unseen_acc_4},
                          args.epochs * global_round + epoch)
    tf_writer.add_scalars('client{}/acc'.format(client_id), {"local_unseen_5": local_unseen_acc_5},
                          args.epochs * global_round + epoch)
    tf_writer.add_scalars('client{}/acc'.format(client_id), {"local_unseen_6": local_unseen_acc_6},
                          args.epochs * global_round + epoch)
    tf_writer.add_scalars('client{}/acc'.format(client_id), {"local_unseen_7": local_unseen_acc_7},
                          args.epochs * global_round + epoch)
    tf_writer.add_scalars('client{}/acc'.format(client_id), {"local_unseen_8": local_unseen_acc_8},
                          args.epochs * global_round + epoch)
    tf_writer.add_scalars('client{}/acc'.format(client_id), {"local_unseen_9": local_unseen_acc_9},
                          args.epochs * global_round + epoch)
    # #
    tf_writer.add_scalars('client{}/acc'.format(client_id), {"global_unseen": global_unseen_acc},
                          args.epochs * global_round + epoch)
    ##
    tf_writer.add_scalar('client{}/nmi/unseen'.format(client_id), unseen_nmi, args.epochs * global_round + epoch)
    tf_writer.add_scalar('client{}/uncert/test'.format(client_id), mean_uncert, args.epochs * global_round + epoch)
    return mean_uncert


def main():
    # 1.参数解析和设置：解析输入参数，设置训练超参数（如客户端数量、数据集类型、全局轮数等）。
    # 2.初始化模型和数据：初始化全局模型和每个客户端的模型、数据加载器、优化器、学习率调度器和 TensorBoard 写入器。
    # 3.联邦训练循环：
    #       在每一轮训练中，所有客户端进行本地训练。
    #       聚合客户端模型参数，更新全局模型。
    # 4.全局聚类和模型同步：执行全局聚类并将全局模型的参数下载到客户端模型。
    # 5.保存模型：在训练结束时保存全局模型和每个客户端的模型。
    # clients_train_label_set = []
    parser = argparse.ArgumentParser(description='orca')  # 创建一个命令行参数解析器 parser
    parser.add_argument('--milestones', nargs='+', type=int, default=[140, 180])  # 学习率下降的轮数
    parser.add_argument('--dataset', default='cifar10', help='dataset setting')  # 数据集名称
    parser.add_argument('--clients-num', default=5, type=int)  # 客户端数量
    parser.add_argument('--global-rounds', default=2, type=int)  # 全局训练轮数  默认 40
    parser.add_argument('--labeled-num', default=50, type=int)  # 标记样本的数量
    parser.add_argument('--labeled-ratio', default=0.5, type=float)  # 标记样本的比例
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')  # 随机数种子
    parser.add_argument('--name', type=str, default='debug')  # 实验的名称
    parser.add_argument('--exp_root', type=str, default='./results/')  # 实验结果的存储目录
    parser.add_argument('--epochs', type=int, default=5)  # 每轮训练的次数
    parser.add_argument('-b', '--batch-size', default=512, type=int,
                        metavar='N',
                        help='mini-batch size')  # mini-batch 的大小
    args = parser.parse_args()  # 解析命令行输入的参数，并将其存储在 args 对象中

    # GPU
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    print("-------------------- use ", device, "-----------")

    args.savedir = os.path.join(args.exp_root, args.name)  # 设置保存实验结果的目录路径，路径为 exp_root 和 name 的组合

    # 如果保存目录不存在，创建该目录
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    # CIFAR-10数据集
    if args.dataset == 'cifar10':
        # train_label_set = client_datasets.OPENWORLDCIFAR10(root='./datasets', labeled=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train']), exist_label_list=[0,1,2,3,4,5,6,7,8,9], clients_num=args.clients_num)

        # 初始化标记数据集
        train_label_set = datasets.OPENWORLDCIFAR10(root='./datasets',
                                                    labeled=True,
                                                    labeled_num=10,
                                                    labeled_ratio=args.labeled_ratio,
                                                    download=True,
                                                    transform=TransformTwice(datasets.dict_transform['cifar_train']))
        # 初始化未标记数据集
        train_unlabel_set = client_datasets.OPENWORLDCIFAR10(root='./datasets',
                                                             labeled=False,
                                                             labeled_num=args.labeled_num,
                                                             labeled_ratio=args.labeled_ratio,
                                                             download=True,
                                                             transform=TransformTwice(
                                                                 datasets.dict_transform['cifar_train']),
                                                             unlabeled_idxs=train_label_set.unlabeled_idxs,
                                                             exist_label_list=[0, 1, 2, 3, 4, 5],
                                                             clients_num=args.clients_num)
        # 初始化测试数据集
        test_set = client_datasets.OPENWORLDCIFAR10(root='./datasets',
                                                    labeled=False,
                                                    labeled_num=args.labeled_num,
                                                    labeled_ratio=args.labeled_ratio,
                                                    download=True,
                                                    transform=datasets.dict_transform['cifar_test'],
                                                    unlabeled_idxs=train_label_set.unlabeled_idxs,
                                                    exist_label_list=[0, 1, 2, 3, 4, 5],
                                                    clients_num=args.clients_num)
        # 设置类别数为 10，这是 CIFAR-10 的类别数
        num_classes = 10

        # ####################################################################################
        # 随机生成 子集 和 全集
        # num_clients = args.clients_num
        # num_classes = 10  # CIFAR-10的类别数
        #
        # # 生成标签列表
        # exist_label_list = []
        # clients_labeled_num = []
        #
        # for i in range(num_clients):
        #     # 生成一个包含随机标签的列表
        #     labels = np.random.choice(num_classes, size=6, replace=False).tolist()
        #     exist_label_list.append(labels)
        #     # 设定每个客户端的标记样本数
        #     clients_labeled_num.append(4)  # 这个数可以根据实际需要调整
        #
        # print("Exist Label List:", exist_label_list)
        # print("Clients Labeled Num:", clients_labeled_num)
        # ####################################################################################

        ### prepare clients dataset ###
        ### 准备客户端数据集 ###
        # 模拟 【non-iid】
        ## 子集
        exist_label_list = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 6], [0, 1, 2, 3, 4, 7], [0, 1, 2, 3, 4, 8],
                            [0, 1, 2, 3, 4, 9]]
        clients_labeled_num = [4, 4, 4, 4, 4]
        ## 全集
        exist_label_list = [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 8],
                            [0, 1, 2, 3, 4, 5, 6, 8], [0, 1, 2, 3, 4, 5, 6, 9]]
        clients_labeled_num = [6, 6, 6, 6, 6]


        ##
        clients_train_label_set = []
        clients_train_unlabel_set = []
        clients_test_set = []

        # 遍历客户端，默认是 5 个客户端
        for i in range(args.clients_num):
            # 为当前客户端初始化标记数据集
            client_train_label_set = client_datasets.OPENWORLDCIFAR10(root='./datasets', labeled=True,
                                                                      labeled_num=clients_labeled_num[i],
                                                                      labeled_ratio=args.labeled_ratio, download=True,
                                                                      transform=TransformTwice(
                                                                          datasets.dict_transform['cifar_train']),
                                                                      exist_label_list=exist_label_list[i],
                                                                      clients_num=args.clients_num)
            print(client_train_label_set)
            # 为当前客户端初始化未标记数据集
            client_train_unlabel_set = client_datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False,
                                                                        labeled_num=clients_labeled_num[i],
                                                                        labeled_ratio=args.labeled_ratio, download=True,
                                                                        transform=TransformTwice(
                                                                            datasets.dict_transform['cifar_train']),
                                                                        unlabeled_idxs=client_train_label_set.unlabeled_idxs,
                                                                        exist_label_list=exist_label_list[i],
                                                                        clients_num=args.clients_num)
            # 为当前客户端初始化测试数据集
            client_test_set = client_datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False,
                                                               labeled_num=args.labeled_num,
                                                               labeled_ratio=args.labeled_ratio, download=True,
                                                               transform=datasets.dict_transform['cifar_test'],
                                                               unlabeled_idxs=client_train_label_set.unlabeled_idxs,
                                                               exist_label_list=exist_label_list[i],
                                                               clients_num=args.clients_num)
            client_test_set = client_datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False,
                                                               labeled_num=args.labeled_num,
                                                               labeled_ratio=args.labeled_ratio, download=True,
                                                               transform=datasets.dict_transform['cifar_test'],
                                                               unlabeled_idxs=train_label_set.unlabeled_idxs,
                                                               exist_label_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                               clients_num=args.clients_num)
            # 将当前客户端的数据集添加到相应的列表中
            clients_train_label_set.append(client_train_label_set)
            clients_train_unlabel_set.append(client_train_unlabel_set)
            clients_test_set.append(client_test_set)
        ###
    # cifar100 数据集
    elif args.dataset == 'cifar100':
        train_label_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=True, labeled_num=args.labeled_num,
                                                     labeled_ratio=args.labeled_ratio, download=True,
                                                     transform=TransformTwice(datasets.dict_transform['cifar_train']))
        train_unlabel_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=False, labeled_num=args.labeled_num,
                                                       labeled_ratio=args.labeled_ratio, download=True,
                                                       transform=TransformTwice(datasets.dict_transform['cifar_train']),
                                                       unlabeled_idxs=train_label_set.unlabeled_idxs)
        test_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=False, labeled_num=args.labeled_num,
                                              labeled_ratio=args.labeled_ratio, download=True,
                                              transform=datasets.dict_transform['cifar_test'],
                                              unlabeled_idxs=train_label_set.unlabeled_idxs)
        num_classes = 100

        ####################################################################################################
        ### 添加客户端初始化逻辑 ###
        # CIFAR-100 共有 100 个类别
        exist_label_list = [[i for i in range(20 * j, 20 * (j + 1))] for j in range(args.clients_num)]
        clients_labeled_num = [args.labeled_num // args.clients_num] * args.clients_num

        clients_train_label_set = []
        clients_train_unlabel_set = []
        clients_test_set = []

        # 遍历客户端
        for i in range(args.clients_num):
            # 初始化标记数据集
            client_train_label_set = client_datasets.OPENWORLDCIFAR100(root='./datasets', labeled=True,
                                                                       labeled_num=clients_labeled_num[i],
                                                                       labeled_ratio=args.labeled_ratio, download=True,
                                                                       transform=TransformTwice(
                                                                           datasets.dict_transform['cifar_train']),
                                                                       exist_label_list=exist_label_list[i],
                                                                       clients_num=args.clients_num)
            # 初始化未标记数据集
            client_train_unlabel_set = client_datasets.OPENWORLDCIFAR100(root='./datasets', labeled=False,
                                                                         labeled_num=clients_labeled_num[i],
                                                                         labeled_ratio=args.labeled_ratio,
                                                                         download=True,
                                                                         transform=TransformTwice(
                                                                             datasets.dict_transform['cifar_train']),
                                                                         unlabeled_idxs=client_train_label_set.unlabeled_idxs,
                                                                         exist_label_list=exist_label_list[i],
                                                                         clients_num=args.clients_num)
            # 初始化测试数据集
            client_test_set = client_datasets.OPENWORLDCIFAR100(root='./datasets', labeled=False,
                                                                labeled_num=args.labeled_num,
                                                                labeled_ratio=args.labeled_ratio, download=True,
                                                                transform=datasets.dict_transform['cifar_test'],
                                                                unlabeled_idxs=client_train_label_set.unlabeled_idxs,
                                                                exist_label_list=exist_label_list[i],
                                                                clients_num=args.clients_num)

            # 将客户端的数据集添加到相应的列表中
            clients_train_label_set.append(client_train_label_set)
            clients_train_unlabel_set.append(client_train_unlabel_set)
            clients_test_set.append(client_test_set)
        ####################################################################################################


    else:
        warnings.warn('Dataset is not listed')
        return

    # 存储每个客户端的标记数据 mini-batch 大小
    clients_labeled_batch_size = []
    for i in range(args.clients_num):
        # gpt
        # if i >= len(clients_train_label_set):
        #     raise IndexError(
        #         f"Index {i} is out of range for clients_train_label_set with length {len(clients_train_label_set)}.")

        # 获取第 i 个客户端的标记数据集的长度（样本数量）
        labeled_len = len(clients_train_label_set[i])
        # 获取第 i 个客户端的未标记数据集的长度
        unlabeled_len = len(clients_train_unlabel_set[i])
        # 根据总 batch 大小和标记、未标记数据的比例，计算出标记数据的 batch 大小
        labeled_batch_size = int(args.batch_size * labeled_len / (labeled_len + unlabeled_len))
        # 将计算出的标记数据 batch 大小添加到 clients_labeled_batch_size 列表中
        clients_labeled_batch_size.append(labeled_batch_size)

    # Initialize the splits  # train_label_loader->client_train_label_loader[];   train_unlabel_loader->client_train_unlabel_loader[]
    # 初始化训练数据的拆分

    # 初始化加载器
    client_train_label_loader = []
    client_train_unlabel_loader = []
    client_test_loader = []
    for i in range(
            args.clients_num):  # train_label_loader->client_train_label_loader[];   train_unlabel_loader->client_train_unlabel_loader[]

        # 创建一个 DataLoader 对象，用于从 clients_train_label_set[i] 中加载数据
        train_label_loader = torch.utils.data.DataLoader(clients_train_label_set[i],
                                                         batch_size=clients_labeled_batch_size[i],  # 批次大小
                                                         shuffle=True,  # 是否在每个epoch开始时打乱数据
                                                         num_workers=2,  # 使用两个子进程加载数据
                                                         drop_last=True)  # 如果最后一个批次的样本数少于 batch_size，则丢弃

        train_unlabel_loader = torch.utils.data.DataLoader(clients_train_unlabel_set[i],
                                                           # 批次大小为 总批次大小 减去 标记数据的批次大小
                                                           batch_size=args.batch_size - clients_labeled_batch_size[i],
                                                           shuffle=True,
                                                           num_workers=2,
                                                           drop_last=True)

        # 将每个客户端的 DataLoader 对象分别添加到对应的列表中，以便后续在训练过程中使用这些加载器来获取数据
        client_train_label_loader.append(train_label_loader)
        client_train_unlabel_loader.append(train_unlabel_loader)

        test_loader = torch.utils.data.DataLoader(clients_test_set[i],
                                                  batch_size=100,  # 批次大小为100
                                                  shuffle=False,  # 不打乱数据顺序
                                                  num_workers=1)  # 使用一个子进程加载数据
        client_test_loader.append(test_loader)

    # Initialize the global_model ##############
    # 初始化全局模型
    global global_model
    # 初始化一个 ResNet-18 模型，指定类别数量
    global_model = models.resnet18(num_classes=num_classes)
    # 将模型移动到指定的设备（cuda:0）
    global_model = global_model.to(device)

    # 根据选择的数据集（CIFAR-10 或 CIFAR-100）加载预训练模型的权重文件
    if args.dataset == 'cifar10':
        state_dict = torch.load('./pretrained/simclr_cifar_10.pth.tar')
    elif args.dataset == 'cifar100':
        state_dict = torch.load('./pretrained/simclr_cifar_100.pth.tar')

    # 将 预训练模型的权重 加载到当前的 global_model 中
    global_model.load_state_dict(state_dict, strict=False)
    global_model = global_model.to(device)

    # Freeze the earlier filters
    # 在模型的前几层中冻结参数，也就是不让它们更新权重
    # 这样做的目的是保留这些层中已经学到的特征（通常是较低级的边缘或纹理）
    # 而只训练模型的后几层，使它们适应新的任务
    for name, param in global_model.named_parameters():
        if 'linear' not in name and 'layer4' not in name:
            param.requires_grad = False  # 冻结所有不包含 'linear' 和 'layer4' 的层
        if "centroids" in name:
            param.requires_grad = True  # 对于包含 'centroids' 的参数，解冻并允许更新
    ###########################################
    # 初始化客户端模型 ##############
    clients_model = []  # model->clients_model[client_id] 用于存储每个客户端的模型对象
    clients_optimizer = []  # optimizer->clients_optimizer[client_id] 用于存储每个客户端的优化器对象
    clients_scheduler = []  # scheduler->clients_scheduler[client_id] 用于存储每个客户端的学习率调度器对象
    clients_tf_writer = []  # tf_writer->clients_tf_writer[client_id] 用于存储每个客户端的 TensorBoard 写入对象 tf(TensorFlow)
    for i in range(args.clients_num):
        # First network intialization: pretrain the RotNet network
        # 网络的初始化：预训练 RotNet 网络
        model = models.resnet18(num_classes=num_classes)
        model = model.to(device)

        if args.dataset == 'cifar10':
            state_dict = torch.load('./pretrained/simclr_cifar_10.pth.tar')
        elif args.dataset == 'cifar100':
            state_dict = torch.load('./pretrained/simclr_cifar_100.pth.tar')
        model.load_state_dict(state_dict, strict=False)
        # 深度复制 global_model，确保不会对全局模型产生副作用
        model = copy.deepcopy(global_model)
        model = model.to(device)

        # Freeze the earlier filters
        # 冻结
        for name, param in model.named_parameters():
            if 'linear' not in name and 'layer4' not in name:
                param.requires_grad = False
            if "centroids" in name:
                param.requires_grad = False

        # 保存当前模型到本地客户端
        clients_model.append(model)

        # Set the optimizer
        # 设置优化器
        # 使用随机梯度下降（SGD）优化器来更新模型的权重
        # 学习率为 1e-1
        # 动量为 0.9，并加入 5e-4 的权重衰减（L2正则化）
        optimizer = optim.SGD(model.parameters(), lr=1e-1, weight_decay=5e-4, momentum=0.9)
        # 设置学习率调度器
        # 当训练到 milestones 指定的轮次时
        # 学习率会按 gamma=0.1 缩减
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
        # 将优化器添加到客户端优化器列表
        clients_optimizer.append(optimizer)
        # 将调度器添加到客户端调度器列表
        clients_scheduler.append(scheduler)

        # 初始化 TensorBoard 的日志写入器，指定日志的保存路径为 args.savedir，用于存储训练过程中的数据，如损失、精度等指标
        tf_writer = SummaryWriter(log_dir=args.savedir)
        # 将创建的日志写入器 tf_writer 添加到 clients_tf_writer 列表中
        # 这样每个客户端都有自己的 TensorBoard 写入器，用于记录它们各自的训练日志
        clients_tf_writer.append(tf_writer)
    tf_writer = SummaryWriter(log_dir=args.savedir)

    ## Start FedAvg training ##
    # 开始 FedAvg 训练
    # 全局训练轮次循环
    for global_round in range(args.global_rounds):
        print(" Start global_round {}: ".format(global_round))
        # 本地客户端循环
        for client_id in range(args.clients_num):
            # 在本地的 epochs 循环
            for epoch in range(args.epochs):
                # 计算不确定性
                mean_uncert = test(args, clients_model[client_id], args.labeled_num, device,
                                   client_test_loader[client_id], epoch, tf_writer, client_id, global_round)
                # 训练客户端模型
                train(args, clients_model[client_id], device, client_train_label_loader[client_id],
                      client_train_unlabel_loader[client_id], clients_optimizer[client_id], mean_uncert, epoch,
                      tf_writer, client_id, global_round)
                # 学习率调度，调整学习率
                clients_scheduler[client_id].step()
            # local_clustering #
            # 本地聚类
            clients_model[client_id].local_clustering(device=device)

        # receive_models
        # 接收客户端模型
        receive_models(clients_model)

        # aggregate_parameters 平均了所有 client 的模型参数
        # 聚合客户端模型参数
        aggregate_parameters()

        # Run global clustering
        # 聚合全局模型的局部聚类
        for client_id in range(args.clients_num):
            # clients_model[client_id].named_parameters() 是 PyTorch 的一个方法，它返回当前模型的所有参数及其对应的名称
            for c_name, old_param in clients_model[client_id].named_parameters():
                # 如果参数名中包含 "local_centroids"，则提取这个局部聚类中心（local_centroids）
                # 并通过 np.concatenate 将所有客户端的聚类中心连接在一起，存储在 Z1 中
                # Z1是聚类中心
                if "local_centroids" in c_name:
                    if client_id == 0:
                        Z1 = np.array(copy.deepcopy(old_param.data.cpu().clone()))
                    else:
                        Z1 = np.concatenate((Z1, np.array(copy.deepcopy(old_param.data.cpu().clone()))), axis=0)
        # 将聚类中心 Z1 转换为 PyTorch 张量，并将其传递给全局模型，调用 global_clustering 方法在全局模型中执行聚类更新。
        Z1 = torch.tensor(Z1, device=device).T
        global_model.global_clustering(Z1.to(device).T)  # update self.centroids in global model
        # set labeled data feature instead of self.centroids
        # 在全局模型中，使用标记数据的特征来设置新的聚类中心，替换已有的聚类中心
        global_model.set_labeled_feature_centroids(device=device)

        # download global model param
        # 下载全局模型参数
        # 指定哪些参数不会被平均
        # 这里的 参数过滤器 包括 mem_projections、local_centroids 和 local_labeled_centroids，这些层的参数不参与全局平均
        # name_filters = ['linear', "mem_projections", "centroids", "local_centroids"]
        # name_filters = ['linear', "mem_projections", "local_centroids", "local_labeled_centroids"] #do not AVG FedRep
        name_filters = ["mem_projections", "local_centroids", "local_labeled_centroids"]  # do not AVG FedAVG
        for client_id in range(args.clients_num):
            # 使用 zip 同时遍历全局模型和客户端模型的参数
            # g_name 和 c_name 分别是 全局模型 和 客户端模型 的参数名
            # new_param 和 old_param 是对应的参数值
            for (g_name, new_param), (c_name, old_param) in zip(global_model.named_parameters(),
                                                                clients_model[client_id].named_parameters()):
                # 如果参数名不包含过滤器中的关键词（如 mem_projections 等），则继续更新该参数。
                if all(keyword not in g_name for keyword in name_filters):
                    old_param.data = new_param.data.clone()
                    # print("Download layer name: ", g_name)
            # sys.exit(0)

    ## finish train ##
    # 使用 torch.save 将 global_model 的参数（通过 state_dict() 方法获取）保存为文件 '0102_Ours_cluster-classifier_global.pth'
    torch.save(global_model.state_dict(), './fedrep-trained-model/0102_Ours_cluster-classifier_global.pth')
    # 遍历每个客户端，使用 torch.save 保存每个客户端模型的 state_dict() 到对应的文件，文件名根据客户端 ID 动态生成。
    for client_id in range(args.clients_num):
        torch.save(clients_model[client_id].state_dict(),
                   './fedrep-trained-model/0102_Ours_cluster-classifier_client{}-model.pth'.format(client_id))
    ## save model


if __name__ == '__main__':
    main()
