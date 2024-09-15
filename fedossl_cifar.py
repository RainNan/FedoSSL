# fedossl
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
from OpenLDN.base.models.build_model import build_model
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
          global_round, simnet=None, optimizer_simnet=None):  # 增加simnet和optimizer_simnet
    # 初始化模型中的局部标记聚类中心为零。
    model.local_labeled_centroids.weight.data.zero_()  # model.local_labeled_centroids.weight.data: torch.Size([10, 512])

    # 记录【每个类别】的标记样本数
    labeled_samples_num = [0 for _ in range(10)]

    # 将模型设置为“训练模式”
    model.train()

    # 使用二元交叉熵损失
    bce = torch.nn.BCELoss()

    m = min(m, 0.5)
    ce = MarginLoss(m=-1 * m)

    # 未标记数据的交叉熵损失
    unlabel_ce = MarginLoss(m=0)
    # 未标记数据加载器，使用 cycle 生成一个无限循环的迭代器。
    unlabel_loader_iter = cycle(train_unlabel_loader)

    # 初始化损失计量器
    bce_losses = AverageMeter('bce_loss', ':.4e')
    ce_losses = AverageMeter('ce_loss', ':.4e')
    entropy_losses = AverageMeter('entropy_loss', ':.4e')

    # OpenLDN部分：初始化成对相似性损失
    losses_pair = AverageMeter('pair_loss', ':.4e')  # 成对相似性损失

    np_cluster_preds = np.array([])  # 聚类预测
    np_unlabel_targets = np.array([])

    # 开始训练
    for batch_idx, ((x, x2), target) in enumerate(train_label_loader):
        # 各个类的不确定性权重（固定值）
        beta = 0.2
        Nk = [1600 * 5, 1600 * 5, 1600 * 5, 1600 * 5, 1600 * 5, 1600 * 5, 1600, 1600, 1600, 1600]
        Nmax = 1600 * 5
        p_weight = [beta ** (1 - Nk[i] / Nmax) for i in range(10)]

        # 从无标签加载器中获取数据
        ((ux, ux2), unlabel_target) = next(unlabel_loader_iter)

        # 拼接张量
        x = torch.cat([x, ux], 0)
        x2 = torch.cat([x2, ux2], 0)

        labeled_len = len(target)
        x, x2, target = x.to(device), x2.to(device), target.to(device)

        optimizer.zero_grad()

        # 前向传播
        output, feat = model(x)
        output2, feat2 = model(x2)

        prob = F.softmax(output, dim=1)
        reg_prob = F.softmax(output[labeled_len:], dim=1)  # 无标签数据的概率
        prob2 = F.softmax(output2, dim=1)
        reg_prob2 = F.softmax(output2[labeled_len:], dim=1)

        # 更新局部标记聚类中心
        for feature, true_label in zip(feat[:labeled_len].detach().clone(), target):
            labeled_samples_num[true_label] += 1
            model.local_labeled_centroids.weight.data[true_label] += feature

        for idx, (feature_centroid, num) in enumerate(
                zip(model.local_labeled_centroids.weight.data, labeled_samples_num)):
            if num > 0:
                model.local_labeled_centroids.weight.data[idx] = feature_centroid / num

        # 生成伪标签
        copy_reg_prob1 = copy.deepcopy(reg_prob.detach())
        copy_reg_prob2 = copy.deepcopy(reg_prob2.detach())

        reg_label1 = np.argmax(copy_reg_prob1.cpu().numpy(), axis=1)
        reg_label2 = np.argmax(copy_reg_prob2.cpu().numpy(), axis=1)

        for idx, (label, oprob) in enumerate(zip(reg_label1, copy_reg_prob1)):
            copy_reg_prob1[idx][label] = 1
        for idx, (label, oprob) in enumerate(zip(reg_label2, copy_reg_prob2)):
            copy_reg_prob2[idx][label] = 1

        # L1正则化损失
        L1_loss = torch.nn.L1Loss()
        L_reg1 = 0.0
        L_reg2 = 0.0
        for idx, (ooutput, otarget, label) in enumerate(zip(reg_prob, copy_reg_prob1, reg_label1)):
            L_reg1 = L_reg1 + L1_loss(reg_prob[idx], copy_reg_prob1[idx]) * p_weight[label]
        for idx, (ooutput, otarget, label) in enumerate(zip(reg_prob2, copy_reg_prob2, reg_label2)):
            L_reg2 = L_reg2 + L1_loss(reg_prob2[idx], copy_reg_prob2[idx]) * p_weight[label]
        L_reg1 = L_reg1 / len(reg_label1)
        L_reg2 = L_reg2 / len(reg_label2)

        ## OpenLDN新增部分 ##
        # 双层优化（Bi-level optimization）中，创建一个中间模型 model_
        model_, _ = build_model(args)  # 从OpenLDN中创建一个新模型
        model_ = model_.cuda()
        model_.load_state_dict(model.state_dict(),strict=False)  # 同步参数

        feat_, logits_ = model_(x)
        logits_ = F.softmax(logits_, dim=1)
        feat_ = F.normalize(feat_, dim=1)

        # 计算成对相似性损失
        feats = torch.cat((feat_[:labeled_len], feat2[:labeled_len]), dim=0)
        sim_feat = simnet(feats)  # 使用simnet计算成对相似性损失
        loss_pair = F.mse_loss(F.softmax(output[:labeled_len]), sim_feat)

        # OpenLDN 更新simnet优化器
        optimizer_simnet.zero_grad()
        loss_pair.backward(retain_graph=True)  # 反向传播成对相似性损失
        optimizer_simnet.step()

        ## OpenLDN新增部分结束 ##

        # 原有的FedoSSL部分
        pos_prob = prob2[:labeled_len]
        pos_sim = torch.bmm(prob.view(args.batch_size, 1, -1), pos_prob.view(args.batch_size, -1, 1)).squeeze()
        ones = torch.ones_like(pos_sim)
        bce_loss = bce(pos_sim, ones)
        ce_loss = ce(output[:labeled_len], target)

        # 总损失计算
        if global_round > 4:  # 根据轮次调整损失计算
            if global_round > 6:
                loss = - entropy(torch.mean(prob, 0)) + ce_loss + bce_loss + 0.5 * loss_pair + unlabel_ce(output[labeled_len:], reg_label1)
            else:
                loss = - entropy(torch.mean(prob, 0)) + ce_loss + bce_loss + 0.5 * loss_pair
        else:
            loss = - entropy(torch.mean(prob, 0)) + ce_loss + bce_loss

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新损失
        bce_losses.update(bce_loss.item(), args.batch_size)
        ce_losses.update(ce_loss.item(), args.batch_size)
        entropy_losses.update(entropy(torch.mean(prob, 0)).item(), args.batch_size)
        losses_pair.update(loss_pair.item(), args.batch_size)  # 更新成对相似性损失

    np_cluster_preds = np_cluster_preds.astype(int)
    unlabel_acc, w_unlabel_acc = cluster_acc_w(np_cluster_preds, np_unlabel_targets)
    print("unlabel_acc: ", unlabel_acc)
    print("w_unlabel_acc: ", w_unlabel_acc)
    print("unlabel target: ", unlabel_target)
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
    # 将模型设置为评估模式
    model.eval()
    preds = np.array([])  # 每个样本的类别预测值
    cluster_preds = np.array([])  # 聚类预测值
    targets = np.array([])  # 真实标签
    confs = np.array([])  # 置信度

    # 上下文管理器，它的主要作用是在其作用域内禁用自动梯度计算
    with torch.no_grad():
        # 将模型中聚类中心的权重矩阵 model.centroids.weight 转置后提取出来
        C = model.centroids.weight.data.detach().clone().T
        for batch_idx, (x, label) in enumerate(test_loader):  # 迭代 batch、数据、标签
            x, label = x.to(device), label.to(device)

            # output 模型的输出（通常是分类任务中的未归一化得分，称为 logits）
            # feat 特征向量
            output, feat = model(x)

            # 用于将 output 转换为概率分布
            prob = F.softmax(output, dim=1)

            # conf 最大类别的概率值，即置信度
            # pred 最大类别的概率值对应的索引
            conf, pred = prob.max(1)

            # cluster pred
            # 特征向量 feat 在维度 1 上标准化
            Z1 = F.normalize(feat, dim=1)

            cZ1 = Z1 @ C

            # tP1 是样本对于每个聚类中心的归一化得分
            tP1 = F.softmax(cZ1 / model.T, dim=1)

            # 找到每个样本对应的最大聚类得分的索引
            # cluster_pred：模型对每个样本所属聚类的预测索引
            _, cluster_pred = tP1.max(1)  # return #1: max data    #2: max data index
            #

            # 将当前批次的真实标签转换为 NumPy 数组，并追加到 targets 数组中
            targets = np.append(targets, label.cpu().numpy())
            # 将当前批次的分类预测结果（pred）转换为 NumPy 数组，并追加到 preds 数组中
            preds = np.append(preds, pred.cpu().numpy())
            # 同理
            cluster_preds = np.append(cluster_preds, cluster_pred.cpu().numpy())
            # 同理
            confs = np.append(confs, conf.cpu().numpy())

    #  NumPy 数组转换为整型
    targets = targets.astype(int)
    preds = preds.astype(int)
    cluster_preds = cluster_preds.astype(int)

    # 【掩码】由 True 或 False 组成，表示哪些元素应被选中或处理，哪些元素应被忽略
    # seen_mask 是一个布尔掩码，用于标记哪些样本属于已经标注过的类别
    # targets < labeled_num 表示：将 targets 中小于 labeled_num 的标签标记为 True，这意味着这些样本属于已经标注的类别。
    seen_mask = targets < labeled_num
    # unseen_mask 是一个布尔掩码，表示哪些样本属于未标注的类别。
    # ~seen_mask 是对 seen_mask 的按位取反操作，将 True 变为 False，False 变为 True，这样就标记出那些属于未标注类别的样本。
    unseen_mask = ~seen_mask
    ## preds <-> cluster_preds ##

    # 原始的 preds
    # preds: 这个变量是模型对测试数据的预测结果，表示模型对每个样本的分类预测
    origin_preds = preds
    # preds = cluster_preds
    ## local_unseen_mask (4) ##

    # 这里生成了一个布尔掩码 local_unseen_mask_4
    # 用于标记 targets 中那些标签为 4 的样本（即属于类别 4 的样本）。这个掩码用来选择属于该类别的样本
    local_unseen_mask_4 = targets == 4
    # 用上一步生成的掩码，提取 preds 和 targets 中对应类别 4 的预测和真实标签。
    # cluster_acc 是一个函数，用来计算给定预测结果和真实标签之间的准确率。
    # local_unseen_acc_4 保存了类别 4 的聚类准确率。
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

    # 整体准确率 overall_acc
    # 加权的整体准确率 w_overall_acc
    overall_acc, w_overall_acc = cluster_acc_w(origin_preds, targets)
    if ((args.epochs * global_round + epoch) % 10 == 0) and (client_id == 0):
        # 每 10 个 epoch 并且在第一个客户端（client_id == 0）打印一次 w_overall_acc
        print("w_overall_acc: ", w_overall_acc)
    # cluster_acc
    # 基于聚类预测的整体准确率 overall_cluster_acc
    overall_cluster_acc = cluster_acc(cluster_preds, targets)
    #
    # seen_acc = accuracy(preds[seen_mask], targets[seen_mask])
    # 已见类准确率
    seen_acc = accuracy(origin_preds[seen_mask], targets[seen_mask])
    #
    # 未见类准确率 和 加权未见类准确率
    unseen_acc, w_unseen_acc = cluster_acc_w(preds[unseen_mask], targets[unseen_mask])
    if ((args.epochs * global_round + epoch) % 10 == 0) and (client_id == 0):
        print("w_unseen_acc: ", w_unseen_acc)

    # 计算未见类的 归一化互信息得分（NMI）
    # metrics.normalized_mutual_info_score NMI 是一个常用于聚类的指标，用于衡量聚类结果与真实标签的相似性
    unseen_nmi = metrics.normalized_mutual_info_score(targets[unseen_mask], preds[unseen_mask])
    # 计算模型预测的平均不确定性 mean_uncert
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
    parser.add_argument('--name', type=str, default='debug')  # 实验的名称，tfwriter保存在 /results/dubug
    parser.add_argument('--exp_root', type=str, default='./results/')  # 实验结果的存储目录
    parser.add_argument('--epochs', type=int, default=5)  # 每轮训练的次数
    parser.add_argument('-b', '--batch-size', default=512, type=int,
                        metavar='N',
                        help='mini-batch size')  # mini-batch 的大小

    parser.add_argument('--arch',type=str,default='resnet18')
    parser.add_argument('--no-class', default=10, type=int, help='total classes')  # 注意这里的no-class代表的是类别总数




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
        '''

        对于所有数据集，我们首先将类划分为 60% 的已见类 和 40% 的未见类
        然后选择已见类中的 50% 作为标记数据，剩下的作为未标记数据
        （即，cifar10 共 10 个类，6 个类作为 seen class，剩下 4 个为 unseen class）
        （然后在 6 个 seen class 中选择一半作为标签数据，另一半作为无标签数据）

        对于 CIFAR-10 和 CINIC-10 数据集
        选择一个未见类作为全局未见类，其余 3 个未见类为本地未见类
        每个客户端拥有所有 6 个已见类、1 个全局未见类和 1 个本地未见类
        
        '''
        ### 准备客户端数据集 ###
        # 模拟 【non-iid】
        ## 子集
        exist_label_list = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 6], [0, 1, 2, 3, 4, 7], [0, 1, 2, 3, 4, 8],
                            [0, 1, 2, 3, 4, 9]]
        clients_labeled_num = [4, 4, 4, 4, 4]
        ## 全集

        # 全部类 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # 0, 1, 2, 3, 4, 5 是所有客户端共享的类，即【已见类 seen class】
        # 6 为 全局未见类
        # 7, 8, 9 是特定客户端拥有，即【未见类 unseen class】
        exist_label_list = [[0, 1, 2, 3, 4, 5, 6, 7],  # 客户端 0 : 缺少 8, 9
                            [0, 1, 2, 3, 4, 5, 6, 7],
                            [0, 1, 2, 3, 4, 5, 6, 8],
                            [0, 1, 2, 3, 4, 5, 6, 8],
                            [0, 1, 2, 3, 4, 5, 6, 9]]
        # 每个客户端有 6 个标记的数据类
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
    ####################################################################################################
    # 初始化客户端模型
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

    '''
    新增
    '''
    _, simnet = build_model(args)
    simnet = simnet.cuda()
    optimizer_simnet = torch.optim.Adam(simnet.params(), lr=0.001)

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
                      tf_writer, client_id, global_round,simnet,optimizer_simnet)
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
