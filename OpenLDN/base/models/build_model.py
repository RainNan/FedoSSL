# # base/models/build_model.py
import torch
import torch.nn as nn

from models import resnet_s
from models.resnet_s import BasicBlock


#
# def build_model(args):
#     if args.arch == 'resnet18':
#         if args.dataset in ['cifar10', 'cifar100', 'svhn']:
#             from . import resnet_cifar as models
#         elif args.dataset == 'tinyimagenet':
#             from . import resnet_tinyimagenet as models
#         else:
#             from . import resnet as models
#         model = models.resnet18(no_class=args.no_class)
#         simnet = models.SimNet(1024, 100, 1)
#
#     # use dataparallel if there's multiple gpus
#     if torch.cuda.device_count() > 1:
#         model = torch.nn.DataParallel(model)
#         simnet = torch.nn.DataParallel(simnet)
#     return model, simnet
#
#
#
def build_model(args):
    # 确保从正确的模块导入 BasicBlock
    if args.arch == 'resnet18':
        if args.dataset in ['cifar10', 'cifar100', 'svhn']:
            from . import resnet_cifar as models
            # from ./models/resnet_s import BasicBlock  # 从 resnet.py 导入 BasicBlock
        elif args.dataset == 'tinyimagenet':
            from . import resnet_tinyimagenet as models
            # from .resnet import BasicBlock  # 同样需要导入 BasicBlock
        else:
            from . import resnet as models
            # from .resnet import BasicBlock  # 继续确保 BasicBlock 的导入

        model = models.resnet18(no_class=args.no_class)
        simnet = models.SimNet(1024, 100, 1)

        # 使用 BasicBlock 的 expansion 参数
        model.centroids = nn.Linear(512 * BasicBlock.expansion, args.no_class, bias=False)
        model.mem_projections = nn.Linear(1024, 512, bias=False)

    # 如果有多个 GPU，使用 DataParallel
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        simnet = torch.nn.DataParallel(simnet)

    return model, simnet

