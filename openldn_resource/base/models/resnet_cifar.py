import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import random


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

# 自定义的元学习的模块，其他resnet类都继承这个类（而不是 nn.Module）
class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            try:
                setattr(curr_mod, name, param)
            except:
                import pdb;
                pdb.set_trace()

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)

# 元学习的线性层
class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

# 元学习的卷积层
class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)

        self.in_channels = ignore.in_channels
        self.out_channels = ignore.out_channels
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        self.kernel_size = ignore.kernel_size

        # # 权重和偏置被注册为缓冲区（而不是标准的可训练参数），这允许在元学习循环中通过自定义方式更新这些参数，而不是通过标准优化器
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    # 返回该卷积层的参数，权重和偏置
    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

# 没用到
class MetaConvTranspose2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.ConvTranspose2d(*args, **kwargs)

        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x, output_size=None):
        output_padding = self._output_padding(x, output_size)
        return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding,
                                  output_padding, self.groups, self.dilation)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

# 元学习的批归一化层
class MetaBatchNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        self.update_batch_stats = True

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            self.training or not self.track_running_stats, self.momentum, self.eps)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

# 没用到
class MetaBatchNorm1d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm1d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        self.update_batch_stats = True

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            self.training or not self.track_running_stats, self.momentum, self.eps)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


########################################################################################################################
class MetaBasicBlock(MetaModule):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(MetaBasicBlock, self).__init__()

        # 与 FedoSSL 一致
        self.conv1 = MetaConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = MetaBatchNorm2d(out_planes)
        self.conv2 = MetaConv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(out_planes)
        self.shortcut = nn.Sequential()
        self.is_padding = 0

        # todo 这里要按FedoSSL改
        if stride != 1 or in_planes != self.expansion * out_planes:
            self.shortcut = nn.AvgPool2d(2)
            if in_planes != self.expansion * out_planes:
                self.is_padding = 1

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 这里openLDN需要判断，而FedoSSL不需要，所以在上面的todo要改
        if self.is_padding:
            shortcut = self.shortcut(x)
            out += torch.cat([shortcut, torch.zeros(shortcut.shape).type(torch.cuda.FloatTensor)], 1)
        else:
            out += self.shortcut(x)
        out = F.relu(out)
        return out


########################################################################################################################
class ResNet(MetaModule):
    def __init__(self, block, num_blocks, no_class=10, batch_norm=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # openLDN 没有像 FedoSSL 那样直接调用 nn.conv2d()
        # 而是自己实现 【元学习】ResNet 模型
        # 3 输入的图像具有 3 个通道（RGB 图像）
        # 64 表示卷积层会使用 64 个卷积核，输出 64 通道的特征图
        # conv1 和 bn1 与 FedoSSL 一致
        self.conv1 = MetaConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = MetaBatchNorm2d(64)

        # 四个残差层与 FedoSSL 一致
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # todo 这里 FedoSSL有一个池化层
        '''
         # 【自适应平均池化层】
        # 将特征图的尺寸缩放到 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        '''

        # 线性层与 FedoSSL 一致
        self.classifier = MetaLinear(512 * block.expansion, no_class)

    # 一致
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # 全局平均池化，将每个通道缩减为 1x1。这里池化的核大小为 4 对特定大小的特征图进行固定的降采样
        # FedoSSL：
        # out = self.avgpool(out)
        # nn.AdaptiveAvgPool2d((1, 1))
        out = F.avg_pool2d(out, 4)

        # 展平
        # FedoSSL
        # out = torch.flatten(out, 1)
        out = out.view(out.size(0), -1)

        # 一致
        out_lin = self.classifier(out)
        return out, out_lin

    def update_batch_stats(self, flag):
        for m in self.modules():
            if isinstance(m, MetaBatchNorm2d):
                m.update_batch_stats = flag


# 成对相似性网络(前馈神经网络)
class SimNet(MetaModule):
    def __init__(self, input, hidden, output):
        super(SimNet, self).__init__()
        self.linear1 = MetaLinear(input, hidden)
        self.linear2 = MetaLinear(hidden, output)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        out = self.linear2(x)
        return torch.sigmoid(out)


# 与FedoSSL一致
def resnet18(**kwargs):
    return ResNet(MetaBasicBlock, [2, 2, 2, 2], **kwargs)
