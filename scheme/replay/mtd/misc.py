import torch
import random
import numpy as np
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment
from scheme.replay.podnet.basenet import BasicNet
from scheme.replay.podnet.my_resnet import ResidualBlock as ResidualBlockPODNetCifar
from scheme.replay.afc.my_resnet_importance import ResidualBlock as ResidualBlockAFCCifar
from scheme.replay.podnet.resnet import BasicBlock as BasicBlockPODNetImageNet
from scheme.replay.afc.resnet_importance import BasicBlock as BasicBlockAFCImageNet
from model.modified_resnet_imagenet import BasicBlock as BasicBlockLUCIRImageNet
from model.resnet_cifar import BasicBlock as BasicBlockSSILCifar


def initialize_instanceA_from_instanceB(instanceA, instanceB):
    insA_dict = instanceA.__dict__
    insB_dict = instanceB.__dict__
    for key, value in insB_dict.items():
        insA_dict[key] = deepcopy(value)


class IdentityConv2d(nn.Conv2d):
    """
    Implement a identical convolution
    """
    def __init__(self, channels, requires_grad=True):
        super(IdentityConv2d, self).__init__(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.weight.data = torch.eye(channels).view(channels, channels, 1, 1)
        self.weight.requires_grad = requires_grad


class PermutableResidualBlockCifar(nn.Module):
    """
    For PODNet -> CifarResNet
    Implement a residual block that can be permuted equivalently
    Operations: permute, not permute
    """
    def __init__(self, instance, iconv_requires_grad=True):
        super(PermutableResidualBlockCifar, self).__init__()
        assert (isinstance(instance, ResidualBlockPODNetCifar) or
                isinstance(instance, ResidualBlockAFCCifar))
        initialize_instanceA_from_instanceB(self, instance)

        self.iconv_a = IdentityConv2d(channels=self.conv_a.out_channels, requires_grad=iconv_requires_grad)
        self.iconv_b = IdentityConv2d(channels=self.conv_b.out_channels, requires_grad=iconv_requires_grad)

    def gen_permutation(self):
        permutation = []
        permutation.append(torch.randperm(self.conv_a.out_channels).to(self.conv_a.weight.device))
        permutation.append(torch.randperm(self.conv_b.out_channels).to(self.conv_b.weight.device))
        return permutation

    def permute(self, permutation):
        index = 0
        assert len(permutation) == 2
        module_names = ['conv_a', 'bn_a', 'iconv_a', 'conv_b', 'bn_b', 'iconv_b']
        for name in module_names:
            module = eval('self.{}'.format(name))
            if isinstance(module, torch.nn.Conv2d):
                weight = module.weight.data
                shape = weight.shape
                if isinstance(module, IdentityConv2d):
                    assert len(permutation[index]) == shape[1]
                    weight_per = weight.permute(1, 0, 2, 3)
                    weight_per = weight_per.reshape(shape[1], -1)
                    weight_per = torch.index_select(weight_per, 0, permutation[index])
                    weight_per = weight_per.reshape((shape[1], shape[0], shape[2], shape[3]))
                    weight_per = weight_per.permute(1, 0, 2, 3)
                    module.weight.data = weight_per.data
                    index += 1
                else:
                    assert len(permutation[index]) == shape[0]
                    weight_per = weight.view(shape[0], -1)
                    weight_per = torch.index_select(weight_per, 0, permutation[index])
                    weight_per = weight_per.reshape(shape)
                    module.weight.data = weight_per.data
                if module.bias is not None:
                    raise NotImplementedError
            if isinstance(module, torch.nn.BatchNorm2d):
                if module.weight is not None:
                    weight = module.weight.data
                    weight = torch.index_select(weight, 0, permutation[index])
                    module.weight.data = weight.data
                if module.bias is not None:
                    bias = module.bias.data
                    bias = torch.index_select(bias, 0, permutation[index])
                    module.bias.data = bias.data
                if module.running_mean is not None:
                    running_mean = module.running_mean.data
                    running_mean = torch.index_select(running_mean, 0, permutation[index])
                    module.running_mean.data = running_mean.data
                if module.running_var is not None:
                    running_var = module.running_var.data
                    running_var = torch.index_select(running_var, 0, permutation[index])
                    module.running_var.data = running_var.data

    @staticmethod
    def pad(x):
        return torch.cat((x, x.mul(0)), 1)

    def forward(self, x):
        y = self.conv_a(x)
        y = self.bn_a(y)
        y = self.iconv_a(y)
        y = F.relu(y, inplace=True)

        y = self.conv_b(y)
        y = self.bn_b(y)
        y = self.iconv_b(y)
        if self.increase_dim:
            x = self.downsampler(x)
            if self._need_pad:
                x = self.pad(x)

        y = x + y
        if self.last_relu:
            y = F.relu(y, inplace=True)
        return y


class PermutableResidualBlockCifarScalable(PermutableResidualBlockCifar):
    """
    Operations:
    permute+scale: random_permute=True, random_scale=True
    permute+not scale: random_permute=True, random_scale=False
    not permute+scale (only scale conv, bn, and iconv layers): random_permute=False, random_scale=True
    not permute+not scale (only add one iconv layer): random_permute=False, random_scale=False
    """
    def __init__(self, instance, iconv_requires_grad=True, random_permute=True, random_scale=True):
        super(PermutableResidualBlockCifarScalable, self).__init__(instance, iconv_requires_grad)
        self.low = 0.8
        self.high = 1.25
        self.random_scale = random_scale
        self.random_permute = random_permute

    def gen_permutation(self):
        permutation = []
        if isinstance(self.random_permute, bool) and self.random_permute:
            permutation.append(torch.randperm(self.conv_a.out_channels).to(self.conv_a.weight.device))
            permutation.append(torch.randperm(self.conv_b.out_channels).to(self.conv_b.weight.device))
        elif isinstance(self.random_permute, bool) and not self.random_permute:
            permutation.append(torch.arange(self.conv_a.out_channels).to(self.conv_a.weight.device))
            permutation.append(torch.arange(self.conv_b.out_channels).to(self.conv_b.weight.device))
        elif isinstance(self.random_permute, str) and self.random_permute == 'match':
            groups = [['conv_a', 'bn_a', 'iconv_a'], ['conv_b', 'bn_b', 'iconv_b']]
            for module_names in groups:
                similarity = 0
                for name in module_names:
                    module = eval('self.{}'.format(name))
                    if isinstance(module, torch.nn.Conv2d):
                        weight = module.weight.data
                        shape = weight.shape
                        if isinstance(module, IdentityConv2d):
                            weight_per = weight.permute(1, 0, 2, 3)
                            weight_per = weight_per.reshape(shape[1], -1)
                            similarity = similarity + weight_per @ weight_per.T
                        else:
                            weight_per = weight.view(shape[0], -1)
                            similarity = similarity + weight_per @ weight_per.T
                        if module.bias is not None:
                            raise NotImplementedError
                    if isinstance(module, torch.nn.BatchNorm2d):
                        if module.weight is not None:
                            weight = module.weight.data
                            weight = weight.unsqueeze(dim=1)
                            similarity = similarity + weight @ weight.T
                        if module.bias is not None:
                            bias = module.bias.data
                            bias = bias.unsqueeze(dim=1)
                            similarity = similarity + bias @ bias.T
                row_indices, col_indices = linear_sum_assignment(similarity.cpu().numpy(), maximize=False)
                permutation.append(torch.tensor(col_indices).to(self.conv_a.weight.device))
        else:
            raise NotImplementedError
        return permutation

    def gen_scale_coef(self):
        coef_a = np.random.uniform(self.low, self.high, self.iconv_a.in_channels)
        sign = np.array([1 if np.random.uniform(0, 1) < 0.5 else -1 for i in range(self.iconv_a.in_channels)])
        coef_a = torch.tensor(coef_a, device=self.iconv_a.weight.device, dtype=torch.float32)
        coef_b = np.random.uniform(self.low, self.high, self.iconv_b.in_channels)
        sign = np.array([1 if np.random.uniform(0, 1) < 0.5 else -1 for i in range(self.iconv_b.in_channels)])
        coef_b = torch.tensor(coef_b, device=self.iconv_b.weight.device, dtype=torch.float32)
        return [coef_a, coef_b]

    def permute(self, permutation):
        if not self.random_scale:
            super(PermutableResidualBlockCifarScalable, self).permute(permutation)
        else:
            index = 0
            assert len(permutation) == 2
            coefs = self.gen_scale_coef()
            module_names = ['conv_a', 'bn_a', 'iconv_a', 'conv_b', 'bn_b', 'iconv_b']
            for name in module_names:
                module = eval('self.{}'.format(name))
                if isinstance(module, torch.nn.Conv2d):
                    weight = module.weight.data
                    shape = weight.shape
                    if isinstance(module, IdentityConv2d):
                        assert len(permutation[index]) == shape[1]
                        weight_per = weight.permute(1, 0, 2, 3)
                        weight_per = weight_per.reshape(shape[1], -1)
                        weight_per = torch.index_select(weight_per, 0, permutation[index]) \
                                     * (1.0 / coefs[index][:, None])  # scale the input channel
                        weight_per = weight_per.reshape((shape[1], shape[0], shape[2], shape[3]))
                        weight_per = weight_per.permute(1, 0, 2, 3)
                        module.weight.data = weight_per.data
                        index += 1
                    else:
                        assert len(permutation[index]) == shape[0]
                        weight_per = weight.view(shape[0], -1)
                        weight_per = torch.index_select(weight_per, 0, permutation[index]) \
                                     * coefs[index][:, None]  # scale the out channel
                        weight_per = weight_per.reshape(shape)
                        module.weight.data = weight_per.data
                    if module.bias is not None:
                        raise NotImplementedError
                if isinstance(module, torch.nn.BatchNorm2d):
                    if module.weight is not None:
                        weight = module.weight.data
                        weight = torch.index_select(weight, 0, permutation[index])
                        module.weight.data = weight.data
                    if module.bias is not None:
                        bias = module.bias.data
                        bias = torch.index_select(bias, 0, permutation[index]) * coefs[index]
                        module.bias.data = bias.data
                    if module.running_mean is not None:
                        running_mean = module.running_mean.data
                        running_mean = torch.index_select(running_mean, 0, permutation[index]) * coefs[index]
                        module.running_mean.data = running_mean.data
                    if module.running_var is not None:
                        running_var = module.running_var.data
                        running_var = torch.index_select(running_var, 0, permutation[index])
                        module.running_var.data = running_var.data


class PermutableBasicBlockImageNet(nn.Module):
    """
        For PODNet -> ResNet
        Implement a residual block that can be permuted equivalently
        """

    def __init__(self, instance, iconv_requires_grad=True):
        super(PermutableBasicBlockImageNet, self).__init__()
        assert (isinstance(instance, BasicBlockPODNetImageNet) or
                isinstance(instance, BasicBlockAFCImageNet))
        initialize_instanceA_from_instanceB(self, instance)
        # TODO, can not get theoretical identity mapping
        self.iconv_1 = IdentityConv2d(channels=self.conv1.out_channels, requires_grad=iconv_requires_grad)
        self.iconv_2 = IdentityConv2d(channels=self.conv2.out_channels, requires_grad=iconv_requires_grad)
        if self.downsample is not None:
            self.iconv_ds = IdentityConv2d(channels=self.downsample[0].out_channels, requires_grad=iconv_requires_grad)

    def gen_permutation(self):
        permutation = []
        permutation.append(torch.randperm(self.conv1.out_channels).to(self.conv1.weight.device))
        permutation.append(torch.randperm(self.conv2.out_channels).to(self.conv2.weight.device))
        if self.downsample is not None:
            permutation.append(torch.randperm(self.downsample[0].out_channels).to(self.conv1.weight.device))
        return permutation

    def permute(self, permutation):
        index = 0
        assert len(permutation) in (2, 3)
        module_names = ['conv1', 'bn1', 'iconv_1', 'conv2', 'bn2', 'iconv_2']
        if self.downsample is not None:
            module_names += ['downsample[0]', 'downsample[1]', 'iconv_ds']
        for name in module_names:
            module = eval('self.{}'.format(name))
            if isinstance(module, torch.nn.Conv2d):
                weight = module.weight.data
                shape = weight.shape
                if isinstance(module, IdentityConv2d):
                    assert len(permutation[index]) == shape[1]
                    weight_per = weight.permute(1, 0, 2, 3)
                    weight_per = weight_per.reshape(shape[1], -1)
                    weight_per = torch.index_select(weight_per, 0, permutation[index])
                    weight_per = weight_per.reshape((shape[1], shape[0], shape[2], shape[3]))
                    weight_per = weight_per.permute(1, 0, 2, 3)
                    module.weight.data = weight_per.data
                    index += 1
                else:
                    assert len(permutation[index]) == shape[0]
                    weight_per = weight.view(shape[0], -1)
                    weight_per = torch.index_select(weight_per, 0, permutation[index])
                    weight_per = weight_per.reshape(shape)
                    module.weight.data = weight_per.data
                if module.bias is not None:
                    raise NotImplementedError
            if isinstance(module, torch.nn.BatchNorm2d):
                if module.weight is not None:
                    weight = module.weight.data
                    weight = torch.index_select(weight, 0, permutation[index])
                    module.weight.data = weight.data
                if module.bias is not None:
                    bias = module.bias.data
                    bias = torch.index_select(bias, 0, permutation[index])
                    module.bias.data = bias.data
                if module.running_mean is not None:
                    running_mean = module.running_mean.data
                    running_mean = torch.index_select(running_mean, 0, permutation[index])
                    module.running_mean.data = running_mean.data
                if module.running_var is not None:
                    running_var = module.running_var.data
                    running_var = torch.index_select(running_var, 0, permutation[index])
                    module.running_var.data = running_var.data

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.iconv_1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.iconv_2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            identity = self.iconv_ds(identity)

        out += identity

        if self.last_relu:
            out = self.relu(out)

        return out


class PermutableBasicBlockImageNetScalable(PermutableBasicBlockImageNet):
    def __init__(self, instance, iconv_requires_grad=True, random_permute=True, random_scale=True):
        super(PermutableBasicBlockImageNetScalable, self).__init__(instance, iconv_requires_grad)
        self.low = 0.8
        self.high = 1.25
        self.random_scale = random_scale
        self.random_permute = random_permute

    def gen_permutation(self):
        permutation = []
        if isinstance(self.random_permute, bool) and self.random_permute:
            permutation.append(torch.randperm(self.conv1.out_channels).to(self.conv1.weight.device))
            permutation.append(torch.randperm(self.conv2.out_channels).to(self.conv2.weight.device))
            if self.downsample is not None:
                permutation.append(torch.randperm(self.downsample[0].out_channels).to(self.conv1.weight.device))
        elif isinstance(self.random_permute, bool) and not self.random_permute:
            permutation.append(torch.arange(self.conv1.out_channels).to(self.conv1.weight.device))
            permutation.append(torch.arange(self.conv2.out_channels).to(self.conv2.weight.device))
            if self.downsample is not None:
                permutation.append(torch.arange(self.downsample[0].out_channels).to(self.conv1.weight.device))
        elif isinstance(self.random_permute, str) and self.random_permute == 'match':
            groups = [['conv1', 'bn1', 'iconv_1'], ['conv2', 'bn2', 'iconv_2']]
            if self.downsample is not None:
                groups += [['downsample[0]', 'downsample[1]', 'iconv_ds']]
            for module_names in groups:
                similarity = 0
                for name in module_names:
                    module = eval('self.{}'.format(name))
                    if isinstance(module, torch.nn.Conv2d):
                        weight = module.weight.data
                        shape = weight.shape
                        if isinstance(module, IdentityConv2d):
                            weight_per = weight.permute(1, 0, 2, 3)
                            weight_per = weight_per.reshape(shape[1], -1)
                            similarity = similarity + weight_per @ weight_per.T
                        else:
                            weight_per = weight.view(shape[0], -1)
                            similarity = similarity + weight_per @ weight_per.T
                        if module.bias is not None:
                            raise NotImplementedError
                    if isinstance(module, torch.nn.BatchNorm2d):
                        if module.weight is not None:
                            weight = module.weight.data
                            weight = weight.unsqueeze(dim=1)
                            similarity = similarity + weight @ weight.T
                        if module.bias is not None:
                            bias = module.bias.data
                            bias = bias.unsqueeze(dim=1)
                            similarity = similarity + bias @ bias.T
                row_indices, col_indices = linear_sum_assignment(similarity.cpu().numpy(), maximize=False)
                permutation.append(torch.tensor(col_indices).to(self.conv1.weight.device))
        else:
            raise NotImplementedError
        return permutation

    def gen_scale_coef(self):
        coef_a = np.random.uniform(self.low, self.high, self.iconv_1.in_channels)
        sign = np.array([1 if np.random.uniform(0, 1) < 0.5 else -1 for i in range(self.iconv_1.in_channels)])
        coef_a = torch.tensor(coef_a, device=self.iconv_1.weight.device, dtype=torch.float32)
        coef_b = np.random.uniform(self.low, self.high, self.iconv_2.in_channels)
        sign = np.array([1 if np.random.uniform(0, 1) < 0.5 else -1 for i in range(self.iconv_2.in_channels)])
        coef_b = torch.tensor(coef_b, device=self.iconv_2.weight.device, dtype=torch.float32)
        if self.downsample is None:
            return [coef_a, coef_b]
        else:
            coef_ds = np.random.uniform(self.low, self.high, self.iconv_ds.in_channels)
            sign = np.array([1 if np.random.uniform(0, 1) < 0.5 else -1 for i in range(self.iconv_ds.in_channels)])
            coef_ds = torch.tensor(coef_ds, device=self.iconv_ds.weight.device, dtype=torch.float32)
            return [coef_a, coef_b, coef_ds]

    def permute(self, permutation):
        if not self.random_scale:
            super(PermutableBasicBlockImageNetScalable, self).permute(permutation)
        else:
            index = 0
            assert len(permutation) in (2, 3)
            module_names = ['conv1', 'bn1', 'iconv_1', 'conv2', 'bn2', 'iconv_2']
            if self.downsample is not None:
                module_names += ['downsample[0]', 'downsample[1]', 'iconv_ds']
            coefs = self.gen_scale_coef()
            for name in module_names:
                module = eval('self.{}'.format(name))
                if isinstance(module, torch.nn.Conv2d):
                    weight = module.weight.data
                    shape = weight.shape
                    if isinstance(module, IdentityConv2d):
                        assert len(permutation[index]) == shape[1]
                        weight_per = weight.permute(1, 0, 2, 3)
                        weight_per = weight_per.reshape(shape[1], -1)
                        weight_per = torch.index_select(weight_per, 0, permutation[index]) \
                                     * (1.0 / coefs[index][:, None])  # scale the input channel
                        weight_per = weight_per.reshape((shape[1], shape[0], shape[2], shape[3]))
                        weight_per = weight_per.permute(1, 0, 2, 3)
                        module.weight.data = weight_per.data
                        index += 1
                    else:
                        assert len(permutation[index]) == shape[0]
                        weight_per = weight.view(shape[0], -1)
                        weight_per = torch.index_select(weight_per, 0, permutation[index]) \
                                     * coefs[index][:, None]
                        weight_per = weight_per.reshape(shape)
                        module.weight.data = weight_per.data
                    if module.bias is not None:
                        raise NotImplementedError
                if isinstance(module, torch.nn.BatchNorm2d):
                    if module.weight is not None:
                        weight = module.weight.data
                        weight = torch.index_select(weight, 0, permutation[index])
                        module.weight.data = weight.data
                    if module.bias is not None:
                        bias = module.bias.data
                        bias = torch.index_select(bias, 0, permutation[index]) * coefs[index]
                        module.bias.data = bias.data
                    if module.running_mean is not None:
                        running_mean = module.running_mean.data
                        running_mean = torch.index_select(running_mean, 0, permutation[index]) * coefs[index]
                        module.running_mean.data = running_mean.data
                    if module.running_var is not None:
                        running_var = module.running_var.data
                        running_var = torch.index_select(running_var, 0, permutation[index])
                        module.running_var.data = running_var.data


class PermutableBasicBlockLUCIRImageNet(nn.Module):
    """
        For LUCIR -> ResNet
        Implement a residual block that can be permuted equivalently
        """

    def __init__(self, instance, iconv_requires_grad=True, random_permute=True):
        super(PermutableBasicBlockLUCIRImageNet, self).__init__()
        assert isinstance(instance, BasicBlockLUCIRImageNet)
        residual_function1 = instance.residual_function1

        part_1 = residual_function1[:2]
        channels = part_1[-1].num_features
        iconv_1 = IdentityConv2d(channels=channels, requires_grad=iconv_requires_grad)
        self.residual_function1_1 = deepcopy(part_1)
        self.residual_function1_1.append(iconv_1)

        part_2 = nn.Sequential(residual_function1[2], residual_function1[3], residual_function1[4])
        channels = part_2[-1].num_features
        iconv_2 = IdentityConv2d(channels=channels, requires_grad=iconv_requires_grad)
        self.residual_function1_2 = deepcopy(part_2)
        self.residual_function1_2.append(iconv_2)

        self.residual_function2 = deepcopy(instance.residual_function2)

        self.shortcut = deepcopy(instance.shortcut)
        if len(self.shortcut) > 0:
            assert len(self.shortcut) == 2
            channels = self.shortcut[-1].num_features
            iconv_shortcut = IdentityConv2d(channels=channels, requires_grad=iconv_requires_grad)
            self.shortcut.append(iconv_shortcut)

    def gen_permutation(self):
        permutation = []
        device = self.residual_function1_1[0].weight.device
        permutation.append(torch.randperm(self.residual_function1_1[0].out_channels).to(device))
        permutation.append(torch.randperm(self.residual_function1_2[1].out_channels).to(device))
        if len(self.shortcut) > 0:
            permutation.append(torch.randperm(self.shortcut[0].out_channels).to(device))
        return permutation

    def permute(self, permutation):
        index = 0
        assert len(permutation) in (2, 3)
        module_names = ['residual_function1_1[0]', 'residual_function1_1[1]', 'residual_function1_1[2]',
                        'residual_function1_2[1]', 'residual_function1_2[2]', 'residual_function1_2[3]']
        if len(self.shortcut) > 0:
            module_names += ['shortcut[0]', 'shortcut[1]', 'shortcut[2]']
        for name in module_names:
            module = eval('self.{}'.format(name))
            if isinstance(module, torch.nn.Conv2d):
                weight = module.weight.data
                shape = weight.shape
                if isinstance(module, IdentityConv2d):
                    assert len(permutation[index]) == shape[1]
                    weight_per = weight.permute(1, 0, 2, 3)
                    weight_per = weight_per.reshape(shape[1], -1)
                    weight_per = torch.index_select(weight_per, 0, permutation[index])
                    weight_per = weight_per.reshape((shape[1], shape[0], shape[2], shape[3]))
                    weight_per = weight_per.permute(1, 0, 2, 3)
                    module.weight.data = weight_per.data
                    index += 1
                else:
                    assert len(permutation[index]) == shape[0]
                    weight_per = weight.view(shape[0], -1)
                    weight_per = torch.index_select(weight_per, 0, permutation[index])
                    weight_per = weight_per.reshape(shape)
                    module.weight.data = weight_per.data
                if module.bias is not None:
                    raise NotImplementedError
            if isinstance(module, torch.nn.BatchNorm2d):
                if module.weight is not None:
                    weight = module.weight.data
                    weight = torch.index_select(weight, 0, permutation[index])
                    module.weight.data = weight.data
                if module.bias is not None:
                    bias = module.bias.data
                    bias = torch.index_select(bias, 0, permutation[index])
                    module.bias.data = bias.data
                if module.running_mean is not None:
                    running_mean = module.running_mean.data
                    running_mean = torch.index_select(running_mean, 0, permutation[index])
                    module.running_mean.data = running_mean.data
                if module.running_var is not None:
                    running_var = module.running_var.data
                    running_var = torch.index_select(running_var, 0, permutation[index])
                    module.running_var.data = running_var.data

    def forward(self, x):
        out = self.residual_function1_1(x)
        out = self.residual_function1_2(out)
        out += self.shortcut(x)
        out = self.residual_function2(out)
        return out


class PermutableBasicBlockSSILCifar(nn.Module):
    """
        For LUCIR -> ResNet
        Implement a residual block that can be permuted equivalently
        """

    def __init__(self, instance, iconv_requires_grad=True, random_permute=True):
        super(PermutableBasicBlockSSILCifar, self).__init__()
        assert isinstance(instance, BasicBlockSSILCifar)
        residual_function1 = instance.residual_function1

        part_1 = residual_function1[:2]
        channels = part_1[-1].num_features
        iconv_1 = IdentityConv2d(channels=channels, requires_grad=iconv_requires_grad)
        self.residual_function1_1 = deepcopy(part_1)
        self.residual_function1_1.append(iconv_1)

        part_2 = nn.Sequential(residual_function1[2], residual_function1[3], residual_function1[4])
        channels = part_2[-1].num_features
        iconv_2 = IdentityConv2d(channels=channels, requires_grad=iconv_requires_grad)
        self.residual_function1_2 = deepcopy(part_2)
        self.residual_function1_2.append(iconv_2)

        self.residual_function2 = deepcopy(instance.residual_function2)

        self.shortcut = deepcopy(instance.shortcut)
        if len(self.shortcut) > 0:
            assert len(self.shortcut) == 2
            channels = self.shortcut[-1].num_features
            iconv_shortcut = IdentityConv2d(channels=channels, requires_grad=iconv_requires_grad)
            self.shortcut.append(iconv_shortcut)

    def gen_permutation(self):
        permutation = []
        device = self.residual_function1_1[0].weight.device
        permutation.append(torch.randperm(self.residual_function1_1[0].out_channels).to(device))
        permutation.append(torch.randperm(self.residual_function1_2[1].out_channels).to(device))
        if len(self.shortcut) > 0:
            permutation.append(torch.randperm(self.shortcut[0].out_channels).to(device))
        return permutation

    def permute(self, permutation):
        index = 0
        assert len(permutation) in (2, 3)
        module_names = ['residual_function1_1[0]', 'residual_function1_1[1]', 'residual_function1_1[2]',
                        'residual_function1_2[1]', 'residual_function1_2[2]', 'residual_function1_2[3]']
        if len(self.shortcut) > 0:
            module_names += ['shortcut[0]', 'shortcut[1]', 'shortcut[2]']
        for name in module_names:
            module = eval('self.{}'.format(name))
            if isinstance(module, torch.nn.Conv2d):
                weight = module.weight.data
                shape = weight.shape
                if isinstance(module, IdentityConv2d):
                    assert len(permutation[index]) == shape[1]
                    weight_per = weight.permute(1, 0, 2, 3)
                    weight_per = weight_per.reshape(shape[1], -1)
                    weight_per = torch.index_select(weight_per, 0, permutation[index])
                    weight_per = weight_per.reshape((shape[1], shape[0], shape[2], shape[3]))
                    weight_per = weight_per.permute(1, 0, 2, 3)
                    module.weight.data = weight_per.data
                    index += 1
                else:
                    assert len(permutation[index]) == shape[0]
                    weight_per = weight.view(shape[0], -1)
                    weight_per = torch.index_select(weight_per, 0, permutation[index])
                    weight_per = weight_per.reshape(shape)
                    module.weight.data = weight_per.data
                if module.bias is not None:
                    raise NotImplementedError
            if isinstance(module, torch.nn.BatchNorm2d):
                if module.weight is not None:
                    weight = module.weight.data
                    weight = torch.index_select(weight, 0, permutation[index])
                    module.weight.data = weight.data
                if module.bias is not None:
                    bias = module.bias.data
                    bias = torch.index_select(bias, 0, permutation[index])
                    module.bias.data = bias.data
                if module.running_mean is not None:
                    running_mean = module.running_mean.data
                    running_mean = torch.index_select(running_mean, 0, permutation[index])
                    module.running_mean.data = running_mean.data
                if module.running_var is not None:
                    running_var = module.running_var.data
                    running_var = torch.index_select(running_var, 0, permutation[index])
                    module.running_var.data = running_var.data

    def forward(self, x):
        out = self.residual_function1_1(x)
        out = self.residual_function1_2(out)
        out += self.shortcut(x)
        out = self.residual_function2(out)
        return out


class ModelWrapper(BasicNet):
    def __init__(self, model, forward_with_branch=False):
        assert not (isinstance(model, nn.DataParallel) or
                    isinstance(model, nn.parallel.DistributedDataParallel))
        super(ModelWrapper, self).__init__(args=model.args)
        self_dict = self.__dict__
        for key, value in model.__dict__.items():
            self_dict[key] = value
        self.forward_with_branch = forward_with_branch

    def branch_training(self, flag=True):
        """
        A built-in method for model to switch training state
        Args:
            self: model (nn.Module)
            flag: bool

        Returns:

        """
        for name, module in self.named_modules():
            if 'branch' not in name:
                continue
            if isinstance(module, nn.Sequential):
                for module_ in module:
                    for module__ in module_.modules():
                        module__.training = flag
            if isinstance(module, nn.Module):
                for module_ in module.modules():
                    module_.training = flag

    def freeze_branch(self):
        for name, module in self.named_modules():
            if 'branch' not in name:
                continue
            if isinstance(module, nn.Module):
                for p in list(module.parameters()):
                    p.requires_grad = False
        self.branch_training(flag=False)

    def branch_parameters(self):
        parameters = []
        for name, p in self.named_parameters():
            if 'branch' not in name:
                continue
            if p.requires_grad:
                parameters.append(p)
        return parameters

    @staticmethod
    def load_param(named_param: OrderedDict, model=None, detach=True, requires_grad=False):
        # printlog('\033[1;30;41mWarning! Using \'load_param\' with \'requires_grad=True\', '
        #          'this will unbind optimizer with model\033[0m')
        for key, value in named_param.items():
            # TODO, requires_grad is unrealized!
            # ********* Warning! this will unbind optimizer with model ********* #
            # this operation will change the memory address of parameters in model!
            # if you need to optimize the model,
            # it must need to recreate the optimizer for the model!
            module_path, _, param_name = key.rpartition(".")
            mod: torch.nn.Module = model.get_submodule(module_path)
            assert hasattr(mod, param_name)
            # assert isinstance(named_param[key], nn.Parameter)
            if not detach:
                value = nn.Parameter(value)
            try:
                if detach:
                    exec('mod.{}.data = value.data.clone().detach()'.format(param_name))
                else:
                    exec('mod.{}.data = value.data'.format(param_name))
            except SyntaxError:
                if isinstance(mod, torch.nn.ParameterList):
                    if detach:
                        exec('mod[{}].data = value.data.clone().detach()'.format(eval(param_name)))
                    else:
                        exec('mod[{}].data = value.data'.format(eval(param_name)))
                else:
                    raise NotImplementedError('This ERROR is caused by special customized structure:{}, '
                                              'not implemented in \'load_param\' function!'.format(mod))

    def forward_ridge_branch(self, feature):
        assert self.training
        results = {}
        named_param = dict(self.named_parameters())
        for index in range(1, self.args.mtd['num']+1):
            # load parameters
            ridge_param = OrderedDict()
            for name, p in named_param.items():
                if 'branch' in name and eval(name.split('.')[0][-1]) == index:
                    if 'classifier' in name or 'post_processor' in name:
                        ori_name = name.replace('branch_', '').replace('_{}'.format(index), '')
                    else:
                        ori_name = name.replace('branch_', 'convnet.').replace('_{}'.format(index), '')
                    if 'iconv' in name:
                        channels = p.shape[0]
                        ori_param = torch.eye(channels).view(channels, channels, 1, 1).to(p.device)
                    else:
                        ori_param = named_param[ori_name]
                    ridge_name = name.replace('_{}'.format(index), '_{}'.format(self.args.mtd['num']+1))
                    ridge_param[ridge_name] = (p.detach() + ori_param.detach()) / 2
            self.load_param(ridge_param, self)

            # forward
            if self.args.dataset == 'CIFAR100':
                if self.args.mtd['stages'][0] == 'stage_3':
                    _, y = eval('self.branch_stage_3_{}'.format(self.args.mtd['num']+1))(feature)
                    y = eval('self.branch_stage_4_{}'.format(self.args.mtd['num']+1))(y)
                elif self.args.mtd['stages'][0] == 'stage_4':
                    y = eval('self.branch_stage_4_{}'.format(self.args.mtd['num']+1))(feature)
                else:
                    raise ValueError
            elif self.args.dataset in ('ImageNet100', 'ImageNet1000'):
                if self.args.mtd['stages'][0] == 'layer3':
                    y = eval('self.branch_layer3_{}'.format(self.args.mtd['num']+1))(self.convnet.end_relu(feature))
                    y = eval('self.branch_layer4_{}'.format(self.args.mtd['num']+1))(self.convnet.end_relu(y))
                elif self.args.mtd['stages'][0] == 'layer4':
                    y = eval('self.branch_layer4_{}'.format(self.args.mtd['num']+1))(self.convnet.end_relu(feature))
                else:
                    raise ValueError
            else:
                raise NotImplementedError

            y = self.convnet.end_features(y)
            y = eval('self.branch_classifier_{}'.format(self.args.mtd['num']+1))(y)

            results['logits_{}->{}'.format(index, 0)] = y['logits']
            results['raw_logits_{}->{}'.format(index, 0)] = y['raw_logits']

        return results

    def forward(self, x):

        if self.forward_with_branch:

            outputs = self.convnet(x)

            if hasattr(self, "classifier_no_act") and self.classifier_no_act:
                selected_features = outputs["raw_features"]
            else:
                selected_features = outputs["features"]

            clf_outputs = self.classifier(selected_features)

            logits_sum = clf_outputs['logits']
            embeding_sum = outputs["raw_features"]
            outputs['logits_0'] = clf_outputs['logits']
            outputs['raw_logits_0'] = clf_outputs['raw_logits']
            for index in range(1, self.args.mtd['num']+1):

                if self.args.mtd['stages'][0] in ('stage_3', 'layer3'):
                    feature = outputs['attention'][-3]
                elif self.args.mtd['stages'][0] in ('stage_4', 'layer4'):
                    feature = outputs['attention'][-2]
                else:
                    raise ValueError

                if 'ridge' in self.args.mtd and self.args.mtd['ridge'] > 0 and self.training:
                    results = self.forward_ridge_branch(feature)
                    outputs.update(results)

                if self.training and 'mixup' in self.args.mtd:
                    # apply mixup to feature
                    bs = x.size()[0]
                    mixup_lambd = np.random.beta(self.args.mtd['mixup']['a'], self.args.mtd['mixup']['b'])
                    mixup_index = torch.randperm(bs).to(x.device)
                    feature = (1 - mixup_lambd) * feature + mixup_lambd * feature[mixup_index]
                    outputs['mixup_lambd_{}'.format(index)] = mixup_lambd
                    outputs['mixup_index_{}'.format(index)] = mixup_index

                if self.training and 'noise' in self.args.mtd:
                    noise = torch.randn_like(feature).to(feature.device)
                    # noise = noise / torch.norm(noise)
                    feature = feature + noise * self.args.mtd['noise']

                if self.args.dataset == 'CIFAR100':
                    if self.args.mtd['stages'][0] == 'stage_3':
                        _, y = eval('self.branch_stage_3_{}'.format(index))(feature)
                        y = eval('self.branch_stage_4_{}'.format(index))(y)
                    elif self.args.mtd['stages'][0] == 'stage_4':
                        y = eval('self.branch_stage_4_{}'.format(index))(feature)
                    else:
                        raise ValueError
                elif self.args.dataset in ('ImageNet100', 'ImageNet1000', 'DTD', 'CUB'):
                    if self.args.mtd['stages'][0] == 'layer3':
                        y = eval('self.branch_layer3_{}'.format(index))(self.convnet.end_relu(feature))
                        y = eval('self.branch_layer4_{}'.format(index))(self.convnet.end_relu(y))
                    elif self.args.mtd['stages'][0] == 'layer4':
                        y = eval('self.branch_layer4_{}'.format(index))(self.convnet.end_relu(feature))
                    else:
                        raise ValueError
                else:
                    raise NotImplementedError

                y = self.convnet.end_features(y)
                embeding_sum = embeding_sum + y
                outputs['raw_features_{}'.format(index)] = y
                y = eval('self.branch_classifier_{}'.format(index))(y)
                logits_sum = logits_sum + y['logits']
                outputs['logits_{}'.format(index)] = y['logits']
                outputs['raw_logits_{}'.format(index)] = y['raw_logits']

            outputs['raw_features_avg'] = embeding_sum / (self.args.mtd['num'] + 1)
            if 'output' in self.args.mtd:
                if self.args.mtd['output'] == 'average':
                    outputs['logits'] = logits_sum / (self.args.mtd['num'] + 1)
                if self.args.mtd['output'] == 'origin':
                    outputs['logits'] = clf_outputs['logits']
                if self.args.mtd['output'] == 'max':
                    logits = []
                    for index in range(self.args.mtd['num'] + 1):
                        logits.append(outputs['logits_{}'.format(index)].unsqueeze(dim=0))
                        # logits.append(F.normalize(outputs['logits_{}'.format(index)], p=2, dim=1).unsqueeze(dim=0))
                    logits = torch.cat(logits, dim=0).max(dim=0)[0]
                    outputs['logits'] = logits
            else:
                outputs['logits'] = clf_outputs['logits']

            return outputs

        else:

            outputs = self.convnet(x)
            if hasattr(self, "classifier_no_act") and self.classifier_no_act:
                selected_features = outputs["raw_features"]
            else:
                selected_features = outputs["features"]
            clf_outputs = self.classifier(selected_features)
            outputs.update(clf_outputs)

            return outputs


class LUCIRModelWraaper(nn.Module):
    def __init__(self, model, args, forward_with_branch=False):
        assert not (isinstance(model, nn.DataParallel) or
                    isinstance(model, nn.parallel.DistributedDataParallel))
        super(LUCIRModelWraaper, self).__init__()
        self_dict = self.__dict__
        for key, value in model.__dict__.items():
            self_dict[key] = value
        self.args = args
        self.forward_with_branch = forward_with_branch

    def branch_training(self, flag=True):
        """
        A built-in method for model to switch training state
        Args:
            self: model (nn.Module)
            flag: bool

        Returns:

        """
        for name, module in self.named_modules():
            if 'branch' not in name:
                continue
            if isinstance(module, nn.Sequential):
                for module_ in module:
                    for module__ in module_.modules():
                        module__.training = flag
            if isinstance(module, nn.Module):
                for module_ in module.modules():
                    module_.training = flag

    def freeze_branch(self):
        for name, module in self.named_modules():
            if 'branch' not in name:
                continue
            if isinstance(module, nn.Module):
                for p in list(module.parameters()):
                    p.requires_grad = False
        self.branch_training(flag=False)

    def branch_parameters(self):
        parameters = []
        for name, p in self.named_parameters():
            if 'branch' not in name:
                continue
            if p.requires_grad:
                parameters.append(p)
        return parameters

    def forward(self, x):

        if self.forward_with_branch:

            out = self.layer0(x)
            out = self.maxpool(out)
            out = self.layer1(out)
            out = self.layer2(out)
            if self.args.mtd['stages'][0] == 'layer3':
                feature = out
            out = self.layer3(out)
            if self.args.mtd['stages'][0] == 'layer4':
                feature = out
            out = self.layer4(out)
            out = self.avgpool(out)
            embeding = out.view(out.size(0), -1)
            out = self.fc(embeding)

            outputs = {}
            logits_sum = out
            embeding_sum = embeding
            outputs['raw_features'] = embeding
            outputs['logits_0'] = out
            for index in range(1, self.args.mtd['num']+1):

                if self.training and 'noise' in self.args.mtd:
                    noise = torch.randn_like(feature).to(feature.device)
                    # noise = noise / torch.norm(noise)
                    feature = feature + noise * self.args.mtd['noise']

                if self.args.dataset in ('ImageNet100', 'ImageNet1000'):
                    if self.args.mtd['stages'][0] == 'layer3':
                        y = eval('self.branch_layer3_{}'.format(index))(feature)
                        y = eval('self.branch_layer4_{}'.format(index))(y)
                    elif self.args.mtd['stages'][0] == 'layer4':
                        y = eval('self.branch_layer4_{}'.format(index))(feature)
                    else:
                        raise ValueError
                else:
                    raise NotImplementedError

                y = self.avgpool(y)
                y = y.view(y.size(0), -1)
                embeding_sum = embeding_sum + y
                outputs['raw_features_{}'.format(index)] = y
                y = eval('self.branch_fc_{}'.format(index))(y)
                logits_sum = logits_sum + y
                outputs['logits_{}'.format(index)] = y

            outputs['embedding_avg'] = embeding_sum / (self.args.mtd['num'] + 1)
            if 'output' in self.args.mtd:
                if self.args.mtd['output'] == 'average':
                    outputs['logits'] = logits_sum / (self.args.mtd['num'] + 1)
                if self.args.mtd['output'] == 'origin':
                    outputs['logits'] = out['logits_0']
                if self.args.mtd['output'] == 'max':
                    logits = []
                    for index in range(self.args.mtd['num'] + 1):
                        logits.append(outputs['logits_{}'.format(index)].unsqueeze(dim=0))
                        # logits.append(F.normalize(outputs['logits_{}'.format(index)], p=2, dim=1).unsqueeze(dim=0))
                    logits = torch.cat(logits, dim=0).max(dim=0)[0]
                    outputs['logits'] = logits
            else:
                outputs['logits'] = out['logits_0']
            return outputs

        else:
            out = self.layer0(x)
            out = self.maxpool(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out


class SSILModelWraaper(nn.Module):
    def __init__(self, model, args, forward_with_branch=False):
        assert not (isinstance(model, nn.DataParallel) or
                    isinstance(model, nn.parallel.DistributedDataParallel))
        super(SSILModelWraaper, self).__init__()
        self_dict = self.__dict__
        for key, value in model.__dict__.items():
            self_dict[key] = value
        self.args = args
        self.forward_with_branch = forward_with_branch

    def branch_training(self, flag=True):
        """
        A built-in method for model to switch training state
        Args:
            self: model (nn.Module)
            flag: bool

        Returns:

        """
        for name, module in self.named_modules():
            if 'branch' not in name:
                continue
            if isinstance(module, nn.Sequential):
                for module_ in module:
                    for module__ in module_.modules():
                        module__.training = flag
            if isinstance(module, nn.Module):
                for module_ in module.modules():
                    module_.training = flag

    def freeze_branch(self):
        for name, module in self.named_modules():
            if 'branch' not in name:
                continue
            if isinstance(module, nn.Module):
                for p in list(module.parameters()):
                    p.requires_grad = False
        self.branch_training(flag=False)

    def branch_parameters(self):
        parameters = []
        for name, p in self.named_parameters():
            if 'branch' not in name:
                continue
            if p.requires_grad:
                parameters.append(p)
        return parameters

    def forward(self, x):

        if self.args.dataset in ('CIFAR100', 'SVHN', 'FIVE'):
            if self.forward_with_branch:

                out = self.layer0(x)
                out = self.layer1(out)
                if self.args.mtd['stages'][0] == 'layer2':
                    feature = out
                out = self.layer2(out)
                if self.args.mtd['stages'][0] == 'layer3':
                    feature = out
                out = self.layer3(out)
                out = self.avgpool(out)
                embeding = out.view(out.size(0), -1)
                out = self.fc(embeding)

                outputs = {}
                logits_sum = out
                embeding_sum = embeding
                outputs['raw_features'] = embeding
                outputs['logits_0'] = out
                for index in range(1, self.args.mtd['num']+1):

                    if self.training and 'noise' in self.args.mtd:
                        noise = torch.randn_like(feature).to(feature.device)
                        # noise = noise / torch.norm(noise)
                        feature = feature + noise * self.args.mtd['noise']

                    if self.args.dataset in ('CIFAR100', 'SVHN', 'FIVE'):
                        if self.args.mtd['stages'][0] == 'layer2':
                            y = eval('self.branch_layer2_{}'.format(index))(feature)
                            y = eval('self.branch_layer3_{}'.format(index))(y)
                        elif self.args.mtd['stages'][0] == 'layer3':
                            y = eval('self.branch_layer3_{}'.format(index))(feature)
                        else:
                            raise ValueError
                    elif self.args.dataset in ('ImageNet100', 'ImageNet1000'):
                        raise NotImplementedError
                    else:
                        raise NotImplementedError

                    y = self.avgpool(y)
                    y = y.view(y.size(0), -1)
                    embeding_sum = embeding_sum + y
                    outputs['raw_features_{}'.format(index)] = y
                    y = eval('self.branch_fc_{}'.format(index))(y)
                    logits_sum = logits_sum + y
                    outputs['logits_{}'.format(index)] = y

                outputs['embedding_avg'] = embeding_sum / (self.args.mtd['num'] + 1)
                if 'output' in self.args.mtd:
                    if self.args.mtd['output'] == 'average':
                        outputs['logits'] = logits_sum / (self.args.mtd['num'] + 1)
                    if self.args.mtd['output'] == 'origin':
                        outputs['logits'] = out['logits_0']
                    if self.args.mtd['output'] == 'max':
                        logits = []
                        for index in range(self.args.mtd['num'] + 1):
                            logits.append(outputs['logits_{}'.format(index)].unsqueeze(dim=0))
                            # logits.append(F.normalize(outputs['logits_{}'.format(index)], p=2, dim=1).unsqueeze(dim=0))
                        logits = torch.cat(logits, dim=0).max(dim=0)[0]
                        outputs['logits'] = logits
                else:
                    outputs['logits'] = out['logits_0']
                return outputs

            else:
                out = self.layer0(x)
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.avgpool(out)
                out = out.view(out.size(0), -1)
                out = self.fc(out)
                return out

        elif 'MNIST' in self.args.dataset:
            if self.forward_with_branch:

                if len(x.shape) > 2:
                    x = x.reshape(x.shape[0], -1)
                out = self.body.layer_0(x)
                feature = out
                out = self.body.layer_1(out)
                embeding = out
                out = self.fc(out)

                outputs = {}
                logits_sum = out
                embeding_sum = embeding
                outputs['raw_features'] = embeding
                outputs['logits_0'] = out
                for index in range(1, self.args.mtd['num'] + 1):

                    if self.training and 'noise' in self.args.mtd:
                        noise = torch.randn_like(feature).to(feature.device)
                        # noise = noise / torch.norm(noise)
                        feature = feature + noise * self.args.mtd['noise']

                    y = eval('self.branch_layer1_{}'.format(index))(feature)
                    y = F.relu(y)
                    embeding_sum = embeding_sum + y
                    outputs['raw_features_{}'.format(index)] = y
                    y = eval('self.branch_fc_{}'.format(index))(y)
                    logits_sum = logits_sum + y
                    outputs['logits_{}'.format(index)] = y

                outputs['embedding_avg'] = embeding_sum / (self.args.mtd['num'] + 1)
                if 'output' in self.args.mtd:
                    if self.args.mtd['output'] == 'average':
                        outputs['logits'] = logits_sum / (self.args.mtd['num'] + 1)
                    if self.args.mtd['output'] == 'origin':
                        outputs['logits'] = out['logits_0']
                    if self.args.mtd['output'] == 'max':
                        logits = []
                        for index in range(self.args.mtd['num'] + 1):
                            logits.append(outputs['logits_{}'.format(index)].unsqueeze(dim=0))
                            # logits.append(F.normalize(outputs['logits_{}'.format(index)], p=2, dim=1).unsqueeze(dim=0))
                        logits = torch.cat(logits, dim=0).max(dim=0)[0]
                        outputs['logits'] = logits
                else:
                    outputs['logits'] = out['logits_0']
                return outputs

            else:
                if len(x.shape) > 2:
                    x = x.reshape(x.shape[0], -1)
                x = self.body(x)
                x = self.fc(x)
                return x


if __name__ == '__main__':
    x = torch.randn(1, 3, 64, 64)
    iconv = IdentityConv2d(3, requires_grad=False)
    y = iconv(x)
    print(torch.all(torch.eq(x, y)))
