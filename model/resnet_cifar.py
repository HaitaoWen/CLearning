import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn.modules._functions import SyncBatchNorm as sync_batch_norm


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.residual_function1 = nn.Sequential(
            conv3x3(inplanes, planes, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes)
        )
        if downsample is None:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = downsample
        self.residual_function2 = nn.Sequential(
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.residual_function1(x)
        out += self.shortcut(x)
        out = self.residual_function2(out)
        return out


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.pointer = -1
        self.counter_bn = 0
        self.path = OrderedDict()
        self.freeze_bn_statistic = False

    def init_path(self):
        for key, value in self.named_parameters():
            self.path[key] = torch.nn.Parameter(0.0 * torch.ones_like(value), requires_grad=True)

    def parameters_dim(self):
        dim = 0
        for value in self.parameters():
            dim += value.numel()
        return dim

    def load_flat_path(self, p):
        start = 0
        stop = 0
        for key, value in self.path.items():
            stop = stop + value.numel()
            p_ = p[start:stop].reshape(value.shape)
            value.data = p_.data
            start = stop

    def flat_path(self):
        p = []
        for key, value in self.path.items():
            p.append(value.view(1, -1))
        p = torch.cat(p, dim=1)
        return p

    def load_grad(self, grad: OrderedDict):
        for k, v in self.named_parameters():
            v.grad = grad[k]

    def load_parameter(self, p):
        for key, value in self.named_parameters():
            value.data = p[key].data

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m.weight is not None:
                    nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_with_parameters(self, modules, parameters, x):
        for name, module in modules.named_children():
            # ****************** Sequential ******************* #
            if type(module) is torch.nn.Sequential:
                x = self.forward_with_parameters(module, parameters, x)
            # ****************** Conv2d ******************* #
            elif type(module) is torch.nn.Conv2d:
                self.pointer += 1
                weight = parameters[self.pointer][1]
                if module.bias is None:
                    bias = None
                else:
                    self.pointer += 1
                    bias = parameters[self.pointer][1]
                x = torch.nn.functional.conv2d(x, weight=weight, bias=bias, stride=module.stride,
                                               padding=module.padding, dilation=module.dilation, groups=module.groups)
            # ****************** BatchNorm2d ******************* #
            elif type(module) is torch.nn.BatchNorm2d:
                # ****************** affine ***************** #
                if module.affine is False:
                    weight = None
                    bias = None
                else:
                    self.pointer += 1
                    weight = parameters[self.pointer][1]
                    if module.bias is None:
                        bias = None
                    else:
                        self.pointer += 1
                        bias = parameters[self.pointer][1]
                # ***************** statistics **************** #
                if ('running_mean' in parameters[self.pointer + 1][0]) and \
                        ('running_var' in parameters[self.pointer + 2][0]):
                    # meta-SGD, model's mean and var is static,
                    # but, parameters' mean and var is changing.
                    self.pointer += 1
                    running_mean = parameters[self.pointer][1]
                    self.pointer += 1
                    running_var = parameters[self.pointer][1]
                    # point to num_batches_tracked
                    self.pointer += 1
                    parameters[self.pointer][1].data += 1
                    track_running_stats = True
                else:
                    # MAML, model's mean and var is changing.
                    state = list(self.state_dict().items())
                    if 'running_mean' in state[self.pointer + self.counter_bn + 1][0] and 'running_var' in \
                            state[self.pointer + self.counter_bn + 2][0]:
                        self.counter_bn += 1
                        running_mean = state[self.pointer + self.counter_bn][1]
                        self.counter_bn += 1
                        running_var = state[self.pointer + self.counter_bn][1]
                        # point to num_batches_tracked
                        self.counter_bn += 1
                        state[self.pointer + self.counter_bn][1].data += 1
                        track_running_stats = True
                        # TODO: (there is a potential bug here, if momentum is None, exponential_average_factor is set
                        #  to 1 / num_batches_tracked (training) or 0 (test))
                    else:
                        running_mean = None
                        running_var = None
                        track_running_stats = False
                x = torch.nn.functional.batch_norm(x,
                                                   running_mean=running_mean.clone()
                                                   if self.freeze_bn_statistic and running_mean is not None
                                                   else running_mean,
                                                   running_var=running_var.clone()
                                                   if self.freeze_bn_statistic and running_var is not None
                                                   else running_var,
                                                   weight=weight, bias=bias,
                                                   training=module.training or not track_running_stats,
                                                   momentum=module.momentum, eps=module.eps)
            # ****************** SyncBatchNorm ******************* #
            elif type(module) is torch.nn.SyncBatchNorm:
                # ****************** affine ***************** #
                if module.affine is False:
                    weight = None
                    bias = None
                else:
                    self.pointer += 1
                    weight = parameters[self.pointer][1]
                    if module.bias is None:
                        bias = None
                    else:
                        self.pointer += 1
                        bias = parameters[self.pointer][1]
                # ***************** statistics **************** #
                if ('running_mean' in parameters[self.pointer + 1][0]) and \
                        ('running_var' in parameters[self.pointer + 2][0]):
                    # meta-SGD, model's mean and var is static,
                    # but, parameters' mean and var is changing.
                    self.pointer += 1
                    running_mean = parameters[self.pointer][1]
                    self.pointer += 1
                    running_var = parameters[self.pointer][1]
                    # point to num_batches_tracked
                    self.pointer += 1
                    parameters[self.pointer][1].data += 1
                    track_running_stats = True
                else:
                    # MAML, model's mean and var is changing.
                    state = list(self.state_dict().items())
                    if 'running_mean' in state[self.pointer + self.counter_bn + 1][0] and 'running_var' in \
                            state[self.pointer + self.counter_bn + 2][0]:
                        self.counter_bn += 1
                        running_mean = state[self.pointer + self.counter_bn][1]
                        self.counter_bn += 1
                        running_var = state[self.pointer + self.counter_bn][1]
                        # point to num_batches_tracked
                        self.counter_bn += 1
                        state[self.pointer + self.counter_bn][1].data += 1
                        track_running_stats = True
                        # TODO: (there is a potential bug here, if momentum is None, exponential_average_factor is set
                        #  to 1 / num_batches_tracked (training) or 0 (test))
                    else:
                        running_mean = None
                        running_var = None
                        track_running_stats = False
                # deal with sync
                if module.training:
                    bn_training = True
                else:
                    bn_training = (module.running_mean is None) and (module.running_var is None)
                # If buffers are not to be tracked, ensure that they won't be updated
                running_mean = (running_mean if not module.training or track_running_stats else None)
                running_var = (running_var if not module.training or track_running_stats else None)
                # Don't sync batchnorm stats in inference mode (model.eval()).
                need_sync = (bn_training and module.training)
                if need_sync:
                    process_group = torch.distributed.group.WORLD
                    if module.process_group:
                        process_group = module.process_group
                    world_size = torch.distributed.get_world_size(process_group)
                    need_sync = world_size > 1
                if not need_sync:
                    x = torch.nn.functional.batch_norm(x,
                                                       running_mean.clone()
                                                       if self.freeze_bn_statistic and running_mean is not None
                                                       else running_mean,
                                                       running_var.clone()
                                                       if self.freeze_bn_statistic and running_var is not None
                                                       else running_var,
                                                       weight, bias,
                                                       bn_training, module.momentum, module.eps)
                else:
                    assert bn_training
                    x = sync_batch_norm.apply(x, weight, bias,
                                              running_mean.clone()
                                              if self.freeze_bn_statistic and running_mean is not None
                                              else running_mean,
                                              running_var.clone()
                                              if self.freeze_bn_statistic and running_var is not None
                                              else running_var,
                                              module.eps,
                                              module.momentum, process_group, world_size)
            # ****************** DerBatchNorm2d ******************* #
            elif type(module) is DerBatchNorm2d:
                eps = module.eps
                self.pointer += 1
                weight = parameters[self.pointer][1]
                self.pointer += 1
                bias = parameters[self.pointer][1]
                self.pointer += 1
                running_mean = parameters[self.pointer][1]
                self.pointer += 1
                running_var = parameters[self.pointer][1]
                # TODO, DBN
                self.pointer += 1
                num_batches_tracked = parameters[self.pointer][1]
                # TODO, waiting for modification
                running_var = torch.nn.functional.relu(running_var)
                x_hat = (x - running_mean[None, :, None, None]) / torch.sqrt(running_var[None, :, None, None] + eps)
                x = weight[None, :, None, None] * x_hat + bias[None, :, None, None]
            # ****************** ReLU ******************* #
            elif type(module) is torch.nn.ReLU:
                # TODO, dedicated to distillation
                if module._forward_hooks:
                    for hook in module._forward_hooks.values():
                        hook(module, (x,), None)
                x = torch.nn.functional.relu(x, inplace=module.inplace)
            # ****************** Linear ******************* #
            elif type(module) is torch.nn.Linear:
                self.pointer += 1
                weight = parameters[self.pointer][1]
                self.pointer += 1
                bias = parameters[self.pointer][1]
                if x.shape[2:4] == (1, 1):
                    x = x.view(x.shape[0], -1)
                input = x.clone()
                x = torch.nn.functional.linear(x, weight=weight, bias=bias)
                output = x.clone()
                if module._forward_hooks:
                    for hook in module._forward_hooks.values():
                        hook(module, (input,), output)
            # ****************** AdaptiveAvgPool2d ******************* #
            elif type(module) is torch.nn.AvgPool2d:
                x = torch.nn.functional.avg_pool2d(x, module.kernel_size, module.stride,
                                                   module.padding, module.ceil_mode, module.count_include_pad)
            # ********************* MaxPool2d ************************ #
            elif type(module) is torch.nn.MaxPool2d:
                x = torch.nn.functional.max_pool2d(x, module.kernel_size, module.stride, module.padding,
                                                   module.dilation, module.ceil_mode, module.return_indices)
            # *********************** Dropout ************************ #
            elif type(module) is torch.nn.Dropout:
                p = module.p
                inplace = module.inplace
                training = module.training
                x = torch.nn.functional.dropout(x, p=p, training=training, inplace=inplace)
            # ****************** BasicBlock ******************* #
            elif type(module) is BasicBlock:
                x1 = self.forward_with_parameters(module.residual_function1, parameters, x)
                x2 = self.forward_with_parameters(module.shortcut, parameters, x)
                x = x1 + x2
                x = self.forward_with_parameters(module.residual_function2, parameters, x)
        return x


class ResNet(BaseNet):

    def __init__(self, args, block, layers, num_classes=100):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, parameters=None):
        if parameters is None:
            out = self.layer0(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
        else:
            self.pointer = -1
            self.counter_bn = 0
            assert type(parameters) is OrderedDict
            parameters = list(parameters.items())
            out = self.forward_with_parameters(self, parameters, x)
        return out


def resnet32(args=None, **kwargs):
    n = 5
    model = ResNet(args, BasicBlock, [n, n, n], **kwargs)
    return model


if __name__ == "__main__":
    def detach_parameters(model):
        parameters = OrderedDict()
        for key, value in model.named_parameters():
            parameters[key] = value.clone().detach()
        return parameters

    def detach_state(model):
        state = OrderedDict()
        for key, value in model.state_dict().items():
            state[key] = value.clone().detach()
        return state

    model = resnet32()
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    parameters = detach_parameters(model)
    state = detach_state(model)
    x = torch.randn(2, 3, 32, 32)
    # x = torch.randn(2, 784)
    y_ = model(x, parameters=state)
    y = model(x)
    print(y - y_)
    a = 1
