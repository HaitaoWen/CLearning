import torch
from .tool import *
from torch import nn
from copy import deepcopy
import torch.nn.functional as F
from collections import OrderedDict


class MLP(nn.Module):
    def __init__(self, args, num_classes=10):
        super(MLP, self).__init__()
        self.args = args
        n_hidden = args.n_hidden
        self.act = OrderedDict()
        self.lin1 = nn.Linear(784, n_hidden, bias=False)
        self.lin2 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.fc1 = nn.Linear(n_hidden, num_classes, bias=False)

    def forward(self, x):
        self.act['Lin1'] = x
        x = self.lin1(x)
        x = F.relu(x)
        self.act['Lin2'] = x
        x = self.lin2(x)
        x = F.relu(x)
        self.act['fc1'] = x
        x = self.fc1(x)
        return x


class AlexNet(nn.Module):
    def __init__(self, args, num_classes=100):
        super(AlexNet, self).__init__()
        self.args = args
        self.act = OrderedDict()
        self.map = []
        self.ksize = []
        self.in_channel = []
        self.map.append(32)
        self.conv1 = nn.Conv2d(3, 64, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        s = compute_conv_output_size(32, 4)
        s = s // 2
        self.ksize.append(4)
        self.in_channel.append(3)
        self.map.append(s)
        self.conv2 = nn.Conv2d(64, 128, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=False)
        s = compute_conv_output_size(s, 3)
        s = s // 2
        self.ksize.append(3)
        self.in_channel.append(64)
        self.map.append(s)
        self.conv3 = nn.Conv2d(128, 256, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(256, track_running_stats=False)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.smid = s
        self.ksize.append(2)
        self.in_channel.append(128)
        self.map.append(256 * self.smid * self.smid)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)

        self.fc1 = nn.Linear(256 * self.smid * self.smid, 2048, bias=False)
        self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.fc2 = nn.Linear(2048, 2048, bias=False)
        self.bn5 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.map.extend([2048])

        i, n, self.taskcla = 0, 0, []
        avg = num_classes // args.tasks
        while True:
            n += avg
            if n <= num_classes:
                self.taskcla.append((i, avg))
            else:
                break
            i += 1
        self.fc3 = torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.fc3.append(torch.nn.Linear(2048, n, bias=False))

    def forward(self, x):
        bsz = deepcopy(x.size(0))
        self.act['conv1'] = x
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

        self.act['conv2'] = x
        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

        self.act['conv3'] = x
        x = self.conv3(x)
        x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

        x = x.view(bsz, -1)
        self.act['fc1'] = x
        x = self.fc1(x)
        x = self.drop2(self.relu(self.bn4(x)))

        self.act['fc2'] = x
        x = self.fc2(x)
        x = self.drop2(self.relu(self.bn5(x)))

        y = []
        for t, i in self.taskcla:
            y.append(self.fc3[t](x))

        return y


class LeNet(nn.Module):
    def __init__(self, args, num_classes=100):
        super(LeNet, self).__init__()
        self.map = []
        self.ksize = []
        self.in_channel = []
        self.act = OrderedDict()

        self.map.append(32)
        self.conv1 = nn.Conv2d(3, 20, 5, bias=False, padding=2)

        s = compute_conv_output_size(32, 5, 1, 2)
        s = compute_conv_output_size(s, 3, 2, 1)
        self.ksize.append(5)
        self.in_channel.append(3)
        self.map.append(s)
        self.conv2 = nn.Conv2d(20, 50, 5, bias=False, padding=2)

        s = compute_conv_output_size(s, 5, 1, 2)
        s = compute_conv_output_size(s, 3, 2, 1)
        self.ksize.append(5)
        self.in_channel.append(20)
        self.smid = s
        self.map.append(50 * self.smid * self.smid)
        self.maxpool = torch.nn.MaxPool2d(3, 2, padding=1)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0)
        self.drop2 = torch.nn.Dropout(0)
        self.lrn = torch.nn.LocalResponseNorm(4, 0.001 / 9.0, 0.75, 1)

        self.fc1 = nn.Linear(50 * self.smid * self.smid, 800, bias=False)
        self.fc2 = nn.Linear(800, 500, bias=False)
        self.map.extend([800])

        i, n, self.taskcla = 0, 0, []
        avg = num_classes // args.tasks
        while True:
            n += avg
            if n <= num_classes:
                self.taskcla.append((i, avg))
            else:
                break
            i += 1
        self.fc3 = torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.fc3.append(torch.nn.Linear(500, n, bias=False))

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                # torch.nn.init.xavier_uniform(m.weight)
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        bsz = deepcopy(x.size(0))
        self.act['conv1'] = x
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.lrn(self.relu(x))))

        self.act['conv2'] = x
        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.lrn(self.relu(x))))

        x = x.reshape(bsz, -1)
        self.act['fc1'] = x
        x = self.fc1(x)
        x = self.drop2(self.relu(x))

        self.act['fc2'] = x
        x = self.fc2(x)
        x = self.drop2(self.relu(x))

        y = []
        for t, i in self.taskcla:
            y.append(self.fc3[t](x))

        return y


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, args=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes,
                               track_running_stats=False)
            )
        self.act = OrderedDict()
        self.count = 0

    def forward(self, x):
        self.count = self.count % 2
        self.act['conv_{}'.format(self.count)] = x
        self.count += 1
        out = F.relu(self.bn1(self.conv1(x)))
        self.count = self.count % 2
        self.act['conv_{}'.format(self.count)] = out
        self.count += 1
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, args, num_classes=100):
        super(ResNet18, self).__init__()
        nf = args.nf
        block = BasicBlock
        num_blocks = [2, 2, 2, 2]

        self.args = args
        self.in_planes = nf
        if args.dataset == 'FIVE':
            self.conv1 = conv3x3(3, nf * 1, 1)
        elif args.dataset == 'miniImageNet':
            self.conv1 = conv3x3(3, nf * 1, 2)
        else:
            raise NotImplementedError
        self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=False)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        i, n, self.taskcla = 0, 0, []
        avg = num_classes // args.tasks
        while True:
            n += avg
            if n <= num_classes:
                self.taskcla.append((i, avg))
            else:
                break
            i += 1
        self.linear = torch.nn.ModuleList()
        for t, n in self.taskcla:
            if args.dataset == 'FIVE':
                self.linear.append(nn.Linear(nf * 8 * block.expansion * 4, n, bias=False))
            elif args.dataset == 'miniImageNet':
                self.linear.append(nn.Linear(nf * 8 * block.expansion * 9, n, bias=False))
            else:
                raise NotImplementedError
        self.act = OrderedDict()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.args))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        if self.args.dataset == 'FIVE':
            self.act['conv_in'] = x.view(bsz, 3, 32, 32)
            out = F.relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))))
        elif self.args.dataset == 'miniImageNet':
            self.act['conv_in'] = x.view(bsz, 3, 84, 84)
            out = F.relu(self.bn1(self.conv1(x.view(bsz, 3, 84, 84))))
        else:
            raise NotImplementedError
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        y = []
        for t, i in self.taskcla:
            y.append(self.linear[t](out))
        return y
