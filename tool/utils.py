import os
import sys
import time
import copy
import torch
import pickle
import random
import shutil
import requests
import warnings
import importlib
import threading
import numpy as np
from PIL import Image
from .parser import args
from torch import distributed
from datetime import datetime
from yacs.config import CfgNode
from collections import defaultdict
from collections import OrderedDict
from torch.utils.data import Dataset
from continuum import ClassIncremental
from model.resnet_cifar import resnet32
from model.resnet_imagenet import resnet18
from continuum import Permutations, Rotations
from torchvision.transforms import transforms
from model.model_task_il import MLP, ResNet18_Task
from torch.nn.parallel.scatter_gather import gather
import torchvision.transforms.functional as TorchVisionFunc
from continuum.datasets import MNIST, SVHN, CIFAR10, CIFAR100, TinyImageNet200, \
    DTD, ImageNet100, CUB200, ImageNet1000, ImageFolderDataset
warnings.filterwarnings("ignore", category=RuntimeWarning)


# **************** input ****************** #
datainfo = {
    'MNIST': {
        'classnum': 10
    },
    'PMNIST': {
        'classnum': 10
    },
    'RMNIST': {
        'classnum': 10
    },
    'FIVE': {
        'classnum': 50
    },
    'SVHN': {
        'classnum': 10
    },
    'CIFAR10': {
        'classnum': 10
    },
    'CIFAR100': {
        'classnum': 100
    },
    'SUPER': {
        'classnum': 100
    },
    'TinyImageNet': {
        'classnum': 200
    },
    'miniImageNet': {
        'classnum': 100
    },
    'DTD': {
        'classnum': 47
    },
    'ImageNet100': {
        'classnum': 100
    },
    'CUB': {
        'classnum': 200
    },
    'ImageNet1000': {
        'classnum': 1000
    }
}


Transforms = {
        'MNIST':
            {'train': [transforms.ToTensor()],
             'eval': [transforms.ToTensor()]},

        'PMNIST':
            {'train': [transforms.ToTensor()],
             'eval': [transforms.ToTensor()]},

        'RMNIST':
            {'train': [transforms.ToTensor()],
             'eval': [transforms.ToTensor()]},

        'SVHN':
            {'train': [
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
            'eval':[
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            },

        'CIFAR10':
            {'train': [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2471, 0.2435, 0.2616])],
                'eval': [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2471, 0.2435, 0.2616])]},

        'CIFAR100':
            {'train': [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4866, 0.4409],
                                     std=[0.2009, 0.1984, 0.2023])],
                'eval': [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5071, 0.4866, 0.4409],
                                         std=[0.2009, 0.1984, 0.2023])]},

        'TinyImageNet':
            {'train': [
                # transforms.ToPILImage(),
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(brightness=63 / 255),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                     std=[0.2302, 0.2265, 0.2262])],
                'eval': [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                         std=[0.2302, 0.2265, 0.2262])]},

        'DTD':
            {'train': [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])],
                'eval': [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])]},

        'ImageNet100':
            {'train': [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])],
                'eval': [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])]},

        'CUB':
            {'train': [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])],
                'eval': [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])]},

        'ImageNet1000':
            {'train': [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])],
                'eval': [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])]},
}


def load_scenario():

    def arrange_increment():
        if args.init is None or args.init == 0:
            increment_ = datainfo[args.dataset]['classnum'] / args.tasks
        else:
            assert isinstance(args.init, int)
            left = datainfo[args.dataset]['classnum'] - args.init
            increment_ = left / (args.tasks - 1)
        if not increment_.is_integer():
            raise ValueError('number of classes {} should be divided by number of tasks {}'.format(
                datainfo[args.dataset]['classnum'] if args.init is None else left,
                args.tasks if args.init is None else args.tasks-1))
        else:
            increment_ = int(increment_)
        return increment_

    # def get_order_map(scenario_train_):
    #     class_order_ = scenario_train_.class_order.tolist()
    #     labelmap_ = []
    #     for i in range(len(class_order_)):
    #         labelmap_.append(class_order_.index(i))
    #     return class_order_, labelmap_

    if hasattr(args, 'datafun'):
        if args.datafun:
            index = args.datafun.rfind('.')
            package = args.datafun[:index]
            object_ = args.datafun[index + 1:]
            modules = importlib.import_module(package)
            scenario_train, scenario_eval, increments = eval('modules.{}'.format(object_))(args)
            args.increments = increments
        else:
            raise ValueError('You have specified a function for this dataset: {}, however, '
                             'the function path is empty!'.format(args.datafun))

    elif args.dataset == 'MNIST':
        increment = arrange_increment()
        if args.order is not None:
            file = open('config/orders.yaml')
            order = CfgNode.load_cfg(file)[args.dataset]['order'][args.order]
        else:
            order = None
        scenario_train = ClassIncremental(
            MNIST(data_path=args.datadir, download=True, train=True),
            increment=increment,
            initial_increment=args.init if args.init is not None else 0,
            transformations=Transforms[args.dataset]['train'],
            class_order=order
        )
        scenario_eval = ClassIncremental(
            MNIST(data_path=args.datadir, download=True, train=False),
            increment=increment,
            initial_increment=args.init if args.init is not None else 0,
            transformations=Transforms[args.dataset]['eval'],
            class_order=order
        )
        args.increments = scenario_train.increments
        # args.class_order, args.labelmap = get_order_map(scenario_train)

    elif args.dataset == 'PMNIST':
        scenario_train = Permutations(
            MNIST(data_path=args.datadir, download=True, train=True),
            nb_tasks=args.tasks,
            base_transformations=Transforms[args.dataset]['train'],
            seed=args.seed,
            shared_label_space=True
        )
        scenario_eval = Permutations(
            MNIST(data_path=args.datadir, download=True, train=False),
            nb_tasks=args.tasks,
            base_transformations=Transforms[args.dataset]['eval'],
            seed=args.seed,
            shared_label_space=True
        )
        args.increments = [datainfo[args.dataset]['classnum']] * args.tasks
        # args.class_order = [i for i in range(datainfo[args.dataset]['classnum']*args.tasks)]
        # args.labelmap = args.class_order
    elif args.dataset == 'RMNIST':
        trsf = []
        degrees = []
        # MC-SGD implementation
        per_task_rotation = 9
        for task_id in range(args.tasks):
            degree = task_id * per_task_rotation
            degrees.append(degree)
            # trsf.append([RotationTransform(degree)])
        # Stable-SGD implementation
        # per_task_rotation = 10
        # for task_id in range(args.tasks):
        #     rotation_degree = task_id * per_task_rotation
        #     rotation_degree -= (np.random.random() * per_task_rotation)
        #     degrees.append(int(rotation_degree))
        #     trsf.append([RotationTransform(rotation_degree)])
        scenario_train = Rotations(
            MNIST(data_path=args.datadir, download=True, train=True),
            nb_tasks=args.tasks,
            base_transformations=Transforms[args.dataset]['train'],
            list_degrees=degrees,
            shared_label_space=True
        )
        scenario_eval = Rotations(
            MNIST(data_path=args.datadir, download=True, train=False),
            nb_tasks=args.tasks,
            base_transformations=Transforms[args.dataset]['eval'],
            list_degrees=degrees,
            shared_label_space=True
        )
        # scenario_train.inc_trsf = trsf
        # scenario_eval.inc_trsf = trsf
        args.increments = [datainfo[args.dataset]['classnum']] * args.tasks
        # args.class_order = [i for i in range(datainfo[args.dataset]['classnum']*args.tasks)]
        # args.labelmap = args.class_order
    elif args.dataset == 'SVHN':
        increment = arrange_increment()
        if args.order is not None:
            file = open('config/orders.yaml')
            order = CfgNode.load_cfg(file)[args.dataset]['order'][args.order]
        else:
            order = None
        scenario_train = ClassIncremental(
            SVHN(data_path=args.datadir, download=True, train=True),
            increment=increment,
            initial_increment=args.init if args.init is not None else 0,
            transformations=Transforms[args.dataset]['train'],
            class_order=order
        )
        scenario_eval = ClassIncremental(
            SVHN(data_path=args.datadir, download=True, train=False),
            increment=increment,
            initial_increment=args.init if args.init is not None else 0,
            transformations=Transforms[args.dataset]['eval'],
            class_order=order
        )
        args.increments = scenario_train.increments
        # args.class_order, args.labelmap = get_order_map(scenario_train)
    elif args.dataset == 'CIFAR10':
        increment = arrange_increment()
        if args.order is not None:
            file = open('config/orders.yaml')
            order = CfgNode.load_cfg(file)[args.dataset]['order'][args.order]
        else:
            order = None
        scenario_train = ClassIncremental(
            CIFAR10(data_path=args.datadir, download=True, train=True),
            increment=increment,
            initial_increment=args.init if args.init is not None else 0,
            transformations=Transforms[args.dataset]['train'],
            class_order=order
        )
        scenario_eval = ClassIncremental(
            CIFAR10(data_path=args.datadir, download=True, train=False),
            increment=increment,
            initial_increment=args.init if args.init is not None else 0,
            transformations=Transforms[args.dataset]['eval'],
            class_order=order
        )
        args.increments = scenario_train.increments
        # args.class_order, args.labelmap = get_order_map(scenario_train)
    elif args.dataset == 'CIFAR100':
        increment = arrange_increment()
        if args.order is not None:
            file = open('config/orders.yaml')
            order = CfgNode.load_cfg(file)[args.dataset]['order'][args.order]
        else:
            order = None
        scenario_train = ClassIncremental(
            CIFAR100(data_path=args.datadir, download=True, train=True),
            increment=increment,
            initial_increment=args.init if args.init is not None else 0,
            transformations=Transforms[args.dataset]['train'],
            class_order=order
        )
        scenario_eval = ClassIncremental(
            CIFAR100(data_path=args.datadir, download=True, train=False),
            increment=increment,
            initial_increment=args.init if args.init is not None else 0,
            transformations=Transforms[args.dataset]['eval'],
            class_order=order
        )
        args.increments = scenario_train.increments
        # args.class_order, args.labelmap = get_order_map(scenario_train)
    elif args.dataset == 'TinyImageNet':
        increment = arrange_increment()
        scenario_train = ClassIncremental(
            TinyImageNet200(data_path=args.datadir, download=True, train=True),
            increment=increment,
            initial_increment=args.init if args.init is not None else 0,
            transformations=Transforms[args.dataset]['train']
        )
        scenario_eval = ClassIncremental(
            TinyImageNet200(data_path=args.datadir, download=True, train=False),
            increment=increment,
            initial_increment=args.init if args.init is not None else 0,
            transformations=Transforms[args.dataset]['eval']
        )
        args.increments = scenario_train.increments
        # args.class_order, args.labelmap = get_order_map(scenario_train)
    elif args.dataset == 'DTD':
        increment = arrange_increment()
        if args.order is not None:
            file = open('config/orders.yaml')
            order = CfgNode.load_cfg(file)[args.dataset]['order'][args.order]
        else:
            order = None
        # TODO, continuum default data split is not corresponding to the split of LUCIR under seed 1993.
        #  we use the class "ImageFolderDataset" to keep consistent with LUCIR.
        scenario_train = ClassIncremental(
            DTD(data_path=args.datadir, download=True, train=True),
            increment=increment,
            initial_increment=args.init if args.init is not None else 0,
            transformations=Transforms[args.dataset]['train'],
            class_order=order
        )
        scenario_eval = ClassIncremental(
            DTD(data_path=args.datadir, download=True, train=False),
            increment=increment,
            initial_increment=args.init if args.init is not None else 0,
            transformations=Transforms[args.dataset]['eval'],
            class_order=order
        )
        args.increments = scenario_train.increments
        # args.class_order, args.labelmap = get_order_map(scenario_train)
    elif args.dataset == 'ImageNet100':
        increment = arrange_increment()
        if args.order is not None:
            file = open('config/orders.yaml')
            order = CfgNode.load_cfg(file)[args.dataset]['order'][args.order]
        else:
            order = None
        # TODO, continuum default data split is not corresponding to the split of LUCIR under seed 1993.
        #  we use the class "ImageFolderDataset" to keep consistent with LUCIR.
        # scenario_train = ClassIncremental(
        #     ImageNet100(data_path=args.datadir+'/ImageNet/', download=True, train=True),
        #     increment=increment,
        #     initial_increment=args.init if args.init is not None else 0,
        #     transformations=Transforms[args.dataset]['train'],
        #     class_order=order
        # )
        # scenario_eval = ClassIncremental(
        #     ImageNet100(data_path=args.datadir+'/ImageNet/', download=True, train=False),
        #     increment=increment,
        #     initial_increment=args.init if args.init is not None else 0,
        #     transformations=Transforms[args.dataset]['eval'],
        #     class_order=order
        # )
        scenario_train = ClassIncremental(
            ImageFolderDataset(data_path=args.datadir+'/ImageNet100/train', download=True, train=True),
            increment=increment,
            initial_increment=args.init if args.init is not None else 0,
            transformations=Transforms[args.dataset]['train'],
            class_order=order
        )
        scenario_eval = ClassIncremental(
            ImageFolderDataset(data_path=args.datadir+'/ImageNet100/val', download=True, train=False),
            increment=increment,
            initial_increment=args.init if args.init is not None else 0,
            transformations=Transforms[args.dataset]['eval'],
            class_order=order
        )
        args.increments = scenario_train.increments
        # args.class_order, args.labelmap = get_order_map(scenario_train)
    elif args.dataset == 'CUB':
        increment = arrange_increment()
        if args.order is not None:
            file = open('config/orders.yaml')
            order = CfgNode.load_cfg(file)[args.dataset]['order'][args.order]
        else:
            order = None
        # TODO, continuum default data split is not corresponding to the split of LUCIR under seed 1993.
        #  we use the class "ImageFolderDataset" to keep consistent with LUCIR.
        scenario_train = ClassIncremental(
            CUB200(data_path=args.datadir, download=True, train=True),
            increment=increment,
            initial_increment=args.init if args.init is not None else 0,
            transformations=Transforms[args.dataset]['train'],
            class_order=order
        )
        scenario_eval = ClassIncremental(
            CUB200(data_path=args.datadir, download=True, train=False),
            increment=increment,
            initial_increment=args.init if args.init is not None else 0,
            transformations=Transforms[args.dataset]['eval'],
            class_order=order
        )
        args.increments = scenario_train.increments
        # args.class_order, args.labelmap = get_order_map(scenario_train)
    elif args.dataset == 'ImageNet1000':
        increment = arrange_increment()
        if args.order is not None:
            file = open('config/orders.yaml')
            order = CfgNode.load_cfg(file)[args.dataset]['order'][args.order]
        else:
            order = None
        scenario_train = ClassIncremental(
            ImageNet1000(data_path=args.datadir+'/ImageNet/', download=True, train=True),
            increment=increment,
            initial_increment=args.init if args.init is not None else 0,
            transformations=Transforms[args.dataset]['train'],
            class_order=order
        )
        scenario_eval = ClassIncremental(
            ImageNet1000(data_path=args.datadir+'/ImageNet/', download=True, train=False),
            increment=increment,
            initial_increment=args.init if args.init is not None else 0,
            transformations=Transforms[args.dataset]['eval'],
            class_order=order
        )
        args.increments = scenario_train.increments
        # args.class_order, args.labelmap = get_order_map(scenario_train)
    else:
        raise ValueError('{} data is not supported!'.format(args.dataset))

    # ***************** MultiTask ***************** #
    if args.scheme == 'MultiTask':
        scenario_train = [scenario_train]
    if args.debug:
        # if args.snapshot:
        #     raise ValueError('Considering the limitation of storage, private variables can not be saved')
        setattr(memory, 'scenario_train', scenario_train)
        setattr(memory, 'scenario_eval', scenario_eval)
    return scenario_train, scenario_eval


class RotationTransform:
    """
    Rotation transforms for the images in `Rotation MNIST` dataset.
    """
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TorchVisionFunc.rotate(x, self.angle, fill=(0,))


class CommonData(Dataset):
    def __init__(self, traindata, transform):
        self.traindata = traindata
        self.transform = transform

    def __getitem__(self, index):
        x, y, t = self.traindata[index]
        if self.transform is not None:
            x = Image.fromarray(x.astype("uint8"))
            x = self.transform(x)
        return x, y, t

    def __len__(self):
        return len(self.traindata)


class CIFARData(Dataset):
    def __init__(self, traindata, memory):
        self.transform = traindata.trsf
        _x, _y, _t = traindata._x, traindata._y, traindata._t
        self.x = np.concatenate((_x, memory.x), axis=0)
        self.y = np.concatenate((_y, memory.y), axis=0)
        self.t = np.concatenate((_t, memory.t), axis=0)

    def __getitem__(self, index):
        x, y, t = self.x[index], self.y[index], self.t[index]
        if self.transform is not None:
            x = Image.fromarray(x.astype("uint8"))
            x = self.transform(x)
        return x, y, t

    def __len__(self):
        return len(self.x)


class MNISTData(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class Images(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        if self.transform is not None:
            if 'ImageNet' in args.dataset or args.dataset in ('DTD', 'CUB'):
                image = Image.open(image).convert("RGB")
            else:
                image = Image.fromarray(image.astype("uint8"))
            image = self.transform(image)
        return image, index

    def __len__(self):
        return len(self.images)


class Memorydata(Dataset):
    def __init__(self, mem, transform):
        if args.dataset in ('CIFAR10', 'CIFAR100', 'SVHN', 'FIVE'):
            self.transform = transform
            self.x, self.y, self.t = mem.x, mem.y, mem.t
        elif 'MNIST' in args.dataset:
            self.transform = None
            self.x, self.y, self.t = zip(*mem.x)
        else:
            raise ValueError

    def __getitem__(self, index):
        x, y, t = self.x[index], self.y[index], self.t[index]
        if self.transform is not None:
            x = Image.fromarray(x.astype("uint8"))
            x = self.transform(x)
        return x, y, t

    def __len__(self):
        return len(self.x)


class ImageNetData(Dataset):
    def __init__(self, x, y, t, transform):
        self.transform = transform
        self.x, self.y, self.t = x, y, t

    def __getitem__(self, index):
        x, y, t = self.x[index], self.y[index], self.t[index]
        image = Image.open(x).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, y, t

    def __len__(self):
        return len(self.x)


def load_model(model):
    printlog('\033[1;30;42mload state:\n{}\033[0m'.format(args.resume))
    loaded_state = torch.load(args.resume)
    if isinstance(model, torch.nn.DataParallel) \
            or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    current_state = model.state_dict()
    missing = []
    state = OrderedDict()
    for key, value in loaded_state.items():
        if 'module.' in key:
            key = key.replace('module.', '')
        if key in current_state:
            state[key] = value
        else:
            missing.append(key)
    if len(missing) > 0:
        printlog('\033[1;30;41mmissing keys:\n{}\033[0m'.format(missing))
    model.load_state_dict(state)
    return model


# **************** output ****************** #
def printlog(string: str, printed=True, delay=0):
    if printed is True and 'slient' not in args.opt:
        print(string)
        if delay > 0:
            time.sleep(delay)
    if args.mode == 'DDP':
        if args.local_rank != 0:
            return 
    if not os.path.exists(args.logdir+args.name):
        os.makedirs(args.logdir+args.name)
    txt = open(args.logdir+args.name+'/log.txt', 'a+')
    txt.write(string + '\n')
    txt.close()


def print_args():
    command = ''
    for s in sys.argv:
        command += (s + ' ')
    command = command[:-1]
    printlog(command)
    printlog('--------------args--------------')
    for k in list(vars(args).keys()):
        printlog('%s: %s' % (k, vars(args)[k]))
    printlog('--------------args--------------')


def backup():
    if 'slient' in args.opt:
        return
    consider = ['config', 'model', 'scheme', 'tool']
    files = [f for f in os.listdir() if '.py' in f or f in consider]
    files_dir = os.path.join(args.logdir+args.name, 'program')
    if os.path.exists(files_dir) is True:
        shutil.rmtree(files_dir)
    if os.path.exists(files_dir) is False:
        os.makedirs(files_dir)
        for name in files:
            os.system('cp -r {} {}'.format(name, files_dir))


def save_model(model, taskid):
    if 'slient' in args.opt:
        return
    if isinstance(model, torch.nn.parallel.DistributedDataParallel) \
            or isinstance(model, torch.nn.DataParallel):
        model = model.module
    state = model.state_dict()
    for key, value in state.items():
        state[key] = value.cpu()
    dir_ = os.path.join(args.logdir+args.name, 'pkl')
    if os.path.exists(dir_) is False:
        os.makedirs(dir_)
    torch.save(state, os.path.join(dir_, 'task{}.pkl'.format(taskid)))


def save_memory(memory, taskid):
    if 'slient' in args.opt:
        return
    memory_dict = dict()
    memory_cpu = Memory()
    for attr, attr_v in memory.__dict__.items():
        if isinstance(attr_v, torch.nn.Module):
            model = copy.deepcopy(attr_v)
            if isinstance(model, torch.nn.parallel.DataParallel) \
                    or isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model = model.module
            device = next(model.parameters()).device
            if device.type == 'cuda':
                model.cpu()
                memory_dict[attr] = model
        else:
            memory_dict[attr] = attr_v
    memory_cpu.__dict__.update(memory_dict)
    dir_ = os.path.join(args.logdir + args.name, 'pkl')
    if os.path.exists(dir_) is False:
        os.makedirs(dir_)
    mem_path = os.path.join(dir_, 'memory{}.pkl'.format(taskid))
    with open(mem_path, 'wb') as f:
        pickle.dump(memory_cpu, f)


def remote_notification(title, info, way='bark'):
    if not os.path.exists('config/user.yaml'):
        return
    file = open('config/user.yaml')
    user = CfgNode.load_cfg(file)
    if way == 'bark' and 'bark_id' in user:
        bark_id = user['bark_id']
        if bark_id is None:
            return
        url = f"https://api.day.app/{bark_id}/{title}/{info}"
        try:
            response = requests.post(url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return


def print_log_metrics(AccMatrix, taskid, remarks=''):
    if memory.task_size:
        task_size = np.array(list(memory.task_size.values()))
        Acc = ((AccMatrix[taskid-1, :taskid] * task_size).sum() / task_size.sum()).round(3)
    else:
        Acc = AccMatrix[taskid-1, :taskid].mean().round(3)
    forget, BWT = 0, 0
    if taskid > 1:
        # TODO, although this compute mode is widely adopt in most Task-IL
        #  research and consistent with the original definition,
        #  it may not be suitable for some settings in Class-IL, i.e.,
        #  unequal number of classes between tasks.
        subMatrix = AccMatrix[:taskid-1, :taskid-1]
        maxAcc = np.max(subMatrix, axis=0)
        curAcc = AccMatrix[taskid-1, :taskid-1]
        forget = np.mean(maxAcc - curAcc).round(3)
        BWT = np.mean(AccMatrix[taskid-1, :taskid-1] - np.diagonal(AccMatrix[:taskid-1, :taskid-1])).round(3)
    Accs = []
    for i in range(taskid):
        if memory.task_size:
            task_size = np.array(list(memory.task_size.values()))
            Accs.append(((AccMatrix[i, :i+1] * task_size[:i+1]).sum() / task_size[:i+1].sum()).round(3))
        else:
            Accs.append(AccMatrix[i, :i+1].mean().round(3))
    AIAcc = np.mean(Accs).round(3)

    info = '' if remarks == '' else remarks + '\n'
    for t in range(taskid):
        info += 'task{}:{} '.format(t+1, round(AccMatrix[taskid-1, t], 3))
    info = info[:-1] + '\n'
    info += 'Acc  :{} forget:{} BWT:{} AIAcc:{}'.format(Acc, forget, BWT, AIAcc)
    printlog(info)
    if 'slient' in args.opt:
        # block tensorboard
        return
    suffix = '' if remarks == '' else '_{}'.format(remarks)
    TBWriter.add_scalar('Acc'+suffix, Acc, taskid)
    TBWriter.add_scalar('Forget'+suffix, forget, taskid)
    TBWriter.add_scalar('BWT'+suffix, BWT, taskid)
    TBWriter.add_scalar('AIAcc'+suffix, AIAcc, taskid)
    for t in range(args.tasks):
        TBWriter.add_scalar('Task{}{}'.format(t+1, suffix), AccMatrix[taskid-1, t], taskid)
    np.savetxt(args.logdir + args.name + '/AccMatrix{}.csv'.format(suffix), AccMatrix, fmt='%.3f')

    if taskid == args.tasks:
        if 'bark' in args.opt:
            title = '{} {} B{}-{}S {}'.format(args.scheme, args.dataset, args.init, args.tasks, remarks)
            content = '{}\nAcc:{} Forget:{} BWT:{} AIA:{}'.format(args.note, Acc,
                                                                     forget, BWT, AIAcc)
            remote_notification(title, content, way='bark')


def summary(logdir: str):
    Accs, Forgets, AIAs = [], [], []
    exps = os.listdir(logdir)
    for exp in exps:
        path = os.path.join(logdir, exp, 'AccMatrix.csv')
        if not os.path.exists(path) or 'search' in path:
            continue
        Matrix = np.loadtxt(path)
        Acc = Matrix[-1, :].mean().round(3)
        lastAcc = Matrix[-1, :-1]
        subMatrix = Matrix[:-1, :-1]
        forget = (np.max(subMatrix, axis=0) - lastAcc).mean().round(3)
        Accs_ = []
        for i in range(Matrix.shape[0]):
            Accs_.append(Matrix[i, :i+1].mean().round(3))
        Accs.append(Acc)
        Forgets.append(forget)
        AIAs.append(np.mean(Accs_).round(3))
    meanAcc = (np.mean(Accs) * 100).round(1)
    devAcc = (np.std(Accs) * 100).round(2)
    meanForget = np.mean(Forgets).round(2)
    devForget = np.std(Forgets).round(2)
    meanAIA = (np.mean(AIAs) * 100).round(1)
    devAIA = (np.std(AIAs) * 100).round(2)
    info = '************************************************************\n'
    info += logdir + '\n{} experiments are summaried as follows:\n'.format(len(Accs))
    info += 'Acc(%):{}(+-{}) Forget:{}(+-{}) AIA(%):{}(+-{})\n'.format(meanAcc, devAcc, meanForget, devForget, meanAIA,
                                                                       devAIA)
    info += '************************************************************'
    print(info)
    with open(os.path.join(logdir, 'summary.txt'), 'w') as f:
        f.write(info)


# **************** training ******************** #
class Memory:
    def __init__(self):
        self.x = None
        self.y = None
        self.t = None
        self.iteration = 0
        self.start_time = time.time()
        self.stop__time = None
        self.task_size = OrderedDict()
        self.AccMatrix = np.zeros((args.tasks, args.tasks))

    def load_memory(self):
        if args.resume is not None:
            index = args.resume.rfind('/')
            string = args.resume[index+1:].replace('task', 'memory')
            mem_path = args.resume[:index+1] + string
            start = eval(args.resume[args.resume.rfind('task') + 4: args.resume.rfind('.')])
            if os.path.exists(mem_path):
                printlog('\033[1;30;42mload memory:\n{}\033[0m'.format(mem_path))
                with open(mem_path, 'rb') as f:
                    loaded_memory = pickle.load(f)
                for attr, attr_v in loaded_memory.__dict__.items():
                    if isinstance(attr_v, torch.nn.Module):
                        model = attr_v
                        if isinstance(model, torch.nn.parallel.DataParallel) \
                                or isinstance(model, torch.nn.parallel.DistributedDataParallel):
                            model = model.module
                        device = next(model.parameters()).device
                        if device.type == 'cuda':
                            model.cpu()
                            loaded_memory.__dict__[attr] = model
                            torch.cuda.empty_cache()
                    if 'AccMatrix' in attr:
                        if args.tasks != attr_v.shape[0] and start != 1:
                            raise ValueError('It must be resumed from the first task!')
                        AccMatrix = np.zeros((args.tasks, args.tasks))
                        AccMatrix[:start, :start] = attr_v[:start, :start]
                        loaded_memory.__dict__[attr] = AccMatrix
                if 'start_time' in loaded_memory.__dict__:
                    del loaded_memory.__dict__['start_time']
                self.__dict__.update(loaded_memory.__dict__)


def build_model():
    r"""
    Priority list: model specified by config.yaml > default model
    Returns:
        model
    """
    num_classes = datainfo[args.dataset]['classnum']
    if hasattr(args, 'model'):
        if args.model:
            index = args.model.rfind('.')
            package = args.model[:index]
            object_ = args.model[index+1:]
            modules = importlib.import_module(package)
            model = eval('modules.{}'.format(object_))(args=args, num_classes=num_classes)
        else:
            raise ValueError('You have specified a model for this scheme: {}, however, '
                             'the model path is empty!'.format(args.scheme))
    elif 'MNIST' in args.dataset:
        model = MLP(args=args, num_classes=num_classes)
    elif 'CIFAR' in args.dataset:
        if args.scenario == 'class':
            model = resnet32(args=args, num_classes=num_classes)
        elif args.scenario == 'task':
            model = ResNet18_Task(args=args, num_classes=num_classes)
        else:
            raise NotImplementedError('Not implement the model in {} for {}'.format(args.scenario, args.dataset))
    elif 'ImageNet' in args.dataset:
        model = resnet18(args=args, num_classes=num_classes)
    else:
        raise ValueError
    if hasattr(model, 'init_parameters'):
        model.init_parameters()
    if args.mode == 'DDP':
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model


def init_state():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.FAST:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        printlog('\033[1;30;41mWarning! FAST IMPLEMENTATION!\n'
                 'This will make CUDNN indeterministic!\033[0m')
    else:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if args.threads:
        torch.set_num_threads(args.threads)


def get_minmax_class(task):
    if (args.scenario == 'task') or (args.scenario == 'domain'):
        # task: id is used to select the corresponding classifier,
        # thus, task_id is known to classifier.
        # domain: id is used to minus the shift of ground truth labels,
        # however, classifier don't know the task_id.
        # for permuted/rotated MNIST, minclass and maxclass is not correct,
        # but these values are not used in activate_head() function.
        minclass = 0
        for i in args.increments[:task-1]:
            minclass += i
        maxclass = minclass + args.increments[task-1] - 1
    elif args.scenario == 'class':
        minclass = 0
        maxclass = 0
        for i in args.increments[:task]:
            maxclass += i
        maxclass -= 1
    else:
        raise ValueError('please choose a scenario from [\'task\', \'domain\', \'class\']')
    return minclass, maxclass


def activate_head(minclass, maxclass, prediction, target):
    target_ = target.clone()
    prediction_ = prediction.clone()
    if args.scenario == 'task':
        prediction_ = prediction_[:, minclass: maxclass+1]
        target_ = target_ - minclass
        # if minclass > 0:
        #     prediction_[:, :minclass].data.fill_(-10e10)
        # if maxclass < prediction_.size(1):
        #     prediction_[:, maxclass+1:prediction_.size(1)].data.fill_(-10e10)
        return prediction_, target_
    elif args.scenario == 'class':
        prediction_ = prediction_[:, 0: maxclass+1]
        return prediction_, target_
    elif args.scenario == 'domain':
        if args.dataset in ('PMNIST', 'RMNIST'):
            return prediction_, target_
        else:
            target_ = target_ - minclass
            return prediction_, target_


# **************** eval ******************** #
def remap_label(y_, minclass):
    y = y_.clone()
    if args.scenario == 'task':
        y = y + minclass
        return y
    elif args.scenario == 'class':
        return y
    elif args.scenario == 'domain':
        if args.dataset in ('PMNIST', 'RMNIST'):
            # TODO, or other muti-domain dataset
            return y
        else:
            # TODO, constructed by regular dataset (e.g. CIFAR100)
            y = y + minclass
            return y


def dp_gather_and_reinit(model, variable_name: str, target_device: int):
    """
    gather variables across GPUs for DP, used in hooks
    Args:
        model:
        variable_name:
        target_device:

    Returns:

    """
    outputs = eval('model.{}'.format(variable_name))
    assert isinstance(outputs, defaultdict)
    ids = list(outputs.keys())
    ids.sort()
    outputs_sort = [outputs[id_] for id_ in ids]
    outputs_gather = gather(outputs_sort, target_device)
    eval('model.{}.clear()'.format(variable_name))
    return outputs_gather


def distributed_concat(tensor):
    """
    synchronization across GPUs for DDP
    """
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat


def set_device():
    # ******************** GPU setting ********************* #
    # default to use GPU-0 for training
    if isinstance(args.gpuid, list):
        if len(args.gpuid) == 1:
            args.mode = 'GPU'
            args.gpuid = args.gpuid[0]
        elif len(args.gpuid) > 1:
            if args.mode == 'DP':
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in args.gpuid])
            elif args.mode == 'DDP':
                if args.local_rank != 0:
                    args.opt.append('slient')
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in args.gpuid])


def time_memory_snapshot(prefix: str = '', device=None, reset=False):
    if 'time_memory' not in args.opt:
        return

    def convert_datetime(elapsed_time_):
        days = elapsed_time_ // 86400
        elapsed_time_ = elapsed_time_ - 86400 * days
        hours = elapsed_time_ // 3600
        elapsed_time_ = elapsed_time_ - 3600 * hours
        minutes = elapsed_time_ // 60
        seconds = elapsed_time_ - 60 * minutes
        return '{}d|{}h|{}m|{}s'.format(int(days), int(hours), int(minutes), round(seconds, 2))

    if device is None:
        device = args.gpuid
    MB = 1024.0 * 1024.0
    cur_time = time.time()
    gpu_max_memory_allocated = round(torch.cuda.max_memory_allocated(device=device) / MB, 2)
    if reset:
        torch.cuda.reset_max_memory_allocated(device=device)
    elapsed_time = cur_time - memory.start_time
    cur_datetime = datetime.now()
    info = '*' * 40 + ' TIME and GPU ' + '*' * 40 + '\n'
    info += prefix + 'cur_time:{} '.format(cur_datetime) + \
            'ela_time:{} '.format(convert_datetime(elapsed_time)) + \
            'GPU_MAX:{} MB'.format(gpu_max_memory_allocated)
    info += '\n' + '*' * 94
    printlog(info)


if 'summary' in args.opt:
    summary(args.logdir + args.name)
    sys.exit(0)
if 'slient' not in args.opt:
    current_time = datetime.now().strftime('/%b%d_%H-%M-%S')
    args.name = args.name + current_time
    if args.note:
        args.name = args.name + '-' + str(args.note)
    from torch.utils.tensorboard import SummaryWriter
    TBWriter = SummaryWriter(log_dir=args.logdir + args.name, flush_secs=20)
if args.debug:
    printlog('\033[1;30;42m************* Debug mode! *************\033[0m')
set_device()
# global values
memory = Memory()
