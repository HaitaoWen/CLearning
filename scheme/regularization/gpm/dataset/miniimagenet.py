# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
from PIL import Image
import os
import os.path
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
import numpy as np

import torch
from torchvision import transforms


class MiniImageNet(torch.utils.data.Dataset):

    def __init__(self, root, train):
        super(MiniImageNet, self).__init__()
        if train:
            self.name='train'
        else:
            self.name='test'
        root = os.path.join(root)
        with open(os.path.join(root,'{}.pkl'.format(self.name)), 'rb') as f:
            data_dict = pickle.load(f)

        self.data = data_dict['images']
        self.labels = data_dict['labels']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img, label = self.data[i], self.labels[i]
        return img, label


class iMiniImageNet(MiniImageNet):

    def __init__(self, root, classes, memory_classes, memory, task_num, train, transform=None):
        super(iMiniImageNet, self).__init__(root=root, train=train)

        self.transform = transform
        if not isinstance(classes, list):
            classes = [classes]

        self.class_mapping = {c: i for i, c in enumerate(classes)}
        self.class_indices = {}

        for cls in classes:
            self.class_indices[self.class_mapping[cls]] = []

        data = []
        labels = []
        tt = []  # task module labels
        td = []  # disctiminator labels

        for i in range(len(self.data)):
            if self.labels[i] in classes:
                data.append(self.data[i])
                labels.append(self.class_mapping[self.labels[i]])
                tt.append(task_num)
                td.append(task_num+1)
                self.class_indices[self.class_mapping[self.labels[i]]].append(i)

        if memory_classes:
            for task_id in range(task_num):
                for i in range(len(memory[task_id]['x'])):
                    if memory[task_id]['y'][i] in range(len(memory_classes[task_id])):
                        data.append(memory[task_id]['x'][i])
                        labels.append(memory[task_id]['y'][i])
                        tt.append(memory[task_id]['tt'][i])
                        td.append(memory[task_id]['td'][i])

        self.data = np.array(data)
        self.labels = labels
        self.tt = tt
        self.td = td

    def __getitem__(self, index):

        img, target, tt, td = self.data[index], self.labels[index], self.tt[index], self.td[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if not torch.is_tensor(img):
            img = Image.fromarray(img)
            img = self.transform(img)
        # return img, target, tt, td
        return img, target

    def __len__(self):
        return len(self.data)


def miniImageNet(args):
    """docstring for DatasetGen"""
    seed = args.seed
    pc_valid = args.pc_valid
    root = os.path.join(args.datadir, 'miniImageNet')

    num_tasks = 20
    num_classes = 100
    inputsize = [3, 84, 84]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transformation = transforms.Compose([
                                transforms.Resize((84,84)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean, std=std)])

    num_workers = 4
    pin_memory = True

    np.random.seed(seed)
    task_ids = np.split(np.random.permutation(num_classes), num_tasks)
    task_ids = [list(arr) for arr in task_ids]

    scenario, scenario_eval, increments = [], [], []
    for task_id in range(num_tasks):
        print('Load data {} into memory'.format(task_id + 1))
        scenario.append({})
        scenario_eval.append({})

        sys.stdout.flush()

        train_set = iMiniImageNet(root=root, classes=task_ids[task_id],
                                  memory_classes=None, memory=None,
                                  task_num=task_id, train=True, transform=transformation)

        test_set = iMiniImageNet(root=root, classes=task_ids[task_id], memory_classes=None,
                                 memory=None, task_num=task_id, train=False, transform=transformation)

        split = int(np.floor(pc_valid * len(train_set)))
        train_split, valid_split = torch.utils.data.random_split(train_set,
                                                                 [len(train_set) - split, split])

        train_loader = torch.utils.data.DataLoader(train_split, batch_size=2450, num_workers=num_workers,
                                                   pin_memory=pin_memory, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_split, batch_size=50,
                                                   num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=500, num_workers=num_workers,
                                                  pin_memory=pin_memory, shuffle=True)

        scenario[task_id]['train'] = {'x': [], 'y': []}
        scenario[task_id]['valid'] = {'x': [], 'y': []}
        for data, label in train_loader:
            scenario[task_id]['train']['x'] = data
            scenario[task_id]['train']['y'] = label
        for data, label in valid_loader:
            scenario[task_id]['valid']['x'] = data
            scenario[task_id]['valid']['y'] = label

        scenario_eval[task_id]['test'] = {'x': [], 'y': []}
        for data, label in test_loader:
            scenario_eval[task_id]['test']['x'] = data
            scenario_eval[task_id]['test']['y'] = label

        scenario[task_id]['t'] = task_id
        scenario_eval[task_id]['t'] = task_id
        scenario[task_id]['name'] = 'iMiniImageNet-{}-{}'.format(task_id, task_ids[task_id])
        scenario_eval[task_id]['name'] = 'iMiniImageNet-{}-{}'.format(task_id, task_ids[task_id])
        increments.append(len(scenario[task_id]['train']['y'].unique()))

    return scenario, scenario_eval, increments
