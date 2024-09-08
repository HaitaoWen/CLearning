import os
import torch
import numpy as np
from tool import printlog
from sklearn.utils import shuffle
from torchvision import datasets, transforms


def CIFAR100(args):
    seed = args.seed
    pc_valid = args.pc_valid

    data = list()
    size = [3, 32, 32]
    cf100_dir = args.datadir
    file_dir = os.path.join(args.datadir, 'binary_cifar100')
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)

        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]

        # CIFAR100
        dat = {}
        dat['train'] = datasets.CIFAR100(cf100_dir, train=True, download=True,
                                         transform=transforms.Compose([transforms.ToTensor(),
                                                                       transforms.Normalize(mean, std)]))
        dat['test'] = datasets.CIFAR100(cf100_dir, train=False, download=True,
                                        transform=transforms.Compose([transforms.ToTensor(),
                                                                      transforms.Normalize(mean, std)]))
        print('sucess')
        for n in range(10):
            data.append({})
            data[n]['name'] = 'cifar100'
            data[n]['ncla'] = 10
            data[n]['train'] = {'x': [], 'y': []}
            data[n]['test'] = {'x': [], 'y': []}
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            for image, target in loader:
                n = target.numpy()[0]
                nn = (n // 10)
                data[nn][s]['x'].append(image)  # 255
                data[nn][s]['y'].append(n % 10)

        # "Unify" and save
        for t in range(10):
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'], os.path.join(os.path.expanduser(file_dir), 'data'+str(t)+s+'x.bin'))
                torch.save(data[t][s]['y'], os.path.join(os.path.expanduser(file_dir), 'data'+str(t)+s+'y.bin'))

    # Load binary files
    data = list()
    # ids=list(shuffle(np.arange(5),random_state=seed))
    ids = list(np.arange(10))
    printlog('Task order =', ids)
    for i in range(10):
        data.append(dict.fromkeys(['name', 'ncla', 'train', 'test']))
        for s in ['train', 'test']:
            data[i][s] = {'x': [], 'y': []}
            data[i][s]['x'] = torch.load(os.path.join(os.path.expanduser(file_dir), 'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y'] = torch.load(os.path.join(os.path.expanduser(file_dir), 'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        if data[i]['ncla'] == 2:
            data[i]['name'] = 'cifar10-'+str(ids[i])
        else:
            data[i]['name'] = 'cifar100-'+str(ids[i])

    # Validation
    for t in range(10):
        r = np.arange(data[t]['train']['x'].size(0))
        r = np.array(shuffle(r, random_state=seed), dtype=int)
        nvalid = int(pc_valid * len(r))
        ivalid = torch.LongTensor(r[:nvalid])
        itrain = torch.LongTensor(r[nvalid:])
        data[t]['valid'] = {}
        data[t]['valid']['x'] = data[t]['train']['x'][ivalid].clone()
        data[t]['valid']['y'] = data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x'] = data[t]['train']['x'][itrain].clone()
        data[t]['train']['y'] = data[t]['train']['y'][itrain].clone()
        data[t]['t'] = t + 1

    # build scenario
    data_eval = []
    increments = []
    for t in range(len(data)):
        data_eval.append({})
        increments.append(data[t]['ncla'])
        data_eval[t]['name'] = data[t]['name']
        data_eval[t]['ncla'] = data[t]['ncla']
        data_eval[t]['test'] = data[t]['test']
        data_eval[t]['t'] = data[t]['t']
        del data[t]['test']

    return data, data_eval, increments
