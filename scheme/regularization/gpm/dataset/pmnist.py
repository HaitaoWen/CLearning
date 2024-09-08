import os
import sys
import torch
import numpy as np
from sklearn.utils import shuffle
from torchvision import datasets, transforms


def PMNIST(args):
    seed = args.seed
    pc_valid = args.pc_valid
    mnist_dir = args.datadir
    pmnist_dir = os.path.join(args.datadir, 'binary_pmnist')
    fixed_order = args.fixed_order if hasattr(args, 'fixed_order') else False

    nperm = 10  # 10 tasks
    size = [1, 28, 28]
    seeds = np.array(list(range(nperm)), dtype=int)
    if not fixed_order:
        seeds = shuffle(seeds, random_state=seed)

    data = list()
    if not os.path.isdir(pmnist_dir):
        os.makedirs(pmnist_dir)
        # Pre-load MNIST
        mean = (0.1307,)
        std = (0.3081,)
        dat = dict()
        dat['train'] = datasets.MNIST(mnist_dir, train=True, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        dat['test'] = datasets.MNIST(mnist_dir, train=False, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        for i, r in enumerate(seeds):
            print(i, end=',')
            sys.stdout.flush()
            data.append({})
            data[i]['name'] = 'pmnist-{:d}'.format(i)
            data[i]['ncla'] = 10
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[i][s] = {'x': [], 'y': []}
                for image, target in loader:
                    aux = image.view(-1).numpy()
                    aux = shuffle(aux, random_state=r * 100 + i)
                    image = torch.FloatTensor(aux).view(size)
                    data[i][s]['x'].append(image)
                    data[i][s]['y'].append(target.numpy()[0])

            # "Unify" and save
            for s in ['train', 'test']:
                data[i][s]['x'] = torch.stack(data[i][s]['x']).view(-1, size[0], size[1], size[2])
                data[i][s]['y'] = torch.LongTensor(np.array(data[i][s]['y'], dtype=int)).view(-1)
                torch.save(data[i][s]['x'], os.path.join(os.path.expanduser(pmnist_dir), 'data' + str(r) + s + 'x.bin'))
                torch.save(data[i][s]['y'], os.path.join(os.path.expanduser(pmnist_dir), 'data' + str(r) + s + 'y.bin'))
        print()

    else:
        # Load binary files
        for i, r in enumerate(seeds):
            data.append(dict.fromkeys(['name', 'ncla', 'train', 'test']))
            data[i]['ncla'] = 10
            data[i]['name'] = 'pmnist-{:d}'.format(i)

            # Load
            for s in ['train', 'test']:
                data[i][s] = {'x': [], 'y': []}
                data[i][s]['x'] = torch.load(os.path.join(os.path.expanduser(pmnist_dir), 'data' + str(r) + s + 'x.bin'))
                data[i][s]['y'] = torch.load(os.path.join(os.path.expanduser(pmnist_dir), 'data' + str(r) + s + 'y.bin'))

    # Validation
    for t in range(len(data)):
        r = np.arange(data[t]['train']['x'].size(0))
        r = np.array(r, dtype=int)
        nvalid = int(pc_valid*len(r))
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
