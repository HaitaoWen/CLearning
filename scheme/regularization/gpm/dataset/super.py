import os
import torch
import pickle
import numpy as np
from torchvision import datasets
from sklearn.utils import shuffle


def CIFAR100(args):
    flat = False
    seed = args.seed
    val_ratio = args.pc_valid  # 0.05
    order = args.order if args.order is not None else 0
    # mean = np.array([x / 255 for x in [125.3, 123.0, 113.9]])
    # std = np.array([x / 255 for x in [63.0, 62.1, 66.7]])

    task_orders = [np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
                  np.array([15, 12, 5, 9, 7, 16, 18, 17, 1, 0, 3, 8, 11, 14, 10, 6, 2, 4, 13, 19]),
                  np.array([17, 1, 19, 18, 12, 7, 6, 0, 11, 15, 10, 5, 13, 3, 9, 16, 4, 14, 2, 8]),
                  np.array([11, 9, 6, 5, 12, 4, 0, 10, 13, 7, 14, 3, 15, 16, 8, 1, 2, 19, 18, 17]),
                  np.array([6, 14, 0, 11, 12, 17, 13, 4, 9, 1, 7, 19, 8, 10, 3, 15, 18, 5, 2, 16])]

    task_order = task_orders[order]

    CIFAR100_LABELS_LIST = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm'
    ]

    sclass = []
    sclass.append(' beaver, dolphin, otter, seal, whale,')  # aquatic mammals
    sclass.append(' aquarium_fish, flatfish, ray, shark, trout,')  # fish
    sclass.append(' orchid, poppy, rose, sunflower, tulip,')  # flowers
    sclass.append(' bottle, bowl, can, cup, plate,')  # food
    sclass.append(' apple, mushroom, orange, pear, sweet_pepper,')  # fruit and vegetables
    sclass.append(' clock, computer keyboard, lamp, telephone, television,')  # household electrical devices
    sclass.append(' bed, chair, couch, table, wardrobe,')  # household furniture
    sclass.append(' bee, beetle, butterfly, caterpillar, cockroach,')  # insects
    sclass.append(' bear, leopard, lion, tiger, wolf,')  # large carnivores
    sclass.append(' bridge, castle, house, road, skyscraper,')  # large man-made outdoor things
    sclass.append(' cloud, forest, mountain, plain, sea,')  # large natural outdoor scenes
    sclass.append(' camel, cattle, chimpanzee, elephant, kangaroo,')  # large omnivores and herbivores
    sclass.append(' fox, porcupine, possum, raccoon, skunk,')  # medium-sized mammals
    sclass.append(' crab, lobster, snail, spider, worm,')  # non-insect invertebrates
    sclass.append(' baby, boy, girl, man, woman,')  # people
    sclass.append(' crocodile, dinosaur, lizard, snake, turtle,')  # reptiles
    sclass.append(' hamster, mouse, rabbit, shrew, squirrel,')  # small mammals
    sclass.append(' maple_tree, oak_tree, palm_tree, pine_tree, willow_tree,')  # trees
    sclass.append(' bicycle, bus, motorcycle, pickup_truck, train,')  # vehicles 1
    sclass.append(' lawn_mower, rocket, streetcar, tank, tractor,')  # vehicles 2

    # download CIFAR100
    datasets.CIFAR100('data', train=True, download=True)
    datasets.CIFAR100('data', train=False, download=True)

    alldata = []

    for split in ['train', 'test']:

        file = open(os.path.join('data', 'cifar-100-python/{}'.format(split)), 'rb')
        dict = pickle.load(file, encoding='bytes')

        # NOTE Image Standardization
        images = (dict[b'data'])
        images = np.float32(images) / 255
        labels = dict[b'fine_labels']
        labels_pair = [[jj for jj in range(100) if ' %s,' % CIFAR100_LABELS_LIST[jj] in sclass[kk]] for kk in range(20)]

        argsort_sup = [[] for _ in range(20)]
        for _i in range(len(images)):
            for _j in range(20):
                if labels[_i] in labels_pair[_j]:
                    argsort_sup[_j].append(_i)

        argsort_sup_c = np.concatenate(argsort_sup)

        position = [_k for _k in range(0, len(images) + 1, int(len(images) / 20))]

        data = []
        for t, idx in enumerate(task_order):
            data.append({})
            data[t]['name'] = 'cifar100'
            data[t]['ncla'] = 5
            data[t][split] = {'x': [], 'y': []}
            data[t]['t'] = t + 1
            gimages = np.take(images, argsort_sup_c[position[idx]:position[idx + 1]], axis=0)

            if not flat:
                gimages = gimages.reshape([gimages.shape[0], 32, 32, 3])
                # gimages = (gimages-mean)/std # mean,std normalization
                gimages = gimages.swapaxes(2, 3).swapaxes(1, 2)
                # gimages = tf.image.per_image_standardization(gimages)

            glabels = np.take(labels, argsort_sup_c[position[idx]:position[idx + 1]])
            for _si, swap in enumerate(labels_pair[t]):
                glabels = ['%d' % _si if x == swap else x for x in glabels]

            data[t][split]['x'] = torch.FloatTensor(gimages)

            data[t][split]['y'] = torch.LongTensor(np.array([np.int32(glabels)], dtype=int)).view(-1)

            if split == 'train':
                r = np.arange(data[t][split]['x'].size(0))
                r = np.array(shuffle(r, random_state=seed), dtype=int)
                nvalid = int(val_ratio * len(r))
                ivalid = torch.LongTensor(r[:nvalid])
                itrain = torch.LongTensor(r[nvalid:])
                data[t]['valid'] = {}
                data[t]['valid']['x'] = data[t]['train']['x'][ivalid].clone()
                data[t]['valid']['y'] = data[t]['train']['y'][ivalid].clone()
                data[t]['train']['x'] = data[t]['train']['x'][itrain].clone()
                data[t]['train']['y'] = data[t]['train']['y'][itrain].clone()
        alldata.append(data)
    increments = [5] * 20

    return alldata[0], alldata[1], increments
