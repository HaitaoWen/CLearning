import os
import pickle

import numpy as np
from PIL import Image


root = 'data/miniImageNet'
splits = ['train', 'val', 'test']
root_save = 'data/miniImageNet'

data_dict = {'images': [], 'labels': []}
labels_unique = []

for split in splits:
    print('read split: {}'.format(split))
    root_split = os.path.join(root, split)
    labels = os.listdir(root_split)
    labels_unique += labels
    for label in labels:
        root_label = os.path.join(root_split, label)
        names_image = os.listdir(root_label)
        for name_image in names_image:
            path_image = os.path.join(root_label, name_image)
            image = Image.open(path_image)
            # plt.imshow(image)
            # plt.show()
            data_dict['images'].append(np.asarray(image))
            data_dict['labels'].append(label)

for index in range(len(data_dict['labels'])):
    label_id = labels_unique.index(data_dict['labels'][index])
    data_dict['labels'][index] = label_id

if not os.path.exists(root_save):
    os.makedirs(root_save)
with open(os.path.join(root_save + '/data.pkl'), 'wb') as f:
    pickle.dump(data_dict, f)
