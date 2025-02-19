import json
import os
import pickle

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import h5py

from transforms import Scale

img = None
img_info = {}
def gqa_feature_loader():
    global img, img_info
    if img is not None:
        return img, img_info

    h = h5py.File('data/a.hdf5', 'r')
    img = h['features']
    img_info = json.load(open('data/gqa_objects_merged_info.json', 'r'))
    return img, img_info


class CLEVR(Dataset):
    def __init__(self, root, split='train', transform=None):
        with open(f'data/CLEVR_{split}.pkl', 'rb') as f:
            self.data = pickle.load(f)

        # self.transform = transform
        self.root = root
        self.split = split

        self.h = h5py.File('data/CLEVR_features.hdf5'.format(split), 'r')
        self.img = self.h['data']

    def close(self):
        self.h.close()

    def __getitem__(self, index):
        id, question, answer, cluster = self.data[index]
        img = torch.from_numpy(self.img[id])

        return img, question, len(question), answer,cluster

    def __len__(self):
        return len(self.data)

class GQA(Dataset):
    def __init__(self, root, split='train', transform=None):
        with open(f'data/gqa_{split}.pkl', 'rb') as f:
            self.data = pickle.load(f)

        self.root = root
        self.split = split
        self.img, self.img_info = gqa_feature_loader()

    def __getitem__(self, index):
        imgfile, question, answer = self.data[index]
        idx = int(self.img_info[imgfile]['index'])
        img = torch.from_numpy(self.img[idx])
        return img, question, len(question), answer

    def __len__(self):
        return len(self.data)

transform = transforms.Compose([
    Scale([224, 224]),
    transforms.Pad(4),
    transforms.RandomCrop([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
])

def collate_data(batch):
    images, lengths, answers, clusters = [], [], [], []
    batch_size = len(batch)

    max_len = max(map(lambda x: len(x[1]), batch))

    questions = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

    for i, b in enumerate(sort_by_len):
        image, question, length, answer, cluster = b
        images.append(image)
        length = len(question)
        questions[i, :length] = question
        lengths.append(length)
        answers.append(answer)
        clusters.append(cluster)
    return torch.stack(images), torch.from_numpy(questions), \
        lengths, torch.LongTensor(answers), clusters
