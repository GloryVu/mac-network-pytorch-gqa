import sys
import pickle
from collections import Counter

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import CLEVR, collate_data, transform

batch_size = 64
n_epoch = 180

train_set = DataLoader(
    CLEVR(sys.argv[1], 'val', transform=None),
    batch_size=batch_size,
    num_workers=4,
    collate_fn=collate_data,
)
net = torch.load(sys.argv[2])
net.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for epoch in range(n_epoch):
    dataset = iter(train_set)
    pbar = tqdm(dataset)
    correct_counts = 0
    total_counts = 0
    acc_by_cluster=[{'total_counts':0,
                     'correct_counts':0} for _ in range(6)]
    for image, question, q_len, answer, cluster in pbar:
        image, question = image.to(device), question.to(device)
        output = net(image, question, q_len)
        correct = output.detach().argmax(1) == answer.to(device)
        for c in correct:
            if c:
                acc_by_cluster[cluster]['correct_counts']+=1
                correct_counts += 1
            acc_by_cluster[cluster]['total_counts']
            total_counts += 1

    print('Avg Acc: {:.5f}'.format(correct_counts / total_counts))
    print('Avg Acc by cluster:')
    for cluster in acc_by_cluster:
        print('{}: {:.5f}'.format(cluster,cluster['correct_counts'] / cluster['total_counts']))
