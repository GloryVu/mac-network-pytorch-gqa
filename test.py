import sys
import pickle
from collections import Counter

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import CLEVR, collate_data, transform
from model_gqa import MACNetwork
batch_size = 64
n_epoch = 180

train_set = DataLoader(
    CLEVR(sys.argv[1], 'val', transform=None),
    batch_size=batch_size,
    num_workers=4,
    collate_fn=collate_data,
)
dataset_type = sys.argv[1]
with open(f'data/{dataset_type}_dic.pkl', 'rb') as f:
    dic = pickle.load(f)

n_words = len(dic['word_dic']) + 1
n_answers = len(dic['answer_dic'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = MACNetwork(n_words, 2048, classes=n_answers, max_step=4).to(device)
net.eval()
net = net.load_state_dict(torch.load(sys.argv[2]), strict=False)

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
            acc_by_cluster[cluster]['total_counts']+=1
            total_counts += 1

    print('Avg Acc: {:.5f}'.format(correct_counts / total_counts))
    print('Avg Acc by cluster:')
    for cluster in acc_by_cluster:
        print('{}: {:.5f}'.format(cluster,cluster['correct_counts'] / cluster['total_counts']))
