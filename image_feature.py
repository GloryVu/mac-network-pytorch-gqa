import h5py
import torch
from torchvision.models.resnet import ResNet, resnet101
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transforms import Scale
import sys
import os
from PIL import Image
from tqdm import tqdm
from torch.autograd import Variable
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)

    return x

transform = transforms.Compose([
    Scale([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

class CLEVR(Dataset):
    def __init__(self, root, split='train'):
        self.root = root
        self.split = split
        self.length = len(os.listdir(os.path.join(root,
                                                'images', split)))

        with open(f'{root}/questions/mini_CLEVR_{split}_questions_translated.json') as f:
            data = json.load(f)
        self.img_idx_map = {}
        i=0
        for question in data['questions']:
            if question['image_index'] not in self.img_idx_map.keys():
                image_idx_map[question['image_index']] = i
                i+=1
        self.idx_img_map = {v:k for k,v in self.img_idx_map.items()}

    def __getitem__(self, index):
        img = os.path.join(self.root, 'images',
                        self.split,
                        'CLEVR_{}_{}.png'.format(self.split,
                                            str(idx_img_map[index]).zfill(6)))
        img = Image.open(img).convert('RGB')
        return transform(img)

    def __len__(self):
        return self.length

batch_size = 50

resnet = resnet101(True).to(device)
resnet.eval()
resnet.forward = forward.__get__(resnet, ResNet)

def create_dataset(split):
    dataloader = DataLoader(CLEVR(sys.argv[1], split), batch_size=batch_size,
                            num_workers=4)

    size = len(dataloader)

    print(split, 'total', size * batch_size)

    f = h5py.File('data/CLEVR_features.hdf5'.format(split), 'w', libver='latest')
    dset = f.create_dataset('data', (size * batch_size, 1024, 14, 14),
                            dtype='f4')

    with torch.no_grad():
        for i, image in tqdm(enumerate(dataloader)):
            image = image.to(device)
            features = resnet(image).detach().cpu().numpy()
            try:
                dset[i * batch_size:(i + 1) * batch_size] = features
            except:
                dset[i * batch_size:i * batch_size+features.shape[0]] = features

    f.close()

create_dataset('val')
create_dataset('train')