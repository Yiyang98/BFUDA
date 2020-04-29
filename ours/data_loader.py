import sys
import numpy as np
import torch
import random
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models
import argparse, time, os
import torch.utils.data as data
from PIL import Image
import os
from usps import *

seed=1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

def generate_compl_labels(labels):
    # args, labels: ordinary labels
    K = torch.max(labels)+1
    candidates = np.arange(K)
    candidates = np.repeat(candidates.reshape(1, K), len(labels), 0)
    mask = np.ones((len(labels), K), dtype=bool)
    mask[range(len(labels)), labels.numpy()] = False
    candidates_ = candidates[mask].reshape(len(labels), K-1)  # this is the candidates without true class
    idx = np.random.randint(0, K-1, len(labels))
    complementary_labels = candidates_[np.arange(len(labels)), np.array(idx)]
    return complementary_labels

def class_prior(complementary_labels):
    return np.bincount(complementary_labels) / len(complementary_labels)

def prepare_train_loaders(full_train_loader, batch_size, ordinary_train_dataset):
    for i, (data, labels) in enumerate(full_train_loader):
            K = torch.max(labels)+1 # K is number of classes, full_train_loader is full batch
    complementary_labels = generate_compl_labels(labels)
    ccp = class_prior(complementary_labels)
    complementary_dataset = torch.utils.data.TensorDataset(data, torch.from_numpy(complementary_labels).long())
    ordinary_train_loader = torch.utils.data.DataLoader(dataset=ordinary_train_dataset, batch_size=batch_size, shuffle=True,num_workers=8)
    complementary_train_loader = torch.utils.data.DataLoader(dataset=complementary_dataset, batch_size=batch_size, shuffle=True,num_workers=8)
    return complementary_dataset, ccp

class GetLoader(data.Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform

        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data

def data_loader(task):
    if task == 'M2U':
        #M1U1
        M_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        ordinary_train_dataset = dsets.MNIST(root='/data/menwu/data/MNIST', train=True, transform=M_transform, download=True)
        full_train_loader = torch.utils.data.DataLoader(dataset=ordinary_train_dataset, batch_size=len(ordinary_train_dataset), shuffle=True)
        complementary_dataset, ccp = prepare_train_loaders(full_train_loader, 128, ordinary_train_dataset)

        U_transform = transforms.Compose([transforms.Resize([28, 28]),transforms.ToTensor(),transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        test_dataset = USPS('/data/menwu/data/USPS2/Test',train=False, download=True, transform=U_transform)
        ordinary_train_dataset2 = USPS('/data/menwu/data/USPS2/train', train=True, download=True, transform=U_transform)
    elif task == 'U2M':
        #M1U1
        U_transform = transforms.Compose([transforms.Resize([28, 28]),transforms.ToTensor(),transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        ordinary_train_dataset = USPS('/data/menwu/data/USPS2/train', train=True, download=True, transform=U_transform)
        full_train_loader = torch.utils.data.DataLoader(dataset=ordinary_train_dataset, batch_size=len(ordinary_train_dataset), shuffle=True)
        complementary_dataset, ccp = prepare_train_loaders(full_train_loader, 128, ordinary_train_dataset)

        M_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        test_dataset = dsets.MNIST(root='/data/menwu/data/MNIST', train=False, transform=M_transform)
        ordinary_train_dataset2 = dsets.MNIST(root='/data/menwu/data/MNIST', train=True, transform=M_transform, download=True)
    else:
        print('Wrong task!')

    return complementary_dataset, ordinary_train_dataset, ordinary_train_dataset2, test_dataset, ccp