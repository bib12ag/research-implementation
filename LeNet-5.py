import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.utils.data import DataLoader

from torchvision import datasets, transforms 
from torchvision.transforms import Compose, Resize, ToTensor

import numpy as np 
import matplotlib.pyplot as plt 

''' gpu '''
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Current device is', DEVICE)

''' data loader '''
transform = Compose([Resize((32, 32)), ToTensor()])

train_dataset = datasets.MNIST(root='.', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='.', train=False, transform=transform, download=True)

BATCH_SIZE = 128

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)