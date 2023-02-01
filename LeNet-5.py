import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, ToTensor

import numpy as np 
import matplotlib.pyplot as plt 

''' gpu '''
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Current device is', DEVICE)

''' data loader '''
transform = Compose([Resize((32, 32)), ToTensor()])

train_dataset = MNIST(root='.', train=True, transform=transform, download=True)
test_dataset = MNIST(root='.', train=False, transform=transform, download=True)

BATCH_SIZE = 128

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

''' model '''
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        
        self.C1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5) # 28*28
        self.S2 = nn.AvgPool2d(kernel_size=2) # 14*14
        self.C3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) # 10*10
        self.S4 = nn.AvgPool2d(kernel_size=2) # 5*5
        self.C5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5) # 1*1
        self.F6 = nn.Linear(in_features=120, out_features=84)
        self.OUTPUT = nn.Linear(in_features=84, out_features=10)
        
    def forward(self, x):
        x = self.C1(x)
        x = self.S2(x)
        x = self.C3(x)
        x = self.S4(x)
        x = self.C5(x)
        x = self.F6(x)
        x = self.OUTPUT(x)
        
        return x
    
model = LeNet5()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model = model.to(device=DEVICE)
criterion = criterion.to(device=DEVICE)

