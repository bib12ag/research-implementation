import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.utils.data import DataLoader

from torchvision import datasets, transforms 
import numpy as np 
import matplotlib.pyplot as plt 

''' gpu '''
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Current device is', DEVICE)
