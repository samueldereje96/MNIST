import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, padding = 1)
        self.layer2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1)
        self.layer3 = nn.Linear(1568, 64)
        self.layer4 = nn.Linear(64, 16)
        self.layer5 = nn.Linear(16, 10)
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        # layer -> Relu -> pooling
        x = self.layer1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size = 2) # This converts our 28*28 to 14 * 14
        x = self.layer2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size = 2)
        x = self.flatten(x)
        x = self.layer3(x)
        x = F.relu(x)
        x = self.layer4(x)
        x = F.relu(x)
        x = self.layer5(x)

        return x

def getCNNModel():
    return CNNModel()