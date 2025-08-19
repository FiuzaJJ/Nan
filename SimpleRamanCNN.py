import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

class SimpleRamanCNN(nn.Module):
    def __init__(self, input_length=1340, num_outputs=3):
        super(SimpleRamanCNN, self).__init__()
        
        # Article CNN
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=9, padding=4)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=7, padding=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_outputs)
        
    def forward(self, x):
        # Input shape: (batch, 1, input_length)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        
        x = self.global_avg_pool(x)  # Shape: (batch, channels, 1)
        x = torch.flatten(x, 1)      # Shape: (batch, channels)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x