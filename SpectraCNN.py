import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectraCNN(nn.Module):
    def __init__(self, input_length=1340, num_outputs=3):
        super(SpectraCNN, self).__init__()
        
        # Convolutions
        self.conv1 = nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7)  
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=11, stride=2, padding=5) 
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3) 
        self.bn3 = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2) 
        self.bn4 = nn.BatchNorm1d(256)

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected head
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_outputs)

        # Sigmoid activation for bounded regression [0,1]
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        # Input: (batch, 1, 1340)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Pooling
        x = self.global_avg_pool(x)   # (batch, 256, 1)
        x = x.squeeze(-1)             # (batch, 256)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigm(x)              # <-- ensures outputs in [0,1]
        return x