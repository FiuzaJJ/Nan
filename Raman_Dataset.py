import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import importlib
import Data_augment
importlib.reload(Data_augment)

from torch.utils.data import Dataset , DataLoader
from Data_augment import sinusoidal_noise ,left_right_shift 



class RamanDataset(Dataset):
    def __init__(self, spectra, targets, train=True):

        if isinstance(spectra, np.ndarray) and spectra.dtype == object:
            spectra = np.stack(spectra)  # fix object array
        self.spectra = torch.tensor(spectra, dtype=torch.float32)
        
        
        self.targets = None if targets is None else torch.tensor(targets, dtype=torch.float32)
        self.train = train

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        # Add channel dimension for CNN: (C, L) where C=1 for 1D signal
        spectrum = self.spectra[idx].unsqueeze(0)  
        
        if self.targets is not None:
            target = self.targets[idx]


        # #data augmentation steps
         if self.train:
        #     #cosign augmentation
            spectrum = spectrum + sinusoidal_noise(1340)
        #     spectrum = left_right_shift(torch.linspace(300,1940,1340),spectrum,3)
        #     print("Data Augmentation")

        if self.targets is not None:
            target = self.targets[idx]
            return spectrum, target
        
        else:
            return spectrum