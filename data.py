from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


class ShipDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        col_names = ["img", "train", "label","img_path"]
        self.label = pd.read_csv(csv_file,names=col_names)
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.label.iloc[idx, -1]
        image = io.imread(img_name)
        label = self.label.iloc[idx][2]
        label = np.array([label])
        label = label.astype('float')
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = {'image': self.transform(image), 'label': label}

        return sample