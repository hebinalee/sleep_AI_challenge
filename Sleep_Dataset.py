#######################################################
## READ DATA FROM FILENAMES AND SAVE IMAGE AND LABELS 
## Sleep_Dataset     : for train dataset
## Sleep_Test_Dataset: for test dataset
#######################################################

import os
import random
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset

from utils import crop_image


class Sleep_Dataset(Dataset):
    def __init__(self, csv_file, transform,mode='origin', data_root_dir='/DATA/'):

        self.mode = mode
        self.data_root_dir = data_root_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file, header=None)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data_root_dir + self.data[0][idx] +'/'+ self.data[1][idx]
        target = self.data[2][idx]
        #print(file_path, target)
        target = self._target_label(target)

        if not os.path.exists(file_path):
            print('dose not exist '+file_path)

        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        image = crop_image(image, mode = self.mode)

        sample = {'image':image, 'label':target}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

    def _target_label(self,target):
        if target == 'Wake' : return 0
        if target == 'N1' : return 1
        if target == 'N2' : return 2
        if target == 'N3' : return 3
        if target == 'REM' : return 4


class Sleep_Test_Dataset(Dataset):
    def __init__(self, csv_file, transform, mode='origin', data_root_dir='/DATA/'):
        self.mode= mode
        self.data_root_dir = data_root_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file, header=None)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data_root_dir + self.data[0][idx] +'/'+ self.data[1][idx]
        #target = self.data[2][idx]
        #print(file_path, target)
        #target = self._target_label(target)

        if not os.path.exists(file_path):
            print('dose not exist '+file_path)

        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        image = crop_image(image, mode = self.mode)

        sample = {'image':image, 'code':self.data[0][idx], 'num':self.data[1][idx]}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample
