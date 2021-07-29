import os
import random
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import models, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet

from utils import createFolder, seed_everything, crop_image
from Sleep_Dataset import Sleep_Test_Dataset
from classifier_utils import submit


seed=20
seed_everything(seed)


batch_size = 32

csv_file_test = '/DATA/testset-for_user.csv'


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform = transforms.Compose(
    [transforms.ToTensor(),
        normalize])


test_set = Sleep_Test_Dataset(csv_file_test, transform)


testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)


classes = ('Wake', 'N1', 'N2', 'N3', 'REM')


file_name = 'trial5_epoch4_norm_drop.pth'

#model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=5)
model = models.resnet18(pretrained=True)
model.load_state_dict(torch.load('./model/'+file_name))
use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")
model.to(device)

submit(model, file_name.split('.')[0], testloader, device)


print("done")
