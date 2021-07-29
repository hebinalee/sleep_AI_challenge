######################################
## FUNCTIONS UTILIZED IN MAIN SCRIPT 
######################################

import os
import random
import numpy as np
import torch
import cv2


def pred_to_label(pred_label):

    label = []
    for i in range(len(pred_label)):
        pred = pred_label[i]

        if pred == 0 : label.append('Wake')
        if pred == 1 : label.append('N1')
        if pred == 2 : label.append('N2')
        if pred == 3 : label.append('N3')
        if pred == 4 : label.append('REM')
    return np.array(label)

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    #print(random.random())
    if torch.cuda.is_available():
        print(f'seed : {seed_value}')
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def crop_image(image, mode='origin', k=2):

    origin_image = cv2.resize(image, dsize=(480,1080))
    if mode == 'drop':
        origin_image = drop_features(origin_image, k=k)

    group1 = origin_image[0:387,:]
    group2 = origin_image[387:731,:]
    group3 = origin_image[731:1075,:]

    #print(group1.shape, group2.shape, group3.shape)

    group1 = cv2.resize(group1, dsize=(480,387))
    group2 = cv2.resize(group2, dsize=(480,387))
    group3 = cv2.resize(group3, dsize=(480,387))

    img = np.stack([group1, group2, group3], axis=-1)

    return img

def drop_features(img, k = 1, heights = 43):
    n_features = 21
    start = [0,1,2,3,4,5,6,7,8,9,10,11,14,15,16,17,18,19,20,21,23,25] # n(feature + end) = 22
    drop_idx_list = np.random.choice(range(n_features), k, replace=False)
#     print(drop_idx_list)
    for drop_idx in drop_idx_list:
        img[start[drop_idx]*heights:start[drop_idx+1]*heights, :] = 0

    return img
