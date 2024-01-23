import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
import glob
import csv
from torchvision import transforms
import numpy as np
from random import randint
from random import choice
import random
from PIL import Image
from os.path import join as pjoin
from tqdm import tqdm
import imageio
import os
import json
import cv2

from utils.coco_process_utils import clean_annot, get_ignore_mask, get_heatmap, get_paf, get_keypoints, FLIP_INDICES
from utils.process_utils import flip, resize, color_augment, resize_hm_paf, normalize, affine_augment


def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride) # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded

def rotation_matrix(degree):
    theta = np.radians(degree)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R

def getTransform(degree, offset, scale):
    H = np.zeros((3,3))
    H[:2, :2] = rotation_matrix(degree) * scale
    H[0,2] = offset[0]
    H[1,2] = offset[1]
    H[2,2] = 1
    O = np.zeros((3,3))
    O[0,0] = O[1,1] = O[2,2] = 1
    O[0,2] = -240
    O[1,2] = -320
    H = np.linalg.inv(O) @ H @ O
    return H

class ClothDataset(Dataset):
    # def __init__(self, mode='train', path='upperL_partial_json_afterflip'):
    def __init__(self, mode='train', path='Cloth2SkeletonDataset_v2', body_part='top', train_ratio=1):
        super().__init__()
        self.samples = []
        self.scale = 0.5
        self.stride = 8
        self.padValue = 128
        self.mode = mode
        if body_part == 'top':
            self.useful = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12]
            self.limbSeq = [[1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9], [8, 12]]
        elif body_part == 'bottom':
            self.useful = [8, 9, 10, 11, 12, 13, 14]
            self.limbSeq = [[8, 9], [9, 10], [10, 11], [8, 12], [12, 13], [13, 14]]

        # Build transform
        transforms_list = []
        if self.mode == 'train':
            transforms_list.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize((0.5), (0.5)))
        self.transform = transforms.Compose(transforms_list)

        transforms_list = []
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize((0.5), (0.5)))
        self.heatmap_transform = transforms.Compose(transforms_list)
        # Grab training data 
        if mode != 'test':
            cat_list = [os.path.basename(cat) for cat in glob.glob(os.path.join(path, 'data_{}'.format(body_part), '*'))]
            
            total_json_files = []
            total_img_files = []
            
            for cat in cat_list:
                temp_json_files = glob.glob(os.path.join(path, 'data_{}'.format(body_part), cat, 'pose', '*.json'))
                # temp_json_files.sort(key=lambda x: os.path.basename(x).split('_')[1])
                temp_json_files.sort()
                temp_img_files = glob.glob(os.path.join(path, 'data_{}'.format(body_part), cat, 'image', '*.jpg'))
                # temp_img_files.sort(key=lambda x: os.path.basename(x).split('_')[1])
                temp_img_files.sort()
                
                if mode == 'train':
                    total_json_files += temp_json_files[:int(len(temp_json_files) * train_ratio)]
                    total_img_files += temp_img_files[:int(len(temp_img_files) * train_ratio)]
                elif mode == 'val':
                    total_json_files += temp_json_files[-int(len(temp_json_files) * train_ratio):]
                    total_img_files += temp_img_files[-int(len(temp_img_files) * train_ratio):]

            for img_dir, json_dir in zip(total_img_files, total_json_files):
                self.samples.append((img_dir, json_dir))
        # Grab testing images
        else:
            img_dirs = glob.glob(os.path.join(path, '*'))
            for img_dir in img_dirs:
                self.samples.append((img_dir, None))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_dir, json_dir = self.samples[idx]
        img_origin = cv2.imread(img_dir)

        img = cv2.resize(img_origin, (480, 640), interpolation=cv2.INTER_AREA)

        
        if self.mode == 'train':
            # Create Transform matrix (get param is staticmethod)
            degree, _, scale, _ = transforms.RandomAffine.get_params(degrees=(-30, 30), translate=(0, 0.2), scale_ranges=(0.5, 1), shears=None, img_size=(480,640))

            # Get offset
            max_dy = float(0.3 * 640)
            ty = int(round(torch.empty(1).uniform_(0, max_dy).item()))
            offset = (0, ty)
            
            H = getTransform(degree, offset, scale)
            img = cv2.warpPerspective(img, H, (480,640), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        
        img = cv2.resize(img, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
        img_resized = padRightDownCorner(img, self.stride, self.padValue)
        img = Image.fromarray(img_resized)
        img = self.transform(img)


        if self.mode == 'test':
            return img, os.path.basename(img_dir).split('.')[0], img_origin

        with open(json_dir) as f:
            label = json.load(f)
        label = label['people'][0]['pose_keypoints_2d']
        label = np.array([label[3*k:3*k+2] for k in self.useful])

        if self.mode == 'train':
            # Label transform
            label = np.concatenate((label, np.ones((label.shape[0], 1))), axis=-1)
            label = (H @ label.T).T
            label = label[:,:2] / label[:,2:]
            label = label / 2

        keypoints = np.zeros((25, 3))
        for idx, use in enumerate(self.useful):
            keypoints[use,:2] = label[idx]
            keypoints[use,2] = 1

        heat_map = get_heatmap(img_resized, keypoints, 7).astype(np.float32)
        paf = get_paf(img_resized, keypoints, 5, True, self.limbSeq).astype(np.float32)
        
        # heatmap_size (40, 30)
        heat_map, paf = resize_hm_paf(heat_map, paf, (30, 40))
        heat_map = self.heatmap_transform(heat_map.transpose(1,2,0))

        if self.mode == 'val':
            return img, img_origin
        else:
            return img, heat_map, paf

