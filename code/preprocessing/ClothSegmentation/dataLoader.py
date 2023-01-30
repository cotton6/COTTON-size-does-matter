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
import imageio
import matplotlib.pyplot as plt
import os
import cv2


def transform(img, crop=None):
    """ Crops and resizes images according to bounding box. """
    img = Image.open(img)
    if crop is not None:
        rect = transforms.RandomCrop(crop).get_params(img, (crop, crop))
        img = transforms.functional.crop(img, rect[0], rect[1], rect[2], rect[3])
    else:
        rect = None
    torchvision_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    img = torchvision_transform(img)
    return img, rect

def label_preprocessing(label, rect):
    
    # mask = imageio.imread(label)
    temp = Image.open(label)
    if rect is not None:
        label_mask = np.zeros((rect[2], rect[3]))
        temp = transforms.functional.crop(temp, rect[0], rect[1], rect[2], rect[3])
    else:
        label_mask = np.zeros((640, 480))
    mask = np.array(temp)

    mask = (mask >= 60).astype(int)
    mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]

    label_mask[mask == 7] = 1  # (White: 111) Barren land 
    label_mask[mask == 0] = 0  # (Black: 000) Unknown 
    
    return label_mask

def label2img(prediction):
    
    prediction = prediction.astype('uint8')
    # print(prediction == 7)
    # print(np.sum((prediction == 7).astype(int)))
    colormap = [[0,255,255], [255,255,0], [255,0,255], [0,255,0], [0,0,255], [255,255,255], [0,0,0], [255, 0, 0]]
    cm = np.array(colormap).astype('uint8')
    prediction = cm[prediction]

    return prediction


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

class Neckline_dataset(Dataset):

    def __init__(self, path='neckline_dataset', mode='train', crop=None):
        self.all_samples = []
        self.crop = crop
        self.mode = mode
        self.data_root = path

        cats = glob.glob('{}/*'.format(self.data_root))
        mask_dir = os.path.join(path, 'all_mask')
        if self.mode != 'test':
            for cat_dir in cats:
                images = glob.glob(os.path.join(cat_dir, '*.jpg'))
                images.sort(key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[1]))
                if self.mode == 'train':
                    # images = images[:30]
                    images = images[:]
                else:
                    images = images[30:]
                masks = [os.path.join(mask_dir, os.path.basename(img).replace('jpg', 'png')) for img in images]
                
                for img, mask in zip(images, masks):
                    self.all_samples.append((img, mask))
        else:
            images = glob.glob(os.path.join(path, '*.jpg'))
            images.sort(key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[1]))
            masks = [os.path.join(mask_dir, os.path.basename(img).replace('jpg', 'png')) for img in images]
            for img, mask in zip(images, masks):
                self.all_samples.append((img, mask))            
        
    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, index):
        img, mask = self.all_samples[index]
        imgName = os.path.basename(img).split('.')[0]
        img, rect = transform(img, crop=self.crop)
        img = img.type(torch.FloatTensor)
        if self.mode != 'train':
            return img, imgName
        mask = label_preprocessing(mask, rect)
        mask = torch.from_numpy(mask).type(torch.LongTensor)

        return img, mask

class Cloth_Segmentation_Dataset(Dataset):
    def __init__(self, path='cloth_segmentation_dataset', mode='train', crop=None):
        self.all_samples = []
        self.crop = crop
        self.mode = mode
        self.data_root = path
        self.rgb_transform_train = transforms.Compose([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ])
        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ])
        if self.mode != 'test':
            data_dict = {}
            data_list = os.path.join(path, 'data_list.txt')
            # Read data_list
            with open(data_list, 'r') as f:
                lines = f.readlines()
            # Categorize images with its class
            for line in lines:
                img_name, img_type = line.split()        
                if img_type in data_dict:
                    data_dict[img_type].append(img_name)
                else:
                    data_dict[img_type] = [img_name]

            self.all_samples = []
            if self.mode == 'train':
                for img_type in data_dict:
                    self.all_samples += data_dict[img_type][:]
            else:
                for img_type in data_dict:
                    self.all_samples += data_dict[img_type][30:]
        else:
            imgs = glob.glob(os.path.join(path, '*.jpg'))
            self.all_samples = [os.path.basename(img).split('.')[0] for img in imgs]
    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, index):
        sample_name = self.all_samples[index]
        if self.mode == 'test':
            img_origin = cv2.imread(os.path.join(self.data_root, sample_name+'.jpg'))
        else:
            img_origin = cv2.imread(os.path.join(self.data_root, 'images', sample_name+'.jpg'))

        img = cv2.resize(img_origin, (480, 640), interpolation=cv2.INTER_AREA)
        
        if self.mode != 'train':
            img = self.rgb_transform(img)
            return img, sample_name, img_origin
        else:
            degree, offset, scale, _ = transforms.RandomAffine.get_params(degrees=(-30, 30), translate=(0, 0.2), scale_ranges=(1, 1.5), shears=None, img_size=(480,640))
            H = getTransform(degree, offset, scale)
            img = cv2.warpPerspective(img, H, (480,640), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
            img = Image.fromarray(img)
            img = self.rgb_transform_train(img)
        
        mask = cv2.imread(os.path.join(self.data_root, 'masks', sample_name+'.png'), 0)
        mask = cv2.warpPerspective(mask, H, (480,640), cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
        mask = torch.from_numpy(mask).type(torch.LongTensor).squeeze(0)

        return img, mask
 