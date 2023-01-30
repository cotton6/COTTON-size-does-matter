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
import os

class Sleeve_Dataset(Dataset):

    def __init__(self, mode='train', path='hw1_data/p1_data/train_50/*.png'):
        self.allClip = []
        self.mode = mode
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((640, 480)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=(-30, 30), translate=(0.2, 0.2), scale=(0.25, 1), fill=255),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((640, 480)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        if mode != 'test':
            # sleeve is class 0 and sleeveless is class 1
            sleeve_images = glob.glob(os.path.join('sleeve_dataset', 'sleeve', '*.jpg'))
            sleeveless_images = glob.glob(os.path.join('sleeve_dataset', 'sleeveless', '*.jpg'))
            
            sleeve_images.sort()
            sleeveless_images.sort()
            
            if mode == 'train':
                sleeve_images = sleeve_images[:int(0.8*len(sleeve_images))]
                sleeveless_images = sleeveless_images[:int(0.8*len(sleeve_images))]
            else:
                sleeve_images = sleeve_images[int(0.8*len(sleeve_images)):]
                sleeveless_images = sleeveless_images[int(0.8*len(sleeve_images)):]                

            for img in sleeve_images:
                self.allClip.append((img, 0))
            for img in sleeveless_images:
                self.allClip.append((img, 1))
        else:
            images = glob.glob(os.path.join(path, '*.jpg'))
            for img in images:
                self.allClip.append((img, None))

    def __len__(self):
        return len(self.allClip)

    def __getitem__(self, index):
        fileName, label = self.allClip[index]
        img = self.transform(Image.open(fileName))
        fileName = os.path.basename(fileName).split('.')[0]
        img = img.type(torch.FloatTensor)
        
        if self.mode == 'test':
            return img, fileName

        label = torch.from_numpy(np.array(label)).type(torch.LongTensor)
        return img, label
