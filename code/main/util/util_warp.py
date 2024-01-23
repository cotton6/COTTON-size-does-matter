# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 01:41:09 2019

@author: Lung
"""
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import os
from PIL import Image

import numpy as np
import argparse

label_colours = [(0,0,0)
            # 0=Background
            ,(128,0,0),(255,0,0),(0,85,0),(170,0,51),(255,85,0)
            # 1=Hat,  2=Hair,    3=Glove, 4=Sunglasses, 5=UpperClothes
            ,(0,0,85),(0,119,221),(85,85,0),(0,85,85),(85,51,0)
            # 6=Dress, 7=Coat, 8=Socks, 9=Pants, 10=Jumpsuits
            ,(52,86,128),(0,128,0),(0,0,255),(51,170,221),(0,255,255)
            # 11=Scarf, 12=Skirt, 13=Face, 14=LeftArm, 15=RightArm
            ,(85,255,170),(170,255,85),(255,255,0),(255,170,0)]
            # 16=LeftLeg, 17=RightLeg, 18=LeftShoe, 19=RightShoe

label_colours_HC = [(0,0,0),(51,170,221),(255,85,0)]
            # 0=Background, 1=Arm, 2=UpperClothes

def save_checkpoint(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.cpu().state_dict(), save_path)
    model.cuda()

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print("parameters not found")
        return
    model.load_state_dict(torch.load(checkpoint_path))
    print("load parameters success")
    model.cuda()
                
def imshow(img):
    '''
        input:  image (channels, h, w) or (h, w) in pytorch Tensor [-1,1]
    '''
    img = img.detach().cpu()/2+0.5 # unnormalize pytorch tensor
    npimg = img.numpy()
    if len(npimg.shape) == 4: #batch_size & RGB
        for i in range(npimg.shape[0]):
            plt.imshow(np.transpose(npimg[i], (1, 2, 0)))
            plt.show()
    elif len(npimg.shape) == 3: #RGB
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    elif len(npimg.shape) == 2: #L
        plt.imshow(npimg)
        plt.show()

def imsave(img, img_name):
    '''
        Used to save parsing image, 
            but need to be "visualize_parse" before this function
        input:  image (batch_size, channels, h, w) in pytorch Tensor [0,255]
    '''
    img = torchvision.utils.make_grid(img.detach().cpu())
    img = torchvision.transforms.ToPILImage()(img)
    #print("img.shape= ",img.shape)
    #img = img [0,:,:]
    #print("img.shape= ",img.shape)
    #img = Image.fromarray(img,'L')
    img.save(img_name)
    
def decode_labels(mask, labels, num_classes=20):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    n, h, w = mask.size()
    #print("mask.size()= ", mask.size())
    #assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((n, h, w, 3), dtype=np.uint8)
    #p_label_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    for i in range(n):
        img = Image.new('RGB', (w, h))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :]): #j_=h, j=tensor(1*w)
            for k_, k in enumerate(j): #k_=w, k=tensor (1*1)
                if k < num_classes:
                    pixels[k_,j_] = labels[k] #for whole parsing: label_colours[k]
        outputs[i] = np.array(img)
    outputs = np.transpose(outputs, (0,3,1,2))
    outputs = torch.from_numpy(outputs)
    return outputs

def visualize_parse(parse, mode):
    '''
        input:  parsing mask (batch_size, channel, h, w) in pytorch Tensor [-1,1]
        output: parsing image (batch_size, h, w) in pytorch Tensor [0,255] with mode 'RGB'
    '''
    parse = torch.argmax(parse.cpu(), dim=1)
    if mode == "whole":
        parse_img = decode_labels(parse, label_colours)
    elif mode == "input1":
        parse_img = decode_labels(parse, label_colours)
    elif mode == "input2":
        parse_img = decode_labels(parse, label_colours_HC)

    return parse_img

def save_gray_parse(parse, img_name):
    '''
        input:  parsing mask (batch_size, channel, h, w) in pytorch Tensor [-1,1]
        output: parsing image (batch_size, h, w) in pytorch Tensor [0,19] with mode 'L'
    '''
    parse = torch.argmax(parse.cpu(), dim=1)
    #print("parse.shape= ", parse.shape)
    parse_img = parse.numpy()
    #print("parse_img.shape= ",parse_img.shape)
    img = parse_img[0,:,:]
    #print("img.shape= ",img.shape)
    #print("type(img)= ", type(img))
    #print("img.dtype= ", img.dtype)
    #print("value of img= ", img[192][50:80])
    img = Image.fromarray(img.astype('uint8'),'L')
    img.save(img_name)

def remap(tensor, condition=1):
    if condition == 1:
        return tensor / 2 + 0.5
    elif condition == 255:
        return ((tensor / 2 + 0.5) * 255).type(torch.ByteTensor)

# add for GFLA
class StoreDictKeyPair(argparse.Action):
     def __call__(self, parser, namespace, values, option_string=None):
         my_dict = {}
         for kv in values.split(","):
             # print(kv)
             k,v = kv.split("=")
             my_dict[k] = int(v)
         setattr(namespace, self.dest, my_dict)   

class StoreList(argparse.Action):
     def __call__(self, parser, namespace, values, option_string=None):
        my_list = [int(item) for item in values.split(',')]
        setattr(namespace, self.dest, my_list)   


# def imsave_set(human, tar_pose, inC, warp_grid, warp_clothes, clothes_gt, img_name):
def imsave_set(human, tar_pose, parsing_c, C_align, warp_grid, warp_clothes, clothes_gt, img_name):
    human = torchvision.utils.make_grid(human.detach().cpu())
    human = torchvision.transforms.ToPILImage()(human)
    tar_pose = torchvision.utils.make_grid(tar_pose.detach().cpu())
    tar_pose = torchvision.transforms.ToPILImage()(tar_pose)
    parsing_c = torchvision.utils.make_grid(parsing_c.detach().cpu())
    parsing_c = torchvision.transforms.ToPILImage()(parsing_c)
    C_align = torchvision.utils.make_grid(C_align.detach().cpu())
    C_align = torchvision.transforms.ToPILImage()(C_align)
    warp_grid = torchvision.utils.make_grid(warp_grid.detach().cpu())
    warp_grid = torchvision.transforms.ToPILImage()(warp_grid)
    warp_clothes = torchvision.utils.make_grid(warp_clothes.detach().cpu())
    warp_clothes = torchvision.transforms.ToPILImage()(warp_clothes)
    clothes_gt = torchvision.utils.make_grid(clothes_gt.detach().cpu())
    clothes_gt = torchvision.transforms.ToPILImage()(clothes_gt)

    new_im = Image.new('RGB', (human.size[0]*7,human.size[1]))
    new_im.paste(human, (human.size[0]*0,0))
    new_im.paste(tar_pose, (human.size[0]*1,0))
    new_im.paste(parsing_c, (human.size[0]*2,0))
    new_im.paste(C_align, (human.size[0]*3,0))
    new_im.paste(warp_grid, (human.size[0]*4,0))
    new_im.paste(warp_clothes, (human.size[0]*5,0))
    new_im.paste(clothes_gt, (human.size[0]*6,0))

    new_im.save(img_name)

def imsave_setIO(human, tar_pose, parsing_c, C_align, warp_clothes, img_name):
    human = torchvision.utils.make_grid(human.detach().cpu())
    human = torchvision.transforms.ToPILImage()(human)
    tar_pose = torchvision.utils.make_grid(tar_pose.detach().cpu())
    tar_pose = torchvision.transforms.ToPILImage()(tar_pose)
    parsing_c = torchvision.utils.make_grid(parsing_c.detach().cpu())
    parsing_c = torchvision.transforms.ToPILImage()(parsing_c)
    C_align = torchvision.utils.make_grid(C_align.detach().cpu())
    C_align = torchvision.transforms.ToPILImage()(C_align)
    warp_clothes = torchvision.utils.make_grid(warp_clothes.detach().cpu())
    warp_clothes = torchvision.transforms.ToPILImage()(warp_clothes)

    new_im = Image.new('RGB', (human.size[0]*5,human.size[1]))
    new_im.paste(human, (human.size[0]*0,0))
    new_im.paste(tar_pose, (human.size[0]*1,0))
    new_im.paste(parsing_c, (human.size[0]*2,0))
    new_im.paste(C_align, (human.size[0]*3,0))
    new_im.paste(warp_clothes, (human.size[0]*4,0))

    new_im.save(img_name)