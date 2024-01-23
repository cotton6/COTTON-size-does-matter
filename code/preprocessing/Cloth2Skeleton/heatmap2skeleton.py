from re import S
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import glob
from random import shuffle
import time
import os
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader 
from dataLoader import ClothDataset
from model import bodypose_model
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import csv
import sys
import json
import torchvision
import cv2
import math
from post import decode_pose, append_result
from utils.process_utils import resize_hm, denormalize

# transfer caffe model to pytorch which will match the layer name
def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        try:
            transfered_model_weights[weights_name] = model_weights['.'.join(weights_name.split('.')[1:])]
        except:
            continue
    return transfered_model_weights

# full_weight_name = 'cloth2skeleton_L1_all'
full_weight_name = 'just_4_test'


def draw_bodypose(canvas, keypoints):
    stickwidth = 4
    limbSeq = [[1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9], \
               [8, 12]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for i in range(len(colors)):
        x, y = keypoints[i][0:2]
        cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
    for i in range(len(limbSeq)):
        cur_canvas = canvas.copy()
        Y = keypoints[np.array(limbSeq[i]), 0]
        X = keypoints[np.array(limbSeq[i]), 1]
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    # plt.imsave("preview.jpg", canvas[:, :, [2, 1, 0]])
    # plt.imshow(canvas[:, :, [2, 1, 0]])
    return canvas


def train(opt):
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('./runs/{}'.format(full_weight_name))

    criterion_L2 = nn.MSELoss().cuda()
    # criterion_L1 = nn.L1Loss().cuda()


    train_dataset = ClothDataset(mode="train")
    dataloader_train = DataLoader(dataset=train_dataset, batch_size=1, num_workers=0, drop_last=True, shuffle=True)

    model = bodypose_model()

    pretrained_dict = transfer(model, torch.load('body_pose_model.pth'))

    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k.find('Mconv7')==-1 and k.find('conv5_5')==-1 and k.find('Mconv1')==-1}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict, strict=False)

    model = model.cuda()

    # Optimizers
    # optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.Adam(model.parameters(),
                                   lr=opt.lr,
                                   betas=(opt.b1, opt.b2))
    if opt.scheduler:  
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.1, step_size=3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
    
    Tensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor

    # Load pretrained weights
    if os.path.isfile(opt.weights_path):
        print("=> loading checkpoint '{}'".format(opt.weights_path))
        checkpoint = torch.load(opt.weights_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # exit()
        if opt.scheduler:
            scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(opt.weights_path, checkpoint['epoch']))
    else:
        start_epoch = 0
        print("=> no checkpoint found at '{}'".format(opt.weights_path))

    # Start training
    for epoch_index in range(start_epoch, opt.n_epochs):
        print('epoch_index=', epoch_index)
        start = time.time()
        # in each minibatch
        # for g in optimizer.param_groups:
            # g['lr'] = 0.0001
        for param_group in optimizer.param_groups:
            print('lr: ' + str(param_group['lr']))

        # in each minibatch
        pbar = tqdm(dataloader_train, desc='training')

        for batchIdx, (imgs, heat_maps, pafs) in enumerate(pbar):
            # time_list = []
            iterNum = epoch_index * len(pbar) + batchIdx

            imgs = Variable(imgs.cuda())
            heat_maps = Variable(heat_maps.cuda())
            pafs = Variable(pafs.cuda())

            heatmap_avg = heat_maps[0].cpu().numpy() / 2 + 0.5
            paf_avg = pafs[0].cpu().numpy()
            #visualize_output_single(img_basic, heatmap_t, paf_t, ignore_mask_t, heatmap_avg, paf_avg)
            # param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
            param = {'thre1': 0.1, 'thre2': 0, 'thre3': 0}
            canvas, to_plot, candidate, subset = decode_pose(np.zeros((640,480,3)), param, heatmap_avg, paf_avg)
            print(candidate)
            print(subset)
            cv2.imshow('heatmap', heatmap_avg[:-1].max(axis=0))
            cv2.imshow('to_plot', to_plot)
            cv2.waitKey(0)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        type=str,
                        default='train',
                        help="operation mode")
    parser.add_argument("--weights_path",
                        type=str,
                        default='weights/{}.pkl'.format(full_weight_name),
                        help="model path for inference")
    parser.add_argument("--input_dir",
                        type=str,
                        default=None)
    parser.add_argument("--output_dir",
                        type=str,
                        default=None)
    parser.add_argument("--batch_size",
                        type=int,
                        default=1,
                        help="size of the batches")
    parser.add_argument("--n_epochs",
                        type=int,
                        default=50,
                        help="number of epochs of training")
    parser.add_argument("--lr",
                        type=float,
                        # default=0.001,
                        default=0.016,
                        help="adam: learning rate")
    parser.add_argument("--b1",
                        type=float,
                        default=0.5,
                        # default=0.0,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2",
                        type=float,
                        default=0.999,
                        # default=0.9,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--scheduler",
                        type=bool,
                        # default=False,
                        default=True,
                        help="number of generated images for inference")
    parser.add_argument("--sample_interval",
                        type=int,
                        default=400,
                        help="interval between image sampling")
    opt = parser.parse_args()

    print(opt)
    # train(opt)

    if opt.mode == 'train':
        train(opt)
    elif opt.mode == 'val':
        val(opt)
    elif opt.mode == 'test':
        test(opt)

