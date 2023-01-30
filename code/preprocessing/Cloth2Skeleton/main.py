from re import S, sub
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
from post import decode_pose
import yaml

# transfer caffe model to pytorch which will match the layer name
def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        try:
            transfered_model_weights[weights_name] = model_weights['.'.join(weights_name.split('.')[1:])]
        except:
            continue
    return transfered_model_weights


def draw_bodypose(canvas, keypoints, body_part='top'):
    stickwidth = 4

    if body_part == 'top':
        limbSeq = [[1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9], \
               [8, 12]]
    elif body_part:
        limbSeq = [[8, 9], [9, 10], [10, 11], [8, 12], [12, 13], [13, 14]]
        

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


def train(config):
    os.makedirs(os.path.join('weights', config['TRAINING_CONFIG']['TRAIN_DIR']), exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('./runs/{}'.format(config['TRAINING_CONFIG']['TRAIN_DIR']))

    criterion_L2 = nn.MSELoss().cuda()

    train_dataset = ClothDataset(mode=config['MODE'], path=config['TRAINING_CONFIG']['DATA_DIR'], body_part=config['MODEL_CONFIG']['BODY_PART'], train_ratio=config['TRAINING_CONFIG']['TRAIN_RATIO'])
    dataloader_train = DataLoader(dataset=train_dataset, batch_size=config['TRAINING_CONFIG']['BATCH_SIZE'], num_workers=16, drop_last=True, shuffle=True)

    model = bodypose_model(body_part=config['MODEL_CONFIG']['BODY_PART'])

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
    if config['TRAINING_CONFIG']['OPTIM'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['TRAINING_CONFIG']['LR'], momentum=0.9, weight_decay=0.0005)
    elif config['TRAINING_CONFIG']['OPTIM'] == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=config['TRAINING_CONFIG']['LR'],
                                    betas=(config['TRAINING_CONFIG']['BETA1'], config['TRAINING_CONFIG']['BETA2']))
    else:
        print("Unsupport optimizer")
        exit()

    if config['TRAINING_CONFIG']['SCHEDULER'] == 'cosine':  
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['TRAINING_CONFIG']['T_MAX'], eta_min=config['TRAINING_CONFIG']['ETA_MIN'])
    
    Tensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor

    weights_path = os.path.join('weights', config['TRAINING_CONFIG']['TRAIN_DIR'], '{}.pkl'.format(config['TRAINING_CONFIG']['TRAIN_DIR']))
    # Load pretrained weights
    if os.path.isfile(weights_path):
        print("=> loading checkpoint '{}'".format(weights_path))
        checkpoint = torch.load(weights_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if config['TRAINING_CONFIG']['SCHEDULER']:
            scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(weights_path, checkpoint['epoch']))
    else:
        checkpoint = torch.load(config['TRAINING_CONFIG']['PRETRAIN_DIR'])
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = 0
        print("=> no checkpoint found at '{}'".format(weights_path))

    # Start training
    for epoch_index in range(start_epoch, config['TRAINING_CONFIG']['EPOCH']):
        print('epoch_index=', epoch_index)
        start = time.time()
        for param_group in optimizer.param_groups:
            print('lr: ' + str(param_group['lr']))

        # in each minibatch
        pbar = tqdm(dataloader_train, desc='training')

        for batchIdx, (imgs, heat_maps, pafs) in enumerate(pbar):
            iterNum = epoch_index * len(pbar) + batchIdx

            imgs = Variable(imgs.cuda())
            heat_maps = Variable(heat_maps.cuda())
            pafs = Variable(pafs.cuda())
            if config['TRAINING_CONFIG']['VIS']:
                cv2.imshow('heatmap', heat_maps[0, :-1].cpu().numpy().max(axis=0) / 2 + 0.5)
                cv2.imshow('pafs', abs(pafs[0].cpu().numpy()).max(axis=0))
                cv2.imshow('img', imgs[0].permute(1,2,0).cpu().numpy())
            
            PAF, JH = model(imgs)

            if config['TRAINING_CONFIG']['VIS']:
                cv2.imshow('JH', JH[0, :-1].detach().cpu().numpy().max(axis=0) / 2 + 0.5)
                cv2.imshow('PAF', abs(PAF[0].detach().cpu().numpy()).max(axis=0))
                cv2.waitKey(0)

            loss_JH = criterion_L2(JH, heat_maps)
            loss_PAF = criterion_L2(PAF, pafs)
            loss = loss_PAF + loss_JH
            pbar.set_description("Loss: {}".format(loss.item()))  
            writer.add_scalar("Loss", loss, iterNum)
            writer.add_scalar("Loss_JH", loss_JH, iterNum)
            writer.add_scalar("Loss_PAF", loss_PAF, iterNum)

            # Tips for accumulative backward
            accum_steps = 1
            loss = loss / accum_steps                
            loss.backward()                           
            if (epoch_index*len(pbar) + batchIdx + 1) % accum_steps == 0:           
                optimizer.step()                       
                optimizer.zero_grad() 

        endl = time.time()
        print('Costing time:', (endl-start)/60)
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(current_time)
        save_info = {
            'epoch': epoch_index + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        if config['TRAINING_CONFIG']['SCHEDULER']:
            scheduler.step()
            save_info['scheduler'] = scheduler.state_dict()

        if epoch_index % config['TRAINING_CONFIG']['SAVE_STEP'] == config['TRAINING_CONFIG']['SAVE_STEP']-1:
            weight_name = '{}_{}.pkl'.format(weights_path.split('.')[0], epoch_index + 1)
            torch.save(save_info, weight_name)
        torch.save(save_info, weights_path)


def val(config):
    model = bodypose_model(body_part=config['MODEL_CONFIG']['BODY_PART']).cuda()
    weights_path = os.path.join('weights', config['TRAINING_CONFIG']['TRAIN_DIR'], '{}.pkl'.format(config['TRAINING_CONFIG']['TRAIN_DIR']))
    weight_name = '{}_{}.pkl'.format(weights_path.split('.')[0], config['VAL_CONFIG']['VAL_EPOCH'])
    checkpoint = torch.load(weight_name)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    test_dataset = ClothDataset(mode="val", path=config['VAL_CONFIG']['DATA_DIR'], body_part=config['MODEL_CONFIG']['BODY_PART'])
    dataloader_test = DataLoader(dataset=test_dataset, batch_size=1, num_workers=1, drop_last=True, shuffle=True)

    pbar = tqdm(dataloader_test, desc='testing')
    correct = 0

    if config['MODEL_CONFIG']['BODY_PART'] == 'top':
        useful = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12]
        limbSeq = [[1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9], [8, 12]]
    elif config['MODEL_CONFIG']['BODY_PART'] == 'bottom':
        useful = [8, 9, 10, 11, 12, 13, 14]
        limbSeq = [[8, 9], [9, 10], [10, 11], [8, 12], [12, 13], [13, 14]]

    for batchIdx, (imgs, imgs_origin) in enumerate(pbar):
        imgs = Variable(imgs.cuda())
        PAF, JH = model(imgs)
        heatmap_avg = JH[0].detach().cpu().numpy() / 2 + 0.5
        paf_avg = PAF[0].detach().cpu().numpy()
        param = {'thre1': 0.1, 'thre2': 0, 'thre3': 0}
        canvas, to_plot, candidate, subset = decode_pose(np.zeros((640,480,3)), param, heatmap_avg, paf_avg, limbSeq)
        keypoints = np.zeros((25, 2))
        for i in useful:
            point = subset[0, i].astype(int)
            if point != -1:
                keypoints[i] = candidate[point, :2]

        origin_img = imgs_origin[0].detach().cpu().numpy()
        origin_img = cv2.normalize(origin_img,  None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        skeleton = draw_bodypose(origin_img.copy(), keypoints, body_part=config['MODEL_CONFIG']['BODY_PART'])
        cv2.imshow('pred', skeleton)
        cv2.waitKey(0)

            
def test(config):
    os.makedirs(config['TEST_CONFIG']['OUTPUT_DIR'], exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(config['TEST_CONFIG']['OUTPUT_DIR']), "product_skeleton"), exist_ok=True)
    model = bodypose_model(body_part=config['MODEL_CONFIG']['BODY_PART']).cuda()
    weights_path = os.path.join('weights', config['TRAINING_CONFIG']['TRAIN_DIR'], '{}.pkl'.format(config['TRAINING_CONFIG']['TRAIN_DIR']))
    weight_name = '{}_{}.pkl'.format(weights_path.split('.')[0], config['TEST_CONFIG']['TEST_EPOCH'])
    checkpoint = torch.load(weight_name)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    test_dataset = ClothDataset(mode="test", path=config['TEST_CONFIG']['INPUT_DIR'], body_part=config['MODEL_CONFIG']['BODY_PART'])
    dataloader_test = DataLoader(dataset=test_dataset, batch_size=1, num_workers=1, drop_last=True, shuffle=False)

    pbar = tqdm(dataloader_test, desc='testing')
    correct = 0
    if config['MODEL_CONFIG']['BODY_PART'] == 'top':
        useful = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12]
        limbSeq = [[1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9], [8, 12]]
    elif config['MODEL_CONFIG']['BODY_PART'] == 'bottom':
        useful = [8, 9, 10, 11, 12, 13, 14]
        limbSeq = [[8, 9], [9, 10], [10, 11], [8, 12], [12, 13], [13, 14]]

    for batchIdx, (imgs, imgNames, imgs_origin) in enumerate(pbar):
        imgs = Variable(imgs.cuda())
        PAF, JH = model(imgs)
        heatmap_avg = JH[0].detach().cpu().numpy() / 2 + 0.5
        paf_avg = PAF[0].detach().cpu().numpy()
        param = {'thre1': 0.1, 'thre2': 0, 'thre3': 0}
        canvas, to_plot, candidate, subset = decode_pose(np.zeros(imgs_origin[0].shape), param, heatmap_avg, paf_avg, limbSeq)
        keypoints = np.zeros((25, 2))
        if len(subset.shape) == 2:
            for i in useful:
                point = subset[0, i].astype(int)
                if point != -1:
                    keypoints[i] = candidate[point, :2]

        origin_img = imgs_origin[0].detach().cpu().numpy()
        origin_img = cv2.normalize(origin_img,  None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        skeleton = draw_bodypose(origin_img.copy(), keypoints, body_part=config['MODEL_CONFIG']['BODY_PART'])
        cv2.imwrite(os.path.join(os.path.dirname(config['TEST_CONFIG']['OUTPUT_DIR']), "product_skeleton", '{}.jpg'.format(imgNames[0])), skeleton)

        pose_format = {
            "version": 1.3,
            "people": [
                        {
                            "person_id": [-1],
                            "pose_keypoints_2d": keypoints.reshape(-1).tolist(),
                            "face_keypoints_2d": [],
                            "hand_left_keypoints_2d": [],
                            "hand_right_keypoints_2d": [],
                            "pose_keypoints_3d": [],
                            "face_keypoints_3d": [],
                            "hand_left_keypoints_3d": [],
                            "hand_right_keypoints_3d": []
                        }
                    ]
                }

        with open(os.path.join(config['TEST_CONFIG']['OUTPUT_DIR'], '{}_keypoints.json'.format(imgNames[0])), 'w') as f:
            json.dump(pose_format, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        type=str,
                        default='train',
                        help="operation mode")
    parser.add_argument("--input_dir",
                        type=str,
                        default=None)
    parser.add_argument("--output_dir",
                        type=str,
                        default=None)
    parser.add_argument("--config",
                        type=str,
                        default='configs/config_top_v2_allData.yaml')
    parser.add_argument("--vis",
                        type=bool,
                        default=False)
    opt = parser.parse_args()

    config = yaml.load(open(opt.config, 'r'), Loader=yaml.FullLoader)
    config['MODE'] = opt.mode
    config['TRAINING_CONFIG']['VIS'] = opt.vis
    config['TEST_CONFIG']['INPUT_DIR'] = opt.input_dir
    config['TEST_CONFIG']['OUTPUT_DIR'] = opt.output_dir

    print(opt)

    if opt.mode == 'train':
        train(config)
    elif opt.mode == 'val':
        val(config)
    elif opt.mode == 'test':
        test(config)

