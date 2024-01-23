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
from dataLoader import Cloth_Dataset
import torchvision.models as models
import cv2
import json

# full_weight_name = 'skirt_clf_SGD'
full_weight_name = 'skirt_clf_Adam'


class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        ResEncoder = models.resnet18(pretrained=True)
        self.backbone = torch.nn.Sequential(*(list(ResEncoder.children())[:-1])).cuda()

        self.cf = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 2),
        )

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.Conv3d)):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Get input size
        x = self.backbone(x)

        b, c, h, w = x.shape
        x = x.reshape(b, -1)
        pred = self.cf(x)

        return pred


def train(opt):
    os.makedirs('weights', exist_ok=True)
    dataloader_train = DataLoader(dataset=Cloth_Dataset(), batch_size=opt.batch_size, num_workers=11, drop_last=True, shuffle=True)

    # Create model
    model = classifier().cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.00001)
    optimizer.zero_grad()

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10*i for i in range(1, 11)], gamma=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    # Load pretrained weights
    if os.path.isfile(opt.weights_path):
        print("=> loading checkpoint '{}'".format(opt.weights_path))
        checkpoint = torch.load(opt.weights_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(opt.weights_path, checkpoint['epoch']))
    else:
        start_epoch = 0
        print("=> no checkpoint found at '{}'".format(opt.weights_path))

    criterion_CE = nn.CrossEntropyLoss().cuda()
    model.train()


    # Start training
    for epoch_index in range(start_epoch, opt.n_epochs):

        print('epoch_index=', epoch_index)

        for param_group in optimizer.param_groups:
            print('lr: ' + str(param_group['lr']))

        start = time.time()


        # in each minibatch
        pbar = tqdm(dataloader_train, desc='training')

        for batchIdx, (imgs, label) in enumerate(pbar):

            imgs = Variable(imgs.cuda())
            label = Variable(label.cuda())

            prediction = model(imgs)

            # Train with Source
            loss = criterion_CE(prediction, label)

            pbar.set_description("Loss: {}".format(loss.item()))   

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        endl = time.time()
        print('Costing time:', (endl-start)/60)
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(current_time)
        save_info = {
            'epoch': epoch_index + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        if epoch_index % 50 == 49:
            weight_name = '{}_{}.pkl'.format(opt.weights_path.split('.')[0], epoch_index + 1)
            torch.save(save_info, weight_name)
        torch.save(save_info, opt.weights_path)


def val(opt):
    # Create model
    model = classifier().cuda().eval()
    # weight_name = '{}_{}.pkl'.format(opt.weights_path.split('.')[0], 50)
    # checkpoint = torch.load(weight_name)
    checkpoint = torch.load(opt.weights_path)
    model.load_state_dict(checkpoint['state_dict'])
    # Original validation Set
    dataloader_test = DataLoader(dataset=Cloth_Dataset(mode=opt.mode), batch_size=1, num_workers=11, drop_last=True, shuffle=True)
        
    pbar = tqdm(dataloader_test, desc='testing')
    correct = 0
    # success_dict = {}

    for batchIdx, (imgs, label) in enumerate(pbar):
        imgs = Variable(imgs.cuda())
        # img_array = imgs[0].permute(1,2,0).detach().cpu().numpy()[:,:,::-1]
        # img_array = cv2.normalize(img_array,  None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # cv2.imshow('img', img_array)
        # cv2.waitKey(0)
        label = Variable(label.cuda())
        prediction = model(imgs)
        prediction = prediction.argmax(dim=1, keepdim=True) 
        correct += prediction.eq(label.view_as(prediction)).sum().item()
        # success_dict[label.item()] = success_dict.get(label.item(), 0) + prediction.eq(label.view_as(prediction)).sum().item()
    # print(correct)
    # print(success_dict)
    print('\nAccuracy validation: {}/{} ({:.0f}%)\n'.format(correct, len(dataloader_test.dataset),
        100. * correct / len(dataloader_test.dataset)))


def test(opt):
    os.makedirs(opt.output_dir, exist_ok=True)
    # Create model
    model = classifier().cuda().eval()
    # weight_name = '{}_{}.pkl'.format(opt.weights_path.split('.')[0], 50)
    # checkpoint = torch.load(weight_name)
    checkpoint = torch.load(opt.weights_path)
    model.load_state_dict(checkpoint['state_dict'])
    # Original validation Set
    dataloader_test = DataLoader(dataset=Cloth_Dataset(mode=opt.mode, path=opt.input_dir), batch_size=1, num_workers=11, drop_last=True, shuffle=True)
        
    pbar = tqdm(dataloader_test, desc='testing')
    correct = 0
    # success_dict = {}

    for batchIdx, (imgs, imgNames) in enumerate(pbar):
        imgs = Variable(imgs.cuda())
        prediction = model(imgs)
        prediction = prediction.argmax(dim=1, keepdim=True) 
        
        info_format = {
            "sub_type": prediction.reshape(-1).tolist(),
                }
        save_info_name = os.path.join(opt.output_dir, '{}.json'.format(imgNames[0]))
        if os.path.isfile(save_info_name):
            with open(save_info_name, 'r') as f:
                product_info = json.load(f)
            product_info.update(info_format)
            with open(save_info_name, 'w') as f:
                json.dump(product_info, f)
        else:
            with open(save_info_name, 'w') as f:
                json.dump(info_format, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        type=str,
                        choices=['train', 'val', 'test', 'preprocess'],
                        default='train',
                        # required=True,
                        help="operation mode")
    parser.add_argument("--weights_path",
                        type=str,
                        default='weights/{}.pkl'.format(full_weight_name),
                        help="model path for inference")
    parser.add_argument("--n_epochs",
                        type=int,
                        default=50,
                        help="number of epochs of training")
    parser.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="size of the batches")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-4,
                        # default=0.016,
                        help="adam: learning rate")
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--brand", type=str)

    opt = parser.parse_args()
    print(opt)

    if opt.mode == 'train':
        train(opt)
    elif opt.mode == 'val':
        val(opt)
    elif opt.mode == 'preprocess':
        opt.mode = 'test'
        data_folder = os.path.join('../parse_filtered_Data', opt.brand)
        cats = [os.path.basename(cat) for cat in glob.glob(os.path.join(data_folder, '*'))]
        for cat in cats:
            print(cat)
            cat_folder = os.path.join(data_folder, cat)
            opt.input_dir = os.path.join(cat_folder, 'product')
            opt.output_dir = os.path.join(cat_folder, 'product_info')
            test(opt)
    else:
        test(opt)

