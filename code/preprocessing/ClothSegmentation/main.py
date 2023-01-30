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
from dataLoader import Neckline_dataset, Cloth_Segmentation_Dataset
import torchvision.models as models
from mean_iou_evaluate import read_masks, mean_iou_score
import imageio
import shutil
import cv2


# full_weight_name = 'fcn_resnet50'
# full_weight_name = 'cloth_parser_all'
# full_weight_name = 'cloth_parser_all_3class'
full_weight_name = 'cloth_parser_all_aug'
NUM_CLASS = 21

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

def train(opt):
    os.makedirs('weights', exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('./runs/{}'.format(full_weight_name))

    dataset_train = Cloth_Segmentation_Dataset()
    dataloader_train = DataLoader(dataset_train, batch_size=opt.batch_size, num_workers=12, drop_last=True, shuffle=True)

    # Create model
    # model = models.segmentation.fcn_resnet50(num_classes=NUM_CLASS, pretrained=True).cuda()
    model = models.segmentation.fcn_resnet50(num_classes=NUM_CLASS, pretrained_backbone=True).cuda()
    optimizer = torch.optim.SGD(model.parameters() , lr=opt.lr, momentum=0.99, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20, 30, 40, 50, 60], gamma=0.5)

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
    criterion_dice = DiceLoss().cuda()
    
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
            # cv2.imshow('img', imgs[0].permute(1,2,0).numpy())
            # cv2.imshow('label',((label[0].numpy()>0)*255).astype(np.uint8))
            # print((label[0].numpy()==2).sum())
            # print((label[0].numpy()).max())
            # cv2.waitKey(0)
            imgs = Variable(imgs.cuda())
            label = Variable(label.cuda())
            b, h, w = label.shape
            label_onehot = Variable(make_one_hot(label.view(b, -1).unsqueeze(1), NUM_CLASS).cuda()).reshape(b, -1, h, w)
            prediction = model(imgs)['out']

            # Train with Source
            loss_CE = criterion_CE(prediction, label)
            loss_dice = criterion_dice(prediction, label_onehot)
            loss = loss_CE + loss_dice

            writer.add_scalar("Loss_CE", loss_CE, epoch_index*len(pbar) + batchIdx)
            writer.add_scalar("Loss_dice", loss_dice, epoch_index*len(pbar) + batchIdx)
            writer.add_scalar("Loss", loss, epoch_index*len(pbar) + batchIdx)

            # Tips for accumulative backward
            accum_steps = 16
            loss = loss / accum_steps                
            loss.backward()                           
            if (epoch_index*len(pbar) + batchIdx + 1) % accum_steps == 0:           
                optimizer.step()                       
                optimizer.zero_grad()   
            
            # Calculate mIOU
            prediction = prediction.argmax(dim=1, keepdim=True) 
            prediction = prediction.squeeze().detach().cpu().numpy()
            label = label.squeeze().detach().cpu().numpy()

            mean_iou = []
            for i in range(3):
                tp_fp = np.sum(prediction == i)
                tp_fn = np.sum(label == i)
                tp = np.sum((prediction == i) * (label == i))
                if (tp_fp + tp_fn - tp) == 0:
                    continue
                iou = tp / (tp_fp + tp_fn - tp)
                mean_iou.append(iou)
            mean_iou = sum(mean_iou) / len(mean_iou)

            writer.add_scalar("mIOU", mean_iou, epoch_index*len(pbar) + batchIdx)
            pbar.set_description("Loss: {}, mIOU: {}".format(loss.item()*accum_steps, mean_iou))   

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
                
    writer.close()

def validate(opt):
    # Create model
    # model = models.segmentation.fcn_resnet50(num_classes=NUM_CLASS).cuda().eval()
    # model = models.segmentation.fcn_resnet50(num_classes=NUM_CLASS, pretrained=True).cuda().eval()
    model = models.segmentation.fcn_resnet50(num_classes=NUM_CLASS, pretrained_backbone=True).cuda().eval()
    max_value = 0
    max_idx = 0

    colormap = [[0,0,0], [61,245,61], [250,250,55]]
    cm = np.array(colormap).astype('uint8')


    best_score = 0
    best_weight = None
    # 930
    # 950 0.9651404365438314.
    # cloth_parser_all_aug: Best weight is 1850 epoch with mIOU equals 0.9533659947443819.
    for i in range(900, 2050, 50):
        print(i)
        weight_name = '{}_{}.pkl'.format(opt.weights_path.split('.')[0], i)
        if not os.path.isfile(weight_name):
            break
        checkpoint = torch.load(weight_name)
        # checkpoint = torch.load(opt.weights_path)
        model.load_state_dict(checkpoint['state_dict'])
        # Original validation Set
        dataset_valid = Cloth_Segmentation_Dataset(mode='valid')
        dataloader_valid = DataLoader(dataset=dataset_valid, batch_size=1, num_workers=11, drop_last=True, shuffle=True)
            
        pbar = tqdm(dataloader_valid, desc='testing')
        correct = 0

        gt_folder = 'valid_gt'
        if not os.path.isdir(gt_folder):
            os.mkdir(gt_folder)

        output_folder = 'valid_output'
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        output_folder_add = 'valid_output_add'
        if not os.path.isdir(output_folder_add):
            os.mkdir(output_folder_add)


        for batchIdx, (imgs, imgName) in enumerate(pbar):
            imgName = imgName[0]
            imgs = Variable(imgs.cuda())
            prediction = model(imgs)['out']
            prediction = prediction.argmax(dim=1, keepdim=True) 
            prediction = prediction.squeeze().detach().cpu().numpy()
            prediction = cm[prediction]

            tensor_img = imgs[0].permute(1, 2, 0).detach().cpu().numpy()
            tensor_img = cv2.normalize(tensor_img,  None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            superimposed_img = cv2.addWeighted(prediction[:, :, ::-1], 0.3, tensor_img[:, :, ::-1], 0.7, 0)
            cv2.imwrite('{}/{}.png'.format(output_folder_add, imgName),superimposed_img)

            imageio.imwrite('{}/{}.png'.format(output_folder, imgName),prediction)
            # shutil.copy2('cloth_segmentation_dataset/masks_vis/{}.png'.format(imgName), gt_folder)
        pred = read_masks('valid_output')
        labels = read_masks('valid_gt')
        temp_score = mean_iou_score(pred, labels)
        if temp_score > best_score:
            best_score = temp_score
            best_weight = i
    print("Best weight is {} epoch with mIOU equals {}.".format(best_weight, best_score))


def test(opt):
    # Create model
    model = models.segmentation.fcn_resnet50(num_classes=NUM_CLASS, pretrained_backbone=True).cuda().eval()
    colormap = [[0,0,0], [61,245,61], [250,250,55]]
    cm = np.array(colormap).astype('uint8')
    weight_name = '{}_{}.pkl'.format(opt.weights_path.split('.')[0], 1850)
    checkpoint = torch.load(weight_name)
    # checkpoint = torch.load(opt.weights_path)
    model.load_state_dict(checkpoint['state_dict'])

    # Original validation Set
    dataset_test = Cloth_Segmentation_Dataset(mode='test', path=opt.input_dir)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=1, num_workers=11, drop_last=True, shuffle=True)
        
    pbar = tqdm(dataloader_test, desc='testing')
    correct = 0

    output_folder = opt.output_dir
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    output_folder_vis = os.path.join(output_folder, 'vis')
    if not os.path.isdir(output_folder_vis):
        os.mkdir(output_folder_vis)


    for batchIdx, (imgs, imgName, imgs_origin) in enumerate(pbar):
        imgName = imgName[0]
        origin_img = imgs_origin[0].detach().cpu().numpy()
        h, w = origin_img.shape[:2]

        imgs = Variable(imgs.cuda())
        prediction = model(imgs)['out']
        prediction = prediction.argmax(dim=1, keepdim=True) 
        prediction = prediction.squeeze().detach().cpu().numpy().astype(np.uint8)

        tensor_img = imgs[0].permute(1, 2, 0).detach().cpu().numpy()[:, :, ::-1]
        tensor_img = cv2.normalize(tensor_img,  None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        superimposed_img = cv2.addWeighted(cm[prediction][:, :, ::-1], 0.3, tensor_img[:, :, ::-1], 0.7, 0)

        prediction = cv2.resize(prediction, (w, h), interpolation=cv2.INTER_NEAREST)
        superimposed_img = cv2.resize(superimposed_img, (w, h), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite('{}/{}.png'.format(output_folder_vis, imgName),superimposed_img)
        imageio.imwrite('{}/{}.png'.format(output_folder, imgName),prediction)  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        type=str,
                        choices=['train', 'test', 'valid', 'preprocess'],
                        default='train',
                        # required=True,
                        help="operation mode")
    parser.add_argument("--weights_path",
                        type=str,
                        default='weights/{}.pkl'.format(full_weight_name),
                        help="model path for inference")
    parser.add_argument("--input_dir",
                        type=str,
                        default='cloth_segmentation_dataset/images',
                        help="model path for inference")
    parser.add_argument("--output_dir",
                        type=str,
                        default='test_output',
                        help="model path for inference")
    parser.add_argument("--n_epochs",
                        type=int,
                        default=2000,
                        help="number of epochs of training")
    parser.add_argument("--batch_size",
                        type=int,
                        default=2,
                        help="size of the batches")
    parser.add_argument("--lr",
                        type=float,
                        # default=0.00005,
                        default=0.001,
                        help="adam: learning rate")



    opt = parser.parse_args()
    print(opt)

    if opt.mode == 'train':
        train(opt)
    elif opt.mode == 'valid':
        validate(opt)
    else:
        test(opt)

# 