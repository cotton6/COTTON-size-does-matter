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
from dataLoader import Neckline_dataset
import torchvision.models as models
import imageio
import shutil
import cv2


def test(opt):
    # Create model
    model = models.segmentation.fcn_resnet50(num_classes=21, pretrained=True).cuda().eval()
    max_value = 0
    max_idx = 0

    best_score = 0
    best_weight = None
    checkpoint = torch.load(opt.weights_path)
    model.load_state_dict(checkpoint['state_dict'])
    # Original validation Set
    dataset_test = Neckline_dataset(mode='test', path=opt.input_dir)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=1, num_workers=11, drop_last=True, shuffle=True)
        
    pbar = tqdm(dataloader_test, desc='testing')
    correct = 0

    output_folder = opt.output_dir if opt.output_dir is not None else opt.input_dir + '_rm'

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    for batchIdx, (imgs, imgName) in enumerate(pbar):
        imgName = imgName[0]
        imgs = Variable(imgs.cuda())
        prediction = model(imgs)['out']
        prediction = prediction.argmax(dim=1, keepdim=True) 
        prediction = prediction.squeeze().detach().cpu().numpy()
        mask = prediction == 1
        # prediction = cm[prediction]

        # tensor_img = imgs[0].permute(1, 2, 0).detach().cpu().numpy()[:, :, ::-1]
        # tensor_img = cv2.normalize(tensor_img,  None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        tensor_img = cv2.imread(os.path.join(opt.input_dir, imgName + '.jpg'))
        test = cv2.imread(os.path.join(opt.input_dir, imgName + '.jpg'), 0)
        
        tensor_img[mask] = 255
        binary_mask = (tensor_img.mean(axis=-1) != 255).astype(np.uint8)
        
        kernel = np.zeros((13,13), np.uint8)
        kernel[6,6] = 1
        kernel[7,6] = 1
        kernel[8,6] = 1
        kernel[9,6] = 1
        kernel[10,6] = 1
        kernel[11,6] = 1
        kernel[12,6] = 1
        binary_mask1 = cv2.erode(binary_mask, kernel, iterations = 1)

        kernel = np.zeros((13,13), np.uint8)
        # kernel[0,6] = 1
        kernel[1,6] = 1
        kernel[2,6] = 1
        kernel[3,6] = 1
        kernel[4,6] = 1
        kernel[5,6] = 1
        kernel[6,6] = 1
        binary_mask2 = cv2.dilate(binary_mask1, kernel, iterations = 1)

        binary_mask2 = binary_mask2.astype(np.bool)
        tensor_img[~binary_mask2] = 255

        cv2.imwrite('{}/{}.png'.format(output_folder, imgName),tensor_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path",
                        type=str,
                        default='pretrained.pkl',
                        help="model path for inference")
    parser.add_argument("--input_dir",
                        type=str,
                        default='test_dataset',
                        help="Choose test on test or train")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Choose test on test or train")


    opt = parser.parse_args()
    test(opt)
