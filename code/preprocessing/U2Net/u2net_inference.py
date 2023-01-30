import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB
import argparse
from tqdm import tqdm
import cv2

tensor_transform_woResize = transforms.Compose([  \
                            transforms.ToTensor(),   \
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    
    mask = (predict_np >= 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(d_dir,img_name.split('.')[0]+'.png'), mask)


def main(opt):

    # --------- 1. get image path and name ---------
    model_name='u2net'#u2netp


    input_dir = opt.input_dir
    output_dir = opt.output_dir
    model_dir = os.path.join('saved_models', model_name, model_name + '.pth')

    img_name_list = glob.glob(input_dir + os.sep + '*')

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                    ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in tqdm(enumerate(test_salobj_dataloader), total=len(test_salobj_dataloader)):

        # print("\r inferencing:",img_name_list[i_test].split(os.sep)[-1], end=" ")

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        # print("output_dir = ", output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        save_output(img_name_list[i_test],pred,output_dir)

        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
                        type=str,
                        default=None)
    parser.add_argument("--output_dir",
                        type=str,
                        default=None)
    opt = parser.parse_args()

    main(opt)
