import os
import torch
import torchvision
import shutil
from PIL import Image
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import glob
import cv2
import argparse
from tqdm import tqdm

label_colours_20 = [(0,0,0)
            , (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0)
            ,(0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0)
            ,(52,86,128), (0,128,0), (0,0,255), (51,170,221), (0,255,255)
            ,(85,255,170), (170,255,85), (255,255,0), (255,170,0)]

label_colours_15 = [(0,0,0)
    # 0=Background
    ,(255,0,0),(0,0,255),(85,51,0),(255,85,0),(0,119,221)
    # 1=Hat,  2=Hair,    3=Glove, 4=Sunglasses, 5=UpperClothes
    ,(0,0,85),(0,85,85),(51,170,221),(0,255,255),(85,255,170)
    # 6=Dress, 7=Coat, 8=Socks, 9=Pants, 10=Jumpsuits/Neck
    ,(170,255,85),(255,255,0),(255,170,0),(85,85,0),(0,255,255)
    # 11=Scarf, 12=Skirt, 13=Face, 14=LeftArm, 15=RightArm
    ,(85,255,170),(170,255,85),(255,255,0),(255,170,0)]
    # 16=LeftLeg, 17=RightLeg, 18=LeftShoe, 19=RightShoe

label_colours_17 = [(0,0,0)
    # 0=Background
    ,(255,0,0),(0,0,255),(85,51,0),(255,85,0),(0,119,221)
    # 1=Hat,  2=Hair,    3=Glove, 4=Sunglasses, 5=UpperClothes
    ,(0,0,85),(0,85,85),(51,170,221),(0,255,255),(85,255,170)
    # 6=Dress, 7=Coat, 8=Socks, 9=Pants, 10=Jumpsuits/Neck
    ,(170,255,85),(255,255,0),(255,170,0),(85,85,0)
    ,(128,0,0),(0,85,0)
    # 11=Scarf, 12=Skirt, 13=Face, 14=LeftArm, 15=RightArm
    ,(170,255,85),(255,255,0),(255,170,0)]
    # 16=LeftLeg, 17=RightLeg, 18=LeftShoe, 19=RightShoe

os.environ["CUDA_VISIBLE_DEVICES"]="0"


"""
CIHP parsing class definition:
        0:  Background              1:  Hat
        2:  Hair                    3:  Glove
        4:  Sunglasses              5:  Upper-clothes
        6:  Dress                   7:  Coat
        8:  Socks                   9:  Pants
        10: Jumpsuits / tosor-skin  11: Scarf
        12: Skirt                   13: Face
        14: Left-arm                15: Right-arm
        16: Left-leg                17: Right-leg
        18: Left-shoe               19: Right-shoe

    New parsing:
        LIP           -> New
        0             -> 0  Background
        1, 2          -> 1  Hair
        4, 13         -> 2  Face
        10            -> 3  Neck
        5             -> 4  Upper-clothes
        7             -> 5  Coat
        6             -> 6  dress
        9, 12         -> 7  Lower-clothes
        14            -> 8  Left-arm
        15            -> 9  Right-arm
        16            -> 10 Left-leg
        17            -> 11 Right-leg
        18            -> 12 Left-shoe 
        19            -> 13 Right-shoe
        3, 8, 11      -> 14 Accessories
"""

def merge_20to15(CIHP_folder, ATR_folder, output_folder):

    CIHP_parse_imgs = glob.glob(os.path.join(CIHP_folder, '*.png'))
    CIHP_parse_imgs.sort(key=lambda x:int(os.path.basename(x).split('_')[0]))
    ATR_parse_imgs = glob.glob(os.path.join(ATR_folder, '*.png'))
    ATR_parse_imgs.sort(key=lambda x:int(os.path.basename(x).split('_')[0]))

    assert(len(CIHP_parse_imgs) == len(ATR_parse_imgs))

    for CIHP_parse_name, ATR_parse_name in tqdm(zip(CIHP_parse_imgs, ATR_parse_imgs)):
        CIHP_parse_img = cv2.imread(CIHP_parse_name, 0)
        ATR_parse_img = cv2.imread(ATR_parse_name, 0)

        bg              = (CIHP_parse_img == 0)
        hair            = (CIHP_parse_img == 1) + (CIHP_parse_img == 2)
        Face            = (CIHP_parse_img == 4) + (CIHP_parse_img == 13)
        Neck            = (CIHP_parse_img == 10)
        Upper_clothes   = (CIHP_parse_img == 5)
        Coat            = (CIHP_parse_img == 7)
        Dress           = (CIHP_parse_img == 6)
        Lower_clothes   = (CIHP_parse_img == 9) + (CIHP_parse_img == 12)
        Left_arm        = (CIHP_parse_img == 14)
        Right_arm       = (CIHP_parse_img == 15)
        Left_leg        = (CIHP_parse_img == 16)
        Right_leg       = (CIHP_parse_img == 17)
        Left_shoe       = (CIHP_parse_img == 18)
        Right_shoe      = (CIHP_parse_img == 19)
        Accessories     = (CIHP_parse_img == 3) + (CIHP_parse_img == 8) + (CIHP_parse_img == 11)
        
        img = np.zeros_like(CIHP_parse_img)
        img[bg            ] = 0
        img[hair          ] = 1
        img[Face          ] = 2
        img[Neck          ] = 3
        img[Upper_clothes ] = 4
        img[Coat          ] = 5
        img[Dress         ] = 6
        img[Lower_clothes ] = 7
        img[Left_arm      ] = 8
        img[Right_arm     ] = 9
        img[Left_leg      ] = 10
        img[Right_leg     ] = 11
        img[Left_shoe     ] = 12
        img[Right_shoe    ] = 13
        img[Accessories   ] = 14

        if ATR_parse_imgs is not None:
            belt        = np.logical_and(ATR_parse_img == 8, CIHP_parse_img == 0)
            bag         = np.logical_and(ATR_parse_img == 16, CIHP_parse_img == 0)

            img[belt          ] = 14
            img[bag           ] = 14

        img = Image.fromarray(img.astype('uint8'),'L')
        img.save('{}/{}'.format(output_folder, os.path.basename(ATR_parse_name)))
        img = np.array(label_colours_17)[img]
        img = Image.fromarray(img.astype('uint8'))
        img.save('{}/vis/{}'.format(output_folder, os.path.basename(ATR_parse_name)))
        


def merge_label(opt):
    data_folder = os.path.join(opt.root, opt.brand)
    cats = [os.path.basename(name) for name in glob.glob(os.path.join(data_folder, '*'))]
    for cat in cats:
        print('='*20 + cat + '='*20)
        tar_dir = os.path.join(opt.root, opt.brand, cat, 'parsing_merge')
        os.makedirs(tar_dir, exist_ok=True)
        os.makedirs(os.path.join(tar_dir, "vis"), exist_ok=True)
        CIHP_folder = os.path.join(data_folder, cat, 'CIHP')
        ATR_folder = os.path.join(data_folder, cat, 'ATR')
        merge_20to15(CIHP_folder, ATR_folder, tar_dir)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand",
                        type=str,
                        default='Example')
    parser.add_argument("--cat",
                        type=str,
                        default=None)
    parser.add_argument("--root",
                        type=str,
                        default='parse_filtered_Data')          
    opt = parser.parse_args()

    merge_label(opt)