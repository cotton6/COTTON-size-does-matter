import argparse
import os
import glob
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image
import shutil

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

    ATR label:
        0    background
        1    hat
        2    hair
        3    sunglasses
        4    upper_clothes
        5    skirt
        6    pants
        7    dress
        8    belt
        9    left_shoe
        10    right_shoe
        11    head
        12    left_leg
        13    right_leg
        14    left_arm
        15    right_arm
        16    bag
        17    scarf
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

def merge_ATRto15(ATR_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'vis'), exist_ok=True)

    ATR_parse_imgs = glob.glob(os.path.join(ATR_folder, '*.png'))
    ATR_parse_imgs.sort(key=lambda x:int(os.path.basename(x).split('_')[0]))

    for ATR_parse_name in tqdm(ATR_parse_imgs):
        # ATR_parse_img = cv2.imread(ATR_parse_name, 0)
        # ATR_parse_img = cv2.imread(ATR_parse_name, cv2.IMREAD_UNCHANGED)
        ATR_parse_img = np.array(Image.open(ATR_parse_name))

        bg              = (ATR_parse_img == 0)
        hair            = (ATR_parse_img == 1) + (ATR_parse_img == 2)
        Face            = (ATR_parse_img == 3) + (ATR_parse_img == 11)
        Upper_clothes   = (ATR_parse_img == 4)
        Dress           = (ATR_parse_img == 7)
        Lower_clothes   = (ATR_parse_img == 5) + (ATR_parse_img == 6)
        Left_arm        = (ATR_parse_img == 14)
        Right_arm       = (ATR_parse_img == 15)
        Left_leg        = (ATR_parse_img == 12)
        Right_leg       = (ATR_parse_img == 13)
        Left_shoe       = (ATR_parse_img == 9)
        Right_shoe      = (ATR_parse_img == 10)
        Accessories     = (ATR_parse_img == 8) + (ATR_parse_img == 16) + (ATR_parse_img == 17)
        
        img = np.zeros_like(ATR_parse_img)
        img[bg            ] = 0
        img[hair          ] = 1
        img[Face          ] = 2
        img[Upper_clothes ] = 4
        img[Dress         ] = 6
        img[Lower_clothes ] = 7
        img[Left_arm      ] = 8
        img[Right_arm     ] = 9
        img[Left_leg      ] = 10
        img[Right_leg     ] = 11
        img[Left_shoe     ] = 12
        img[Right_shoe    ] = 13
        img[Accessories   ] = 14

        img = Image.fromarray(img.astype('uint8'),'L')
        img.save('{}/{}.png'.format(output_folder, int(os.path.basename(ATR_parse_name).split('_')[0])))
        img = np.array(label_colours_17)[img]
        img = Image.fromarray(img.astype('uint8'))
        img.save('{}/vis/{}.png'.format(output_folder, int(os.path.basename(ATR_parse_name).split('_')[0])))
        
def move_file(file_folder, output_folder, cls=0, fileType='jpg'):
    os.makedirs(output_folder, exist_ok=True)
    model_imgs = glob.glob(os.path.join(file_folder, '*_{}.{}'.format(cls, fileType)))
    for model_img in tqdm(model_imgs):
        shutil.copyfile(model_img, os.path.join(output_folder, '{}.{}'.format(int(os.path.basename(model_img).split('_')[0]), fileType)))

def main(opt):
    TAR_DIR = os.path.join('./parse_filtered_Data', '{}_{}'.format(opt.brand, opt.cat), opt.cat)
    SRC_DIR = '{}_body'.format(opt.cat)
    os.makedirs(TAR_DIR, exist_ok=True)
    
    # Parsing
    tar_parsing_dir = os.path.join(TAR_DIR, 'parsing_merge')
    src_parsing_dir = os.path.join(SRC_DIR, 'label_maps')
    merge_ATRto15(src_parsing_dir, tar_parsing_dir)

    # Model image
    tar_image_dir = os.path.join(TAR_DIR, 'model')
    src_image_dir = os.path.join(SRC_DIR, 'images')
    move_file(src_image_dir, tar_image_dir, 0, 'jpg')

    # Product image
    tar_image_dir = os.path.join(TAR_DIR, 'product')
    src_image_dir = os.path.join(SRC_DIR, 'images')
    move_file(src_image_dir, tar_image_dir, 1, 'jpg')

    # Product mask
    tar_image_dir = os.path.join(TAR_DIR, 'product-mask')
    src_image_dir = os.path.join(SRC_DIR, 'product-mask')
    move_file(src_image_dir, tar_image_dir, 1, 'png')

    # Train Test Split
    pair_list = []
    with open(os.path.join(SRC_DIR, "train_pairs.txt"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line=line.strip("\n")
            pair_list.append((int(line.split()[0].split('_')[0]), int(line.split()[1].split('_')[0])))
    f = open(os.path.join(SRC_DIR, 'train.txt'),"w")
    for pair in pair_list:
        f.write("{}.jpg {}.jpg\n".format(pair[0], pair[1]))
    f.close()

    pair_list = []
    with open(os.path.join(SRC_DIR, 'test_pairs_paired.txt'), "r") as f:
        lines = f.readlines()
        for line in lines:
            line=line.strip("\n")
            pair_list.append((int(line.split()[0].split('_')[0]), int(line.split()[1].split('_')[0])))
    f = open(os.path.join(SRC_DIR, 'val.txt'),"w")
    for pair in pair_list:
        f.write("{}.jpg {}.jpg\n".format(pair[0], pair[1]))
    f.close()

    pair_list = []
    with open(os.path.join(SRC_DIR, 'test_pairs_unpaired.txt'), "r") as f:
        lines = f.readlines()
        for line in lines:
            line=line.strip("\n")
            pair_list.append((int(line.split()[0].split('_')[0]), int(line.split()[1].split('_')[0])))
    f = open(os.path.join(SRC_DIR, 'test.txt'),"w")
    for pair in pair_list:
        f.write("{}.jpg {}.jpg\n".format(pair[0], pair[1]))
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand",
                        type=str,
                        default='DressCode')
    parser.add_argument("--cat",
                        type=str,
                        default=None)
    parser.add_argument("--root",
                        type=str,
                        default='./') 
    opt = parser.parse_args()
    
    main(opt)