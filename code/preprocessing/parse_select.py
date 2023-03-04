import argparse
import os
import glob
from re import L
import subprocess
from unittest import result
from tqdm import tqdm
import json
import numpy as np
import cv2
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

def parse_valid_check(parse, cloth_type):
    if cloth_type == 'top':
        if (parse == 7).sum() > 0:
            return False
        else:
            return True
    elif cloth_type == 'bottom':
        if (parse == 6).sum() > 0:
            return False
        else:
            return True
    else:
        return True


    


def parse_check(opt):
    data_folder = os.path.join(opt.root, opt.brand)
    processed_dir = os.path.join('parse_filtered_Data', opt.brand)
    cloth_type = opt.brand.split('_')[-1]

    os.makedirs(processed_dir, exist_ok=True)

    cats = glob.glob(os.path.join(data_folder, '*'))

    print(cats)
    total_garbage = 0

    for cat in cats:
        temp_garbage = 0

        cat_folder = os.path.join(processed_dir, os.path.basename(cat))
        os.makedirs(cat_folder, exist_ok=True)

        temp_model_folder = os.path.join(cat_folder, 'model')
        temp_product_folder = os.path.join(cat_folder, 'product')
        temp_pose_folder = os.path.join(cat_folder, 'pose')
        temp_CIHP_folder = os.path.join(cat_folder, 'CIHP')
        temp_CIHP_vis_folder = os.path.join(cat_folder, 'CIHP', 'vis')
        os.makedirs(temp_model_folder, exist_ok=True)
        os.makedirs(temp_product_folder, exist_ok=True)
        os.makedirs(temp_pose_folder, exist_ok=True)
        os.makedirs(temp_CIHP_folder, exist_ok=True)
        os.makedirs(temp_CIHP_vis_folder, exist_ok=True)

        parses = glob.glob(os.path.join(cat, 'CIHP', '*.png'))
        for data in parses:
            parse = cv2.imread(data)

            valid = parse_valid_check(parse, cloth_type)
            if valid:
                shutil.copy2(data.replace('CIHP', 'model').replace('.png', '.jpg'), temp_model_folder)
                shutil.copy2(data.replace('CIHP', 'pose').replace('.png', '_keypoints.json'), temp_pose_folder)

                if opt.multi_product:
                    products = [
                        data.replace('CIHP', 'product').replace('model.png', 'product_{}.jpg'.format(i)) for
                        i in range(1, 5)]
                    for product in products:
                        if os.path.isfile(product):
                            shutil.copy2(product, temp_product_folder)
                else:
                    product = data.replace('CIHP', 'product').replace('model.png', 'product.jpg')
                    shutil.copy2(product, temp_product_folder)

                shutil.copy2(data, temp_CIHP_folder)
                shutil.copy2(os.path.join(os.path.dirname(data), 'vis', os.path.basename(data)), temp_CIHP_vis_folder)
            else:
                temp_garbage += 1
        print(temp_garbage, len(parses))
        total_garbage += temp_garbage
    print("Total disgard {} images.".format(total_garbage))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand",
                        type=str,
                        default='Example')
    parser.add_argument("--cat",
                        type=str,
                        default=None)
    parser.add_argument("--root",
                        type=str,
                        default='pose_filtered_Data')          
    parser.add_argument("--multi-product", action='store_true')          
    opt = parser.parse_args()

    parse_check(opt)
