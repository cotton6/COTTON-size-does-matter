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

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def pose_valid_check(skeleton):
    # Check if all required points exist
    required_points = [i for i in range(19)]
    if sum(skeleton[required_points, -1] == 0) > 0:
        return False

    # Check if facing forward
    shoulder_width = np.linalg.norm(skeleton[2, :2] - skeleton[5, :2])
    neck_length = np.linalg.norm(skeleton[0, :2] - skeleton[1, :2])
    if neck_length / shoulder_width > 1:
        return False
    
    # Check if hand raised
    min_hand_height = min(skeleton[[3,4,6,7],1])
    if min_hand_height < skeleton[1, 1]:
        return False

    # Check if stand straight
    limbs = [(9, 10), (10, 11), (12, 13), (13, 14)]
    limbs_vec = [skeleton[limb[1], :2] - skeleton[limb[0], :2] for limb in limbs]

    for vec in limbs_vec:
        if not -30 <= np.degrees(angle_between(vec, np.array([0, 1]))) <= 30:
            return False

    return True


    


def openpose_check(opt):
    data_folder = os.path.join(opt.root, opt.brand)
    processed_dir = os.path.join('pose_filtered_Data', opt.brand)
    
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
        os.makedirs(temp_model_folder, exist_ok=True)
        os.makedirs(temp_product_folder, exist_ok=True)
        os.makedirs(temp_pose_folder, exist_ok=True)

        poses = glob.glob(os.path.join(cat, 'pose', '*'))
        for data in poses:
            with open(data) as f:
                pose = json.load(f)
            if len(pose['people']) == 0:
                continue
            skeleton = np.array(pose['people'][0]['pose_keypoints_2d']).reshape(-1, 3)

            valid = pose_valid_check(skeleton)
            if valid:
                shutil.copy2(data.replace('pose', 'model').replace('_keypoints.json', '.jpg'), temp_model_folder)
                if opt.multi_product:
                    products = [
                        data.replace('pose', 'product').replace('model_keypoints.json', 'product_{}.jpg'.format(i)) for
                        i in range(1, 5)]
                    for product in products:
                        if os.path.isfile(product):
                            shutil.copy2(product, temp_product_folder)
                else:
                    product = data.replace('pose', 'product').replace('model_keypoints.json', 'product.jpg')
                    shutil.copy2(product, temp_product_folder)
                shutil.copy2(data, temp_pose_folder)
            else:
                temp_garbage += 1
        print(temp_garbage, len(poses))
        total_garbage += temp_garbage
    print("Total disgard {} images.".format(total_garbage))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand",
                        type=str,
                        default='Example_top')
    parser.add_argument("--cat",
                        type=str,
                        default=None)
    parser.add_argument("--root",
                        type=str,
                        default='raw_Data')  
    parser.add_argument("--multi-product", action='store_true')          
    opt = parser.parse_args()

    openpose_check(opt)
