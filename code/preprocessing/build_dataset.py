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
import glob
import json

def align_top(img, mask, shape=(1024, 768), cloth_width_ratio=0.6):
    h, w = shape
    top_margin_ratio = 0.05
    tar_c_waist_width = int(w * cloth_width_ratio)
    cloth_width_per_row = (mask > 0).sum(axis=1)

    mask_nonzero = np.transpose(np.nonzero(mask)).tolist()
    h_sorted_mask_nonzero = sorted(mask_nonzero, key=lambda s:s[0])
    w_sorted_mask_nonzero = sorted(mask_nonzero, key=lambda s:s[1])
    h_min, h_max = h_sorted_mask_nonzero[0][0], h_sorted_mask_nonzero[-1][0]
    w_min, w_max = w_sorted_mask_nonzero[0][1], w_sorted_mask_nonzero[-1][1]
    cm_width = w_max-w_min
    cm_height = h_max-h_min

    cm_waist_width = cloth_width_per_row[(h_min+h_max)//2]

    scaling = tar_c_waist_width / cm_width
    tar_cm_width = int(round(cm_width*scaling))
    tar_cm_height = int(round(cm_height*scaling))


    # NOTE: crop and align C
    img = img[h_min:h_max, w_min:w_max]
    mask = mask[h_min:h_max, w_min:w_max]
    img = cv2.resize(img, (tar_cm_width, tar_cm_height), cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (tar_cm_width, tar_cm_height), cv2.INTER_NEAREST)

    top_margin = int(h * top_margin_ratio)
    left_margin = int((w - tar_cm_width) /2)
    
    new_img = np.ones((h, w, 3)).astype(np.uint8) * 255
    new_img[top_margin:top_margin+tar_cm_height, left_margin:left_margin+tar_cm_width] = img[:min(top_margin+tar_cm_height, h)-top_margin]

    new_mask = np.zeros((h, w)).astype(np.uint8)
    new_mask[top_margin:top_margin+tar_cm_height, left_margin:left_margin+tar_cm_width] = mask[:min(top_margin+tar_cm_height, h)-top_margin]

    return new_img, new_mask

def align_bottom(img, mask, shape=(1024, 768), cloth_width_ratio=0.3):
    h, w = shape
    top_margin_ratio = 0.05
    tar_c_waist_width = int(w * cloth_width_ratio)
    cloth_width_per_row = (mask > 0).sum(axis=1)

    mask_nonzero = np.transpose(np.nonzero(mask)).tolist()
    h_sorted_mask_nonzero = sorted(mask_nonzero, key=lambda s:s[0])
    w_sorted_mask_nonzero = sorted(mask_nonzero, key=lambda s:s[1])
    h_min, h_max = h_sorted_mask_nonzero[0][0], h_sorted_mask_nonzero[-1][0]
    w_min, w_max = w_sorted_mask_nonzero[0][1], w_sorted_mask_nonzero[-1][1]
    cm_width = w_max-w_min
    cm_height = h_max-h_min

    # cm_waist_width = cloth_width_per_row[(h_min+h_max)//2]

    scaling = tar_c_waist_width / cm_width
    tar_cm_width = int(round(cm_width*scaling))
    tar_cm_height = int(round(cm_height*scaling))


    # NOTE: crop and align C
    img = img[h_min:h_max, w_min:w_max]
    mask = mask[h_min:h_max, w_min:w_max]
    img = cv2.resize(img, (tar_cm_width, tar_cm_height), cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (tar_cm_width, tar_cm_height), cv2.INTER_NEAREST)

    top_margin = int(h * top_margin_ratio)
    left_margin = int((w - tar_cm_width) /2)
    
    new_img = np.ones((h, w, 3)).astype(np.uint8) * 255
    new_img[top_margin:top_margin+tar_cm_height, left_margin:left_margin+tar_cm_width] = img[:min(top_margin+tar_cm_height, h)-top_margin]

    new_mask = np.zeros((h, w)).astype(np.uint8)
    new_mask[top_margin:top_margin+tar_cm_height, left_margin:left_margin+tar_cm_width] = mask[:min(top_margin+tar_cm_height, h)-top_margin]

    return new_img, new_mask


def alignHuman(data_folder, output_folder, shape=(640, 480)):
    h, w = shape
    
    cat_list = [os.path.basename(name) for name in glob.glob(os.path.join(data_folder, '*'))]
    human_images        = []
    human_parses        = []
    human_parses_vis    = []
    human_poses         = []

    for cat in cat_list:
        print(cat)
        human_images        += sorted(glob.glob(os.path.join(data_folder, cat, 'model'              , '*.jpg')) , key=lambda x:int(os.path.basename(x).split('.')[0].split('_')[0]))
        human_parses        += sorted(glob.glob(os.path.join(data_folder, cat, 'parsing_merge'      , '*.png')) , key=lambda x:int(os.path.basename(x).split('.')[0].split('_')[0]))
        human_parses_vis    += sorted(glob.glob(os.path.join(data_folder, cat, 'parsing_merge/vis'  , '*.png')) , key=lambda x:int(os.path.basename(x).split('.')[0].split('_')[0]))
        human_poses         += sorted(glob.glob(os.path.join(data_folder, cat, 'pose'               , '*.json')), key=lambda x:int(os.path.basename(x).split('.')[0].split('_')[0]))

    tar_model_folder = os.path.join(output_folder, "human_model")
    tar_parse_folder = os.path.join(output_folder, "human_parsing")
    tar_parse_vis_folder = os.path.join(output_folder, "human_parsing", 'vis')
    tar_pose_folder  = os.path.join(output_folder, "human_pose")

    os.makedirs(tar_model_folder, exist_ok=True)
    os.makedirs(tar_parse_folder, exist_ok=True)
    os.makedirs(tar_parse_vis_folder , exist_ok=True)
    os.makedirs(tar_pose_folder , exist_ok=True)

    human_height_ratio = 0.95
    for idx, (model, parse, parse_vis, pose) in enumerate(tqdm(zip(human_images, human_parses, human_parses_vis, human_poses), total=len(human_images))):
        align_height = round(h * human_height_ratio)

        model_img = cv2.imread(model)
        parse_img = cv2.imread(parse, 0)
        parse_vis_img = cv2.imread(parse_vis)
        parse_bg = (parse_img == 0)
        model_img[parse_bg] = 255
        parse_roi = 1-parse_bg

        mask_nonzero = np.transpose(np.nonzero(parse_roi)).tolist()

        h_sorted_mask_nonzero = sorted(mask_nonzero, key=lambda s:s[0])
        w_sorted_mask_nonzero = sorted(mask_nonzero, key=lambda s:s[1])
        h_min, h_max = h_sorted_mask_nonzero[0][0], h_sorted_mask_nonzero[-1][0]
        w_min, w_max = w_sorted_mask_nonzero[0][1], w_sorted_mask_nonzero[-1][1]

        scale_ratio = align_height/(h_max-h_min) if int(round((w_max-w_min) * align_height/(h_max-h_min))) < 768 else 1
        align_width = int(round((w_max-w_min) * scale_ratio))

        h_top = round((h-align_height)/2)
        w_left = round((w-align_width)/2)


        model_img = model_img[h_min:h_max, w_min:w_max]
        parse_img = parse_img[h_min:h_max, w_min:w_max]
        parse_vis_img = parse_vis_img[h_min:h_max, w_min:w_max]

        if scale_ratio < 1:
            model_img = cv2.resize(model_img, (align_width, align_height), interpolation=cv2.INTER_AREA)
        else:
            model_img = cv2.resize(model_img, (align_width, align_height), interpolation=cv2.INTER_CUBIC)
        parse_img = cv2.resize(parse_img, (align_width, align_height), interpolation=cv2.INTER_NEAREST)
        parse_vis_img = cv2.resize(parse_vis_img, (align_width, align_height), interpolation=cv2.INTER_NEAREST)

        
        aligned_model_img = (np.ones((h, w, 3)) * 255).astype(np.uint8)
        aligned_model_img[h_top:h_top+align_height, w_left:w_left+align_width] = model_img

        aligned_parse_img = np.zeros((h, w)).astype(np.uint8)
        aligned_parse_img[h_top:h_top+align_height, w_left:w_left+align_width] = parse_img

        aligned_parse_vis_img = np.zeros((h, w, 3)).astype(np.uint8)
        aligned_parse_vis_img[h_top:h_top+align_height, w_left:w_left+align_width] = parse_vis_img
        
        if not opt.keep_order:
            cv2.imwrite(os.path.join(tar_model_folder,      '{}.jpg'.format(idx)), aligned_model_img)
            cv2.imwrite(os.path.join(tar_parse_folder,      '{}.png'.format(idx)), aligned_parse_img)
            cv2.imwrite(os.path.join(tar_parse_vis_folder,  '{}.png'.format(idx)), aligned_parse_vis_img)
        else:
            cv2.imwrite(os.path.join(tar_model_folder,      '{}.jpg'.format(int(os.path.basename(model).split('.')[0].split('_')[0]))), aligned_model_img)
            cv2.imwrite(os.path.join(tar_parse_folder,      '{}.png'.format(int(os.path.basename(model).split('.')[0].split('_')[0]))), aligned_parse_img)
            cv2.imwrite(os.path.join(tar_parse_vis_folder,  '{}.png'.format(int(os.path.basename(model).split('.')[0].split('_')[0]))), aligned_parse_vis_img)

        with open(pose, "r") as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints_2d']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1,3))

        exist_idx = pose_data[:, 2] != 0
        pose_data[exist_idx, 0] = (pose_data[exist_idx, 0] - w_min) * scale_ratio + w_left
        pose_data[exist_idx, 1] = (pose_data[exist_idx, 1] - h_min) * scale_ratio + h_top
        pose_format = {
            "version": 1.3,
            "people": [
                        {
                            "person_id": [-1],
                            "pose_keypoints_2d": pose_data.reshape(-1).tolist(),
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
        if not opt.keep_order:
            with open(os.path.join(tar_pose_folder, '{}.json'.format(idx)), 'w') as f:
                json.dump(pose_format, f)
        else:
            with open(os.path.join(tar_pose_folder, '{}.json'.format(int(os.path.basename(model).split('.')[0].split('_')[0]))), 'w') as f:
                json.dump(pose_format, f)

def alignProduct(data_folder, output_folder, shape=(640, 480), cloth_type='top'):
    h, w = shape
    
    cat_list = [os.path.basename(name) for name in glob.glob(os.path.join(data_folder, '*'))]
    product_images      = []
    product_infos       = []
    product_masks       = []

    for cat in cat_list:
        product_images      += sorted(glob.glob(os.path.join(data_folder, cat, 'product'            , '*.jpg')) , key=lambda x:int(os.path.basename(x).split('.')[0].split('_')[0]))
        product_infos       += sorted(glob.glob(os.path.join(data_folder, cat, 'product_info'       , '*.json')), key=lambda x:int(os.path.basename(x).split('.')[0].split('_')[0]))
        product_masks       += sorted(glob.glob(os.path.join(data_folder, cat, 'product-mask'       , '*.png')) , key=lambda x:int(os.path.basename(x).split('.')[0].split('_')[0]))
        

    tar_product_folder = os.path.join(output_folder, "product")
    tar_prodcut_info_folder = os.path.join(output_folder, "product_info")
    tar_mask_folder = os.path.join(output_folder, "product-mask")

    os.makedirs(tar_product_folder, exist_ok=True)
    os.makedirs(tar_prodcut_info_folder, exist_ok=True)
    os.makedirs(tar_mask_folder , exist_ok=True)

    for idx, (product, mask, info) in enumerate(tqdm(zip(product_images, product_masks, product_infos), total=len(product_images))):
        product_img = cv2.imread(product)
        mask_img = cv2.imread(mask, 0)
        product_img[mask_img == 0] = 255
        
        with open(info, "r") as f:
            product_info = json.load(f)

        if cloth_type == 'top' or cloth_type == 'upper':
            no_sleeve = product_info['sleeve_type'][0]

            if no_sleeve:
                product_img, mask_img = align_top(product_img, mask_img, shape=shape, cloth_width_ratio=0.4)
            else:
                product_img, mask_img  = align_top(product_img, mask_img, shape=shape)
            product_info['product_type'] = 'top'

        elif cloth_type == 'bottom' or cloth_type == 'lower':
            product_img, mask_img = align_bottom(product_img, mask_img, shape=shape)
            product_info['product_type'] = 'bottom'

        

        if not opt.keep_order:
            cv2.imwrite(os.path.join(tar_product_folder, '{}.jpg'.format(idx)), product_img)
            cv2.imwrite(os.path.join(tar_mask_folder, '{}.png'.format(idx)), mask_img)
            with open(os.path.join(tar_prodcut_info_folder, '{}.json'.format(idx)), 'w') as f:
                json.dump(product_info, f)
        else:
            cv2.imwrite(os.path.join(tar_product_folder, '{}.jpg'.format(int(os.path.basename(product).split('.')[0].split('_')[0]))), product_img)
            cv2.imwrite(os.path.join(tar_mask_folder, '{}.png'.format(int(os.path.basename(product).split('.')[0].split('_')[0]))), mask_img)
            with open(os.path.join(tar_prodcut_info_folder, '{}.json'.format(int(os.path.basename(product).split('.')[0].split('_')[0]))), 'w') as f:
                json.dump(product_info, f)

def train_val_split(data_folder, output_folder):

    train_set = []
    val_set = []

    train_file = os.path.join(output_folder, 'train.txt')
    val_file = os.path.join(output_folder, 'val.txt')

    cat_list = [os.path.basename(name) for name in glob.glob(os.path.join(data_folder, '*'))]
    acc_num = 0
    for cat in cat_list:
        data_num = len(glob.glob(os.path.join(data_folder, cat, 'product', '*.jpg')))
        for i in range(data_num):
            if i < int(0.8*data_num):
                train_set.append(acc_num+i)
            else:
                val_set.append(acc_num+i)
        acc_num += data_num
    
    f = open(train_file,"w")
    for idx in train_set:
        f.write("{}.jpg {}.jpg\n".format(idx, idx))
    f.close()

    f = open(val_file,"w")
    for idx in val_set:
        f.write("{}.jpg {}.jpg\n".format(idx, idx))
    f.close()


def build_dataset(opt):
    data_folder = os.path.join(opt.root, opt.brand)
    new_dataset = 'Training_Dataset/{}x{}'.format(opt.h, opt.w)
    os.makedirs(new_dataset, exist_ok=True)
    output_folder = os.path.join(new_dataset, opt.brand)
    model_folder = os.path.join(output_folder, 'model')
    product_folder = os.path.join(output_folder, 'product')

    alignHuman(data_folder, model_folder, shape=(opt.h, opt.w))
    alignProduct(data_folder, product_folder, shape=(opt.h, opt.w), cloth_type=opt.brand.split('_')[-1])
    train_val_split(data_folder, output_folder)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand",
                        type=str,
                        default='Example_top')
    parser.add_argument("--cat",
                        type=str,
                        default=None)
    parser.add_argument("--root",
                        type=str,
                        default='parse_filtered_Data')     
    parser.add_argument("--h", type=int, default=1024)          
    parser.add_argument("--w", type=int, default=768)          
    parser.add_argument("--keep_order", type=bool, default=False)          
    opt = parser.parse_args()
    build_dataset(opt)

