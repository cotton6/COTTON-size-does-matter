"""
    parsing class definition:
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

    real-time joints definition:
    COCO_18
        (0-'nose'	1-'neck' 2-'right_shoulder' 3-'right_elbow' 4-'right_wrist'
        5-'left_shoulder' 6-'left_elbow'	    7-'left_wrist'  8-'right_hip'
        9-'right_knee'	 10-'right_ankle'	11-'left_hip'   12-'left_knee'
        13-'left_ankle'	 14-'right_eye'	    15-'left_eye'   16-'right_ear'
        17-'left_ear' )
    
    BODY_25:
        {0,  "Nose"}, {1,  "Neck"}, {2,  "RShoulder"}, {3,  "RElbow"}, {4,  "RWrist"}, {5,  "LShoulder"},
        {6,  "LElbow"}, {7,  "LWrist"}, {8,  "MidHip"}, {9,  "RHip"}, {10, "RKnee"}, {11, "RAnkle"},
        {12, "LHip"}, {13, "LKnee"}, {14, "LAnkle"}, {15, "REye"}, {16, "LEye"}, {17, "REar"},
        {18, "LEar"}, {19, "LBigToe"}, {20, "LSmallToe"}, {21, "LHeel"}, {22, "RBigToe"},
        {23, "RSmallToe"}, {24, "RHeel"}, {25, "Background"}

"""
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from PIL import ImageDraw

import os
import numpy as np
import json

import random
import torchvision
import cv2
import glob

def perpendicular( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def normalize(a):
    a = np.array(a)
    return a/np.linalg.norm(a)

def get_box(a, b):
    width = 0.2 * np.linalg.norm(a-b)
    u = normalize(perpendicular(a-b)) * width
    a = a + 0.2 * (a-b)
    b = b + 0.2 * (b-a)
    return np.array([a+u, a-u, b-u, b+u])

def get_dist(a, b):
    return np.linalg.norm(a-b)

def get_uintVec(a, b):
    # from a point toward b
    return (b-a) / get_dist(a,b)

def get_Vec(a, b):
    # from a point toward b
    return (b-a)


class TryonDataset(Dataset):
    def __init__(self, config):
        super(TryonDataset,self).__init__()
        self.config = config
        self.mode = config['MODE'] #train or test
        self.dataroot = config['TRAINING_CONFIG']['DATA_DIR'] if self.mode == 'train' else config['VAL_CONFIG']['DATA_DIR']
        self.w = config['TRAINING_CONFIG']['RESOLUTION'][1]
        self.h = config['TRAINING_CONFIG']['RESOLUTION'][0]
        self.tuck = config['TUCK']
        self.parse = config['PARSE']
        self.adj_pose = 'none' if self.mode == 'train' else 'long'
        self.scale = config['VAL_CONFIG']['SCALE']

        self.transform = transforms.Compose([  \
                transforms.ToTensor(),   \
                transforms.Normalize((0.5), (0.5))])

        self.human_names = []
        self.product_names = []

        with open(os.path.join(self.dataroot, "{}.txt".format(self.mode)), "r") as f:
            lines = f.readlines()
            for line in lines:
                line=line.strip("\n")
                self.human_names.append(line.split( )[0])
                self.product_names.append(line.split( )[1])
    
    def __len__(self):
        return len(self.product_names)

    def remove_neckline(self, img, mask, cloth_parse):
        h, w = img.shape[:2]
        kernel_size = max(h, w) // 50
        kernel_size = kernel_size + 1 if kernel_size%2==1 else kernel_size
        kernel_mid = kernel_size // 2
        neckline = cloth_parse == 1
        img[neckline] = 255
        binary_mask = (img.mean(axis=-1) != 255).astype(np.uint8)
        
        kernel = np.zeros((kernel_size,kernel_size), np.uint8)
        kernel[kernel_mid:, kernel_mid] = 1
        binary_mask1 = cv2.erode(binary_mask, kernel, iterations = 1)

        kernel = np.zeros((kernel_size,kernel_size), np.uint8)
        kernel[1:kernel_mid+1, kernel_mid] = 1
        binary_mask2 = cv2.dilate(binary_mask1, kernel, iterations = 1)

        binary_mask2 = binary_mask2.astype(bool)
        img[~binary_mask2] = 255
        mask[~binary_mask2] = 0

        return img, mask


    def cords_to_map(self, cords, img_size, old_size=None, affine_matrix=None, sigma=6):
        old_size = img_size if old_size is None else old_size
        cords[:,[0,1]] = cords[:,[1,0]]
        cords[cords==0] = -1
        cords = cords.astype(float)
        result = np.zeros(img_size + cords.shape[0:1], dtype='float32')
        for i, point in enumerate(cords):
            if point[0] == -1 or point[1] == -1:
                continue
            point[0] = point[0]/old_size[0] * img_size[0]
            point[1] = point[1]/old_size[1] * img_size[1]
            if affine_matrix is not None:
                point_ =np.dot(affine_matrix, np.matrix([point[1], point[0], 1]).reshape(3,1))
                point_0 = int(point_[1])
                point_1 = int(point_[0])
            else:
                point_0 = int(point[0])
                point_1 = int(point[1])
            xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
            result[..., i] = np.exp(-((yy - point_0) ** 2 + (xx - point_1) ** 2) / (2 * sigma ** 2))
        result = result.max(axis=-1)[np.newaxis, ...]
        return result

    def warping_top(self, cloth_rgb, cloth_pose, human_pose, c_parse_array, sleeve_type):
        # Measure human body length
        human_shoulder_length = get_dist(human_pose[2],human_pose[5])
        human_limb_length = 0
        limbs = [(2,3), (3,4), (5,6), (6,7)]
        for limb in limbs:
            if human_pose[limb[0]].sum() == 0 or human_pose[limb[1]].sum() == 0:
                continue
            human_limb_length = max(human_limb_length, get_dist(human_pose[limb[0]],human_pose[limb[1]]))

        if self.adj_pose == 'custom':
            human_torso_length = get_dist(human_pose[1],human_pose[8])
            human_waist_length = get_dist(human_pose[9],human_pose[12])

            # Get cloth shoulder length
            cloth_shoulder_length = get_dist(cloth_pose[2],cloth_pose[5])
            
            # Get body ratio
            limb_ratio = human_limb_length / human_shoulder_length
            torso_ratio = human_torso_length / human_shoulder_length
            waist_ratio = human_waist_length / human_shoulder_length

            # Start adjusting
            cloth_pose_adjusted = cloth_pose.copy()
            cloth_pose_adjusted[3] = cloth_pose_adjusted[2] + get_uintVec(cloth_pose[2], cloth_pose[3]) * cloth_shoulder_length * limb_ratio #right_elbow
            cloth_pose_adjusted[4] = cloth_pose_adjusted[3] + get_uintVec(cloth_pose[3], cloth_pose[4]) * cloth_shoulder_length * limb_ratio #right_wrist

            cloth_pose_adjusted[6] = cloth_pose_adjusted[5] + get_uintVec(cloth_pose[5], cloth_pose[6]) * cloth_shoulder_length * limb_ratio #left_elbow
            cloth_pose_adjusted[7] = cloth_pose_adjusted[6] + get_uintVec(cloth_pose[6], cloth_pose[7]) * cloth_shoulder_length * limb_ratio #left_wrist

            cloth_pose_adjusted[8] = cloth_pose_adjusted[1] + np.array([0, cloth_shoulder_length * torso_ratio])
            cloth_pose_adjusted[9] = cloth_pose_adjusted[8] + np.array([-cloth_shoulder_length * waist_ratio / 2, 0])
            cloth_pose_adjusted[12] = cloth_pose_adjusted[8] + np.array([cloth_shoulder_length * waist_ratio / 2, 0])

            cloth_pose = cloth_pose_adjusted

        elif self.adj_pose == 'long':
            # Get cloth shoulder length
            cloth_shoulder_length = get_dist(cloth_pose[2],cloth_pose[5])
            # limb_ratio = human_limb_length / human_shoulder_length * scale

            # Start adjusting
            scale_factor = 1.414 * self.scale
            cloth_pose_adjusted = cloth_pose.copy()

            cloth_pose_adjusted[3] = cloth_pose_adjusted[2] + get_Vec(cloth_pose[2], cloth_pose[3]) * self.scale #right_elbow
            cloth_pose_adjusted[4] = cloth_pose_adjusted[3] + get_Vec(cloth_pose[3], cloth_pose[4]) * self.scale #right_wrist

            cloth_pose_adjusted[6] = cloth_pose_adjusted[5] + get_Vec(cloth_pose[5], cloth_pose[6]) * self.scale #left_elbow
            cloth_pose_adjusted[7] = cloth_pose_adjusted[6] + get_Vec(cloth_pose[6], cloth_pose[7]) * self.scale #left_wrist


            cloth_pose_adjusted[8, 1] = cloth_pose_adjusted[1, 1] + scale_factor * cloth_shoulder_length
            cloth_pose_adjusted[9, 1] = cloth_pose_adjusted[1, 1] + scale_factor * cloth_shoulder_length
            cloth_pose_adjusted[12,1] = cloth_pose_adjusted[1, 1] + scale_factor * cloth_shoulder_length

            cloth_pose = cloth_pose_adjusted
        else:
            pass

        # arms
        arms = np.zeros_like(cloth_rgb)
        
        # If it is a normal cloth with sleeve
        if sleeve_type == 0:
            cloth_boxes = []
            human_boxes = []
            limbs = [(2,3), (3,4), (5,6), (6,7)]
            for limb in limbs:
                if cloth_pose[limb[0]].sum() == 0 or cloth_pose[limb[1]].sum() == 0 or human_pose[limb[0]].sum() == 0 or human_pose[limb[1]].sum() == 0:
                    continue
                cloth_boxes.append(get_box(cloth_pose[limb[0]], cloth_pose[limb[1]]).astype(np.int32))
                human_boxes.append(get_box(human_pose[limb[0]], human_pose[limb[1]]).astype(np.int32))

            parts = []
            cloth_sleeve = cloth_rgb.copy()
            cloth_sleeve[c_parse_array!=2] = 0
            for cloth_box, human_box in zip(cloth_boxes, human_boxes):
                try:
                    # Get bounding box of region of interest
                    mask = np.zeros(cloth_sleeve.shape[:2]).astype(np.uint8)
                    cv2.drawContours(mask, [cloth_box], -1, (255, 255, 255), -1, cv2.LINE_AA)
                    parts.append(cv2.bitwise_and(cloth_sleeve, cloth_sleeve, mask=mask))

                    # Calculate the homography
                    M = cv2.findHomography(cloth_box, human_box, cv2.RANSAC, 5.0)[0]
                    parts[-1] = cv2.warpPerspective(parts[-1], M, (self.w, self.h), cv2.INTER_LINEAR)
                    img2gray = cv2.cvtColor(parts[-1].copy(),cv2.COLOR_BGR2GRAY)
                    ret, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_BINARY)
                    mask_inv = ~mask
                    arms = cv2.bitwise_and(arms, arms, mask = mask_inv)
                    arms = cv2.add(parts[-1],arms)
                except:
                    continue

            # remove the sleeve part in torso
            cloth_rgb[c_parse_array==2] = 0

        # Torso
        points1 = cloth_pose[[1,2,5,8,9,12]]
        points2 = human_pose[[1,2,5,8,9,12]]
        M = cv2.findHomography(points1, points2, 0)[0]
        torso_warped = cv2.warpPerspective(cloth_rgb,M,(self.w, self.h))

        # Using opening to remove noise
        c_torso_gray = cv2.cvtColor(torso_warped.copy(),cv2.COLOR_BGR2GRAY)
        ret, mask_torso = cv2.threshold(c_torso_gray, 0, 255, cv2.THRESH_BINARY)
        mask_torso = cv2.morphologyEx(mask_torso, cv2.MORPH_OPEN, np.ones((9,9),np.uint8))
        torso_warped = cv2.bitwise_and(torso_warped, torso_warped, mask = mask_torso)

        return arms, torso_warped

    def warping_bottom(self, cloth_rgb, cloth_pose, human_pose, cloth_sub_type):
        # correct waist of cloth skeleton to horizontal
        cloth_pose[9,1] = cloth_pose[8,1]
        cloth_pose[12,1] = cloth_pose[8,1]

        if self.adj_pose == 'custom':
            # Measure human body length
            human_waist_length = get_dist(human_pose[9],human_pose[12])
            human_thigh_length = max(get_dist(human_pose[9],human_pose[10]), get_dist(human_pose[12],human_pose[13]))
            human_calf_length = max(get_dist(human_pose[10],human_pose[11]), get_dist(human_pose[13],human_pose[14]))

            # Get cloth shoulder length
            cloth_waist_length = get_dist(cloth_pose[9],cloth_pose[12])
            
            # Get body ratio
            thigh_ratio = human_thigh_length / human_waist_length
            calf_ratio = human_calf_length / human_waist_length

            # Start adjusting
            cloth_pose_adjusted = cloth_pose.copy()
            cloth_pose_adjusted[10] = cloth_pose_adjusted[9] + get_uintVec(cloth_pose[9], cloth_pose[10]) * cloth_waist_length * thigh_ratio #right_knee
            cloth_pose_adjusted[11] = cloth_pose_adjusted[10] + get_uintVec(cloth_pose[10], cloth_pose[11]) * cloth_waist_length * calf_ratio #right_ankle

            cloth_pose_adjusted[13] = cloth_pose_adjusted[12] + get_uintVec(cloth_pose[12], cloth_pose[13]) * cloth_waist_length * thigh_ratio #left_knee
            cloth_pose_adjusted[14] = cloth_pose_adjusted[13] + get_uintVec(cloth_pose[13], cloth_pose[14]) * cloth_waist_length * calf_ratio #left_ankle

            cloth_pose = cloth_pose_adjusted

        elif self.adj_pose == 'long':
            if cloth_sub_type == 0:
                mask_nonzero = np.transpose(np.nonzero(cloth_rgb)).tolist()
                h_sorted_mask_nonzero = sorted(mask_nonzero, key=lambda s:s[0])
                w_sorted_mask_nonzero = sorted(mask_nonzero, key=lambda s:s[1])
                h_min, h_max = h_sorted_mask_nonzero[0][0], h_sorted_mask_nonzero[-1][0]
                w_min, w_max = w_sorted_mask_nonzero[0][1], w_sorted_mask_nonzero[-1][1]
                cloth_width = w_max - w_min
                cloth_height = h_max - h_min
                cloth_ratio = cloth_height / cloth_width

                if max(cloth_pose[11,1], cloth_pose[14,1]) < h_max:
                    pose_leg_length = max(cloth_pose[11,1], cloth_pose[14,1]) - min(cloth_pose[9,1], cloth_pose[12,1])
                    cloth_leg_length = h_max - min(cloth_pose[9,1], cloth_pose[12,1])
                    rescale_ratio = cloth_leg_length / pose_leg_length * self.scale
                else:
                    rescale_ratio = self.scale

                cloth_pose_adjusted = cloth_pose.copy()
                cloth_pose_adjusted[10] = cloth_pose_adjusted[9] + get_Vec(cloth_pose[9], cloth_pose[10]) * rescale_ratio #right_knee
                cloth_pose_adjusted[11] = cloth_pose_adjusted[10] + get_Vec(cloth_pose[10], cloth_pose[11]) * rescale_ratio #right_ankle

                cloth_pose_adjusted[13] = cloth_pose_adjusted[12] + get_Vec(cloth_pose[12], cloth_pose[13]) * rescale_ratio #left_knee
                cloth_pose_adjusted[14] = cloth_pose_adjusted[13] + get_Vec(cloth_pose[13], cloth_pose[14]) * rescale_ratio #left_ankle
                cloth_pose = cloth_pose_adjusted
            
        else:
            pass

        
        # legs
        cloth_boxes = []
        human_boxes = []
        limbs = [(9,10), (10,11), (12,13), (13,14)]
        for idx, limb in enumerate(limbs):
            if cloth_pose[limb[0]].sum() == 0 or cloth_pose[limb[1]].sum() == 0 or human_pose[limb[0]].sum() == 0 or human_pose[limb[1]].sum() == 0:
                continue
            # Only warped long pants or skirts
            cloth_boxes.append(get_box(cloth_pose[limb[0]], cloth_pose[limb[1]]).astype(np.int32))
            human_boxes.append(get_box(human_pose[limb[0]], human_pose[limb[1]]).astype(np.int32))

        # If it is a trouser
        if cloth_sub_type == 0:
            parts = []
            legs = np.zeros_like(cloth_rgb)
            for cloth_box, human_box in zip(cloth_boxes, human_boxes):
                try:
                    mask = np.zeros(cloth_rgb.shape[:2]).astype(np.uint8)
                    cv2.drawContours(mask, [cloth_box], -1, (255, 255, 255), -1, cv2.LINE_AA)
                    parts.append(cv2.bitwise_and(cloth_rgb, cloth_rgb, mask=mask))
                    
                    img2gray_for_check = cv2.cvtColor(parts[-1].copy(),cv2.COLOR_BGR2GRAY)
                    ret, mask_for_check = cv2.threshold(img2gray_for_check, 0, 255, cv2.THRESH_BINARY)

                    M = cv2.findHomography(cloth_box, human_box, cv2.RANSAC, 5.0)[0]
                    parts[-1] = cv2.warpPerspective(parts[-1], M, (self.w, self.h), cv2.INTER_LINEAR)
                    img2gray = cv2.cvtColor(parts[-1].copy(),cv2.COLOR_BGR2GRAY)
                    ret, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_BINARY)
                    mask_inv = ~mask
                    legs = cv2.bitwise_and(legs, legs, mask = mask_inv)
                    legs = cv2.add(parts[-1],legs)
                except:
                    continue
        # If it is a skirt
        else:
            human_pose_adjusted = human_pose.copy()
            # correct waist to horizontal
            human_pose_adjusted[9, 0] = human_pose_adjusted[8, 0] - get_dist(human_pose[8], human_pose[9])
            human_pose_adjusted[9, 1] = human_pose_adjusted[8, 1]
            human_pose_adjusted[12, 0] = human_pose_adjusted[8, 0] + get_dist(human_pose[8], human_pose[12])
            human_pose_adjusted[12, 1] = human_pose_adjusted[12, 1]
            
            try:
                cloth_box = get_box(cloth_pose[9], cloth_pose[12]).astype(np.int32)
                human_box_adjusted = get_box(human_pose_adjusted[9], human_pose_adjusted[12]).astype(np.int32)
                M = cv2.findHomography(cloth_box, human_box_adjusted, cv2.RANSAC, 5.0)[0]
                legs = cv2.warpPerspective(cloth_rgb,M,(self.w, self.h))
            except:
                legs = np.zeros_like(cloth_rgb)

        # Remove lower part of lower for torso part
        lower_bound = int(cloth_pose[8, 1] + get_dist(human_pose[9], human_pose[12]))
        cloth_rgb[lower_bound:] = 0

        try:
            cloth_box = get_box(cloth_pose[9], cloth_pose[12]).astype(np.int32)
            human_box = get_box(human_pose[9], human_pose[12]).astype(np.int32)
            M = cv2.findHomography(cloth_box, human_box, cv2.RANSAC, 5.0)[0]
            torso_warped = cv2.warpPerspective(cloth_rgb,M,(self.w, self.h))
        except:
            torso_warped = cloth_rgb

        return legs, torso_warped


    def mask_top(self, human, human_parse_img, pose_data, mask_torso, mask_aux):
        human_masked, human_parse_masked = human, human_parse_img

        parse_array = np.array(human_parse_img)
        parse_bg = (parse_array == 0).astype(np.float32)
        parse_upper = (parse_array == 4).astype(np.float32)
        parse_arms = (parse_array == 8) | (parse_array == 9)

        if self.tuck:
            torso_over_neck = cv2.bitwise_and(mask_torso, (((parse_array == 3))*255).astype(np.uint8))
        else:    
            torso_over_neck = cv2.bitwise_and(mask_torso, (((parse_array == 7) | (parse_array == 3) | (parse_array == 14))*255).astype(np.uint8))
        aux_over_arms = cv2.bitwise_and(mask_aux, (parse_arms*255).astype(np.uint8))
        
        pbg = torch.from_numpy(parse_bg)
        pbg.unsqueeze_(0) 

        if (self.mode == 'train' and random.random() > 0.5) or (self.mode != 'train' and self.config['VAL_CONFIG']['MASK_ARM']):
            # Mask arms
            r = 8
            for parse_id, pose_ids in [(8, [2, 5, 6, 7]), (9, [5, 2, 3, 4])]:
                mask_arm = Image.new('L', (self.w, self.h), 'black')
                mask_arm_draw = ImageDraw.Draw(mask_arm)
                i_prev = pose_ids[0]
                for i in pose_ids[1:]:
                    # print("i_prev = {}, i = {}".format(i_prev, i))
                    if (pose_data[i_prev, 0] == 0.0 and pose_data[i_prev, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                        continue
                    mask_arm_draw.line([tuple(pose_data[j]) for j in [i_prev, i]], 'white', width=r*10)
                    pointx, pointy = pose_data[i]
                    radius = r*2 if i == pose_ids[-1] else r*5
                    mask_arm_draw.ellipse((pointx-radius, pointy-radius, pointx+radius, pointy+radius), 'white', 'white')
                    i_prev = i
                parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
                
                human_parse_masked.paste(0, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))
                human_masked.paste((255,255,255), None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))
        
        # mask shoulder
        mask_shoulder = Image.new('L', (self.w, self.h), 'black')
        mask_shoulder_draw = ImageDraw.Draw(mask_shoulder)
        vector = pose_data[5] - pose_data[2]
        if (pose_data[2, 0] == 0.0 and pose_data[2, 1] == 0.0) or (pose_data[5, 0] == 0.0 and pose_data[5, 1] == 0.0):
            pass
        else:
            mask_shoulder_draw.line([tuple(pose_data[2] - 0.5*vector), tuple(pose_data[5] + 0.5*vector)], 'white', width=int(0.8*np.linalg.norm(pose_data[2]-pose_data[5])))
            parse_shoulder = (np.array(mask_shoulder) / 255) * ((parse_array == 3) | (parse_array == 8) | (parse_array == 9)).astype(np.float32)
            human_parse_masked.paste(0, None, Image.fromarray(np.uint8(parse_shoulder * 255), 'L'))
            human_masked.paste((255,255,255), None, Image.fromarray(np.uint8(parse_shoulder * 255), 'L'))

        if self.mode != 'train':
            # Mask arms
            human_parse_masked.paste(0, None, Image.fromarray(aux_over_arms, 'L'))
            human_masked.paste((255,255,255), None, Image.fromarray(aux_over_arms, 'L'))

        # mask torso & neck
        human_parse_masked.paste(0, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
        human_parse_masked.paste(0, None, Image.fromarray(torso_over_neck, 'L'))
        human_masked.paste((255,255,255), None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
        human_masked.paste((255,255,255), None, Image.fromarray(torso_over_neck, 'L'))
        # mask bg
        human_masked.paste((255,255,255), None, Image.fromarray(np.uint8(parse_bg * 255), 'L'))
        return human_masked, human_parse_masked, pbg

    def mask_bottom(self, human, human_parse_img, pose_data, mask_torso, mask_aux):
        human_masked, human_parse_masked = human, human_parse_img

        parse_array = np.array(human_parse_img)
        parse_bg = (parse_array == 0).astype(np.float32)
        parse_lower = (parse_array == 7).astype(np.float32)
        parse_arms = (parse_array == 10) | (parse_array == 11)
        
        aux_over_legs = cv2.bitwise_and(mask_aux, (parse_arms*255).astype(np.uint8))

        pbg = torch.from_numpy(parse_bg)
        pbg.unsqueeze_(0) 

        if self.mode != 'train':
            # Mask legs
            human_parse_masked.paste(0, None, Image.fromarray(aux_over_legs, 'L'))
            human_masked.paste((255,255,255), None, Image.fromarray(aux_over_legs, 'L'))

        # mask lower
        human_parse_masked.paste(0, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))
        human_masked.paste((255,255,255), None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))
        
        # mask bg
        human_masked.paste((255,255,255), None, Image.fromarray(np.uint8(parse_bg * 255), 'L'))
        return human_masked, human_parse_masked, pbg


    def __getitem__(self, index):

        human_name = self.human_names[index]
        c_name = self.product_names[index] 


        human = Image.open(os.path.join(self.dataroot, 'model', 'human_model', human_name))
        human_parse_img = Image.open(os.path.join(self.dataroot, 'model', 'human_parsing', human_name.replace('.jpg', '.png')))

        # Load human pose
        human_pose_name = human_name.replace('.jpg', '.json')
        with open(os.path.join(self.dataroot, 'model', 'human_pose', human_pose_name), "r") as f:
            pose_label = json.load(f)
            human_pose_data = pose_label['people'][0]['pose_keypoints_2d']
            human_pose_data = np.array(human_pose_data)
            human_pose_data = human_pose_data.reshape((-1,3))[:, :2]
        # Load cloth pose
        cloth_pose_name = c_name.replace('.jpg', '_keypoints.json')
        with open(os.path.join(self.dataroot, 'product', 'product_pose', cloth_pose_name), "r") as f:
            cloth_pose_label = json.load(f)
            cloth_pose_data = cloth_pose_label['people'][0]['pose_keypoints_2d']
            cloth_pose_data = np.array(cloth_pose_data)
            cloth_pose_data = cloth_pose_data.reshape((-1,2))[:, :2].astype(np.float32) 

        # Load cloth info
        cloth_info_name = c_name.replace('.jpg', '.json')
        with open(os.path.join(self.dataroot, 'product', 'product_info', cloth_info_name), "r") as f:
            product_info = json.load(f)
            c_type = product_info['product_type']

        c_rgb = Image.open(os.path.join(self.dataroot, 'product', 'product', c_name))
        c_rgb_array = np.array(c_rgb)

        c_mask_img = Image.open(os.path.join(self.dataroot, 'product', 'product-mask', c_name.replace("jpg", "png"))).convert("L")
        c_mask_img_array = np.array(c_mask_img)

        
        if 'bottom' not in c_type:
            c_parse_img = Image.open(os.path.join(self.dataroot, 'product', 'product_parsing', c_name.replace("jpg", "png")))
            c_parse_array = np.array(c_parse_img)
            c_rgb_array, c_mask_img_array = self.remove_neckline(c_rgb_array, c_mask_img_array, c_parse_array)
            # Sleeve type 0 means normal and type 1 represents sleeveless
            cloth_sleeve_type = product_info['sleeve_type'][0]

            c_rgb_array[c_mask_img_array == 0] = 0
            c_aux_warped, c_torso_warped = self.warping_top(c_rgb_array, cloth_pose_data, human_pose_data, c_parse_array, cloth_sleeve_type)
            # Generate mask for mask_top
            c_torso_gray = cv2.cvtColor(c_torso_warped.copy(),cv2.COLOR_BGR2GRAY)
            ret, mask_torso = cv2.threshold(c_torso_gray, 0, 255, cv2.THRESH_BINARY)
            c_aux_gray = cv2.cvtColor(c_aux_warped.copy(),cv2.COLOR_BGR2GRAY)
            ret, mask_aux = cv2.threshold(c_aux_gray, 0, 255, cv2.THRESH_BINARY)
            kernel = np.zeros((7,7),np.uint8)
            kernel[3] = 1
            mask_torso = cv2.dilate(mask_torso, kernel, iterations = 10)
            mask_aux = cv2.dilate(mask_aux, kernel, iterations = 7)
        else:
            c_rgb_array[c_mask_img_array == 0] = 0
            # Load lower subtype
            cloth_info_name = c_name.replace('.jpg', '_info.json')
            cloth_sub_type = product_info['sub_type'][0]
            c_aux_warped, c_torso_warped = self.warping_bottom(c_rgb_array, cloth_pose_data, human_pose_data, cloth_sub_type)
            # Generate mask for mask_bottom
            c_torso_gray = cv2.cvtColor(c_torso_warped.copy(),cv2.COLOR_BGR2GRAY)
            ret, mask_torso = cv2.threshold(c_torso_gray, 0, 255, cv2.THRESH_BINARY)
            c_aux_gray = cv2.cvtColor(c_aux_warped.copy(),cv2.COLOR_BGR2GRAY)
            ret, mask_aux = cv2.threshold(c_aux_gray, 0, 255, cv2.THRESH_BINARY)
            kernel = np.zeros((7,7),np.uint8)
            kernel[3] = 1
            mask_torso = cv2.dilate(mask_torso, kernel, iterations = 5)
            mask_aux = cv2.dilate(mask_aux, kernel, iterations = 5)

        c_aux_warped = self.transform(c_aux_warped)
        c_torso_warped = self.transform(c_torso_warped)
        
        human_pose = self.cords_to_map(human_pose_data.copy(), (self.h, self.w))
        human_pose = torch.Tensor(human_pose)

        human_img = self.transform(human)


        # Convert parsing from 3*w*h -> 15*w*h
        human_parse_label = np.array(human_parse_img)
        human_parse_label = torch.from_numpy(human_parse_label).type(torch.LongTensor)
        human_parse = self.transform(human_parse_img)

        # Parsing masked limb, neck, and clothes
        if c_type == 'bottom':
            human_masked, human_parse_masked_img, bg_mask  = self.mask_bottom(human, human_parse_img, human_pose_data, mask_torso, mask_aux)
        else:
            human_masked, human_parse_masked_img, bg_mask  = self.mask_top(human, human_parse_img, human_pose_data, mask_torso, mask_aux)

        # Convert parsing from 3*w*h -> 15*w*h
        human_parse_masked_label = np.array(human_parse_masked_img)
        human_parse_masked_label = torch.from_numpy(human_parse_masked_label).type(torch.LongTensor)
        human_parse_masked = self.transform(human_parse_masked_img)
        
        human_masked = self.transform(human_masked)

        result = {
            # Name
            "human_name":  human_name,
            "c_name"    :   c_name,

            # Input
            "human_masked": human_masked,
            "human_pose":  human_pose,
            "human_parse_masked": human_parse_masked,
            "c_aux_warped": c_aux_warped,
            "c_torso_warped": c_torso_warped,
            "c_rgb": self.transform(c_rgb),
            # "c_GFLA": c_GFLA,
            
            # Supervision
            "human_img":   human_img,
            "human_parse_label": human_parse_label,
            "human_parse_masked_label": human_parse_masked_label,

        }
        return result
