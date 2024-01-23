import os
import glob
import json
import numpy as np
import shutil
import cv2
from main import draw_bodypose

# BODY_PART = 'upper'

# if BODY_PART == 'upper':
#     useful = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12]
#     limbSeq = [[1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9], [8, 12]]
# elif BODY_PART == 'lower':
#     useful = [8, 9, 10, 11, 12, 13, 14]
#     limbSeq = [[8, 9], [9, 10], [10, 11], [8, 12], [12, 13], [13, 14]]

# useful = [8, 9, 10, 11, 12, 13, 14]

target_folder = 'lower_train_pose_filtered'
os.makedirs(target_folder, exist_ok=True)

# potentials = glob.glob('lower_train_pose/*.json')
potentials = glob.glob('upperL_partial_json_afterflip/*.json')
print(len(potentials))
for p in potentials:
    with open(p) as f:
        label = json.load(f)
    try:
        label = np.array(label['people'][0]['pose_keypoints_2d']).reshape(25, 3)
    except:
        continue
    sym_pairs = [(9, 12), (10, 13), (11, 14)]
    # if np.all(label[useful, -1] > 0):
    if True:
        img_name = p.replace('upperL_partial_json_afterflip', 'upperL').replace('_keypoints.json', '.jpg')
        img = cv2.imread(img_name)
        for pair in sym_pairs:
            l_idx, r_idx = pair
            if label[l_idx, 0] > label[r_idx, 0]:
                label[[l_idx, r_idx]] = label[[r_idx, l_idx]]
        skeleton = draw_bodypose(img, label, body_part='upper')
        cv2.imshow('skeleton', skeleton)
        cv2.waitKey(0)
        # new_name = os.path.join(target_folder, os.path.basename(p))
        # shutil.copyfile(p, new_name)
        # shutil.copyfile(p.replace('pose', 'skeleton').replace('keypoints.json', 'rendered.png'), new_name.replace('pose', 'skeleton').replace('keypoints.json', 'rendered.png'))