import os
import pickle
import numpy as np
from .process_utils import DrawGaussian

# COCO typical constants
MIN_KEYPOINTS = 5
MIN_AREA = 32 * 32

# Non traditional body parts
# BODY_PARTS = [
#     (0,1),   # nose - left eye
#     (0,2),   # nose - right eye
#     (1,3),   # left eye - left ear
#     (2,4),   # right eye - right ear
#     (17,5),  # neck - left shoulder
#     (17,6),  # neck - right shoulder
#     (5,7),   # left shoulder - left elbow
#     (6,8),   # right shoulder - right elbow
#     (7,9),   # left elbow - left hand
#     (8,10),  # right elbow - right hand
#     (17,11), # neck - left waist
#     (17,12), # neck - right waist
#     (11,13), # left waist - left knee
#     (12,14), # right waise - right knee
#     (13,15), # left knee - left foot
#     (14,16), # right knee - right foot
#     (0,17)   # nose - neck
# ]

# BODY_PARTS = [
#     (0,1),   # nose - chest
#     (1,2),   # chest - right shoulder
#     (2,3),   # right shoulder - right elbow
#     (3,4),   # right elbow - right waist
#     (1,5),   # chest - left shoulder
#     (5,6),   # left shoulder - left elbow
#     (6,7),   # left elbow - left waist
#     (1,8),   # chest - mid hip
#     (8,9),   # mid hip - right hip
#     (9,10),   # right hip - right knee
#     (10,11),   # right knee - right foot
#     (8,12),   # mid hip - left hip
#     (12,13),   # left hip - left knee
#     (13,14),   # left knee - left foot
#     (0,15),   # nose - right eye
#     (0,16),   # nose - left eye
#     (15,17),   # right eye - right ear
#     (16,18),   # left eye - left ear
# ]

BODY_PARTS_UPPERL = [
    (1,2),   # chest - right shoulder
    (2,3),   # right shoulder - right elbow
    (3,4),   # right elbow - right waist
    (1,5),   # chest - left shoulder
    (5,6),   # left shoulder - left elbow
    (6,7),   # left elbow - left waist
    (1,8),   # chest - mid hip
    (8,9),   # mid hip - right hip
    (8,12),   # mid hip - left hip
]

FLIP_INDICES = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16)]
FLIP_INDICES_PAF = [(0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13), (14,15)]


def check_annot(annot):
    return annot['num_keypoints'] >= MIN_KEYPOINTS and annot['area'] > MIN_AREA and not annot['iscrowd'] == 1


def get_heatmap(img, keypoints, sigma):
    n_joints = keypoints.shape[0]
    out_map = np.zeros((n_joints + 1, img.shape[0], img.shape[1]))
    for i in range(keypoints.shape[0]):
        keypoint = keypoints[i]
        # Ignore unannotated keypoints
        if keypoint[2] > 0:
            out_map[i] = np.maximum(out_map[i], DrawGaussian(out_map[i], keypoint[0:2], sigma=sigma))
    out_map[n_joints] = 1 - np.sum(out_map[0:n_joints], axis=0) # Last heatmap is background
    return out_map

def get_paf(img, keypoints, sigma_paf, variable_width, limbSeq):
    out_pafs = np.zeros((len(limbSeq), 2, img.shape[0], img.shape[1]))
    n_person_part = np.zeros((len(limbSeq),img.shape[0],img.shape[1]))
    for i in range(len(limbSeq)):
        part = limbSeq[i]
        keypoint_1 = keypoints[part[0], :2]
        keypoint_2 = keypoints[part[1], :2]
        if keypoints[part[0], 2] > 0 and keypoints[part[1], 2] > 0:
            part_line_segment = keypoint_2 - keypoint_1
            # Notation from paper
            l = np.linalg.norm(part_line_segment)
            if l>1e-2:
                sigma = sigma_paf
                if variable_width:
                    sigma = sigma_paf *  l * 0.025
                v = part_line_segment/l
                v_per = v[1], -v[0]
                x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
                dist_along_part = v[0] * (x - keypoint_1[0]) + v[1] * (y - keypoint_1[1])
                dist_per_part = np.abs(v_per[0] * (x - keypoint_1[0]) + v_per[1] * (y - keypoint_1[1]))
                mask1 = dist_along_part >= 0
                mask2 = dist_along_part <= l
                mask3 = dist_per_part <= sigma
                mask = mask1 & mask2 & mask3
                out_pafs[i, 0] = out_pafs[i, 0] + mask.astype('float32') * v[0]
                out_pafs[i, 1] = out_pafs[i, 1] + mask.astype('float32') * v[1]
                n_person_part[i] += mask.astype('float32')
    n_person_part = n_person_part.reshape(out_pafs.shape[0], 1, img.shape[0], img.shape[1])
    out_pafs = out_pafs/(n_person_part + 1e-8)
    return out_pafs


def add_neck(keypoints):
    right_shoulder = keypoints[6, :]
    left_shoulder = keypoints[5, :]
    neck = np.zeros(3)
    if right_shoulder[2] > 0 and left_shoulder[2] > 0:
        neck = (right_shoulder + left_shoulder) / 2
        neck[2] = 2

    neck = neck.reshape(1, len(neck))
    neck = np.round(neck)
    keypoints = np.vstack((keypoints,neck))

    return keypoints


def get_keypoints(coco, img, annots):
    keypoints = []
    for annot in annots:
        person_keypoints = np.array(annot['keypoints']).reshape(-1, 3)
        person_keypoints = add_neck(person_keypoints)
        keypoints.append(person_keypoints)
    return np.array(keypoints)


def get_ignore_mask(coco, img, annots):
    mask_union = np.zeros((img.shape[0], img.shape[1]), 'bool')
    masks = []
    for annot in annots:
        mask = coco.annToMask(annot).astype('bool')
        masks.append(mask)
        if check_annot(annot):
            mask_union = mask_union | mask
    ignore_mask = np.zeros((img.shape[0], img.shape[1]), 'bool')
    for i in range(len(annots)):
        annot = annots[i]
        mask = masks[i]
        if not check_annot(annot):
            ignore_mask = ignore_mask | (mask & ~mask_union)

    return ignore_mask.astype('uint8')


def clean_annot(coco, data_path, split):
    ids_path = os.path.join(data_path, split + '_ids.pkl')
    if os.path.exists(ids_path):
        print('Loading filtered annotations for {} from {}'.format(split,ids_path))
        with open(ids_path, 'rb') as f:
            return pickle.load(f)
    else:
        print('Filtering annotations for {}'.format(split))
        person_ids = coco.getCatIds(catNms=['person'])
        indices_tmp = sorted(coco.getImgIds(catIds=person_ids))
        indices = np.zeros(len(indices_tmp))
        valid_count = 0
        for i in range(len(indices_tmp)):
            anno_ids = coco.getAnnIds(indices_tmp[i])
            annots = coco.loadAnns(anno_ids)
            # Coco standard constants
            annots = list(filter(lambda annot: check_annot(annot), annots))
            if len(annots) > 0:
                indices[valid_count] = indices_tmp[i]
                valid_count += 1
            if i%100==0:
                print(i)
        indices = indices[:valid_count]
        print('Saving filtered annotations for {} to {}'.format(split, ids_path))
        with open(ids_path, 'wb') as f:
            pickle.dump(indices, f)
        return indices
