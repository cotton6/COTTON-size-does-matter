import numpy as np
import random
import cv2

sigma_inp = 7
n = sigma_inp * 6 + 1
g_inp = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        g_inp[i, j] = np.exp(-((i - n / 2) ** 2 + (j - n / 2) ** 2) / (2. * sigma_inp * sigma_inp))


# IMAGE NET CONSTANTS
MEAN = [0.485, 0.456, 0.406],
STD = [0.229, 0.224, 0.225]


# https://github.com/xingyizhou/pytorch-pose-hg-3d/blob/master/src/utils/img.py
def Gaussian(sigma):
    if sigma == 7:
        return np.array([0.0529, 0.1197, 0.1954, 0.2301, 0.1954, 0.1197, 0.0529,
                         0.1197, 0.2709, 0.4421, 0.5205, 0.4421, 0.2709, 0.1197,
                         0.1954, 0.4421, 0.7214, 0.8494, 0.7214, 0.4421, 0.1954,
                         0.2301, 0.5205, 0.8494, 1.0000, 0.8494, 0.5205, 0.2301,
                         0.1954, 0.4421, 0.7214, 0.8494, 0.7214, 0.4421, 0.1954,
                         0.1197, 0.2709, 0.4421, 0.5205, 0.4421, 0.2709, 0.1197,
                         0.0529, 0.1197, 0.1954, 0.2301, 0.1954, 0.1197, 0.0529]).reshape(7, 7)
    elif sigma == n:
        return g_inp
    else:
        raise Exception('Gaussian {} Not Implement'.format(sigma))


# https://github.com/xingyizhou/pytorch-pose-hg-3d/blob/master/src/utils/img.py
def DrawGaussian(img, pt, sigma):
    tmpSize = int(np.math.ceil(3 * sigma))
    ul = [int(np.math.floor(pt[0] - tmpSize)), int(np.math.floor(pt[1] - tmpSize))]
    br = [int(np.math.floor(pt[0] + tmpSize)), int(np.math.floor(pt[1] + tmpSize))]

    if ul[0] > img.shape[1] or ul[1] > img.shape[0] or br[0] < 1 or br[1] < 1:
        return img

    size = 2 * tmpSize + 1
    g = Gaussian(size)

    g_x = [max(0, -ul[0]), min(br[0], img.shape[1]) - max(0, ul[0]) + max(0, -ul[0])]
    g_y = [max(0, -ul[1]), min(br[1], img.shape[0]) - max(0, ul[1]) + max(0, -ul[1])]

    img_x = [max(0, ul[0]), min(br[0], img.shape[1])]
    img_y = [max(0, ul[1]), min(br[1], img.shape[0])]

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img


def flip(img, ignore_mask, keypoints, flip_indices):
    width = img.shape[1]
    img = cv2.flip(img, 1)
    ignore_mask = cv2.flip(ignore_mask, 1)
    keypoints[:,:,0] = width - 1 - keypoints[:,:,0]
    for flip_id in flip_indices:
        temp =  keypoints[:, flip_id[0], :].copy()
        keypoints[:, flip_id[0], :] = keypoints[:, flip_id[1], :]
        keypoints[:, flip_id[1], :] = temp
    return img, ignore_mask, keypoints


def resize(img, ignore_mask, keypoints, imgSize):
    width, height = img.shape[0], img.shape[1]
    img = cv2.resize(img, (imgSize, imgSize))
    ignore_mask = cv2.resize( ignore_mask, (imgSize, imgSize))
    keypoints[:, :, 0] = keypoints[:, :, 0] * imgSize / height
    keypoints[:, :, 1] = keypoints[:, :, 1] * imgSize / width
    return img, ignore_mask, keypoints


def resize_hm(heatmap, hm_size):
    if np.isscalar(hm_size):
        hm_size = (hm_size, hm_size)
    heatmap = cv2.resize(heatmap.transpose(1, 2, 0), hm_size,interpolation=cv2.INTER_CUBIC)
    return heatmap.transpose(2, 0, 1)


def resize_hm_paf(heatmap, paf, hm_size):
    heatmap = resize_hm(heatmap, hm_size)
    paf = paf.transpose(2,3,0,1)
    paf = paf.reshape(paf.shape[0], paf.shape[1], paf.shape[2] * paf.shape[3])
    paf = cv2.resize(paf, hm_size,interpolation=cv2.INTER_CUBIC)
    paf = paf.transpose(2, 0, 1)
    return heatmap, paf


def color_augment(img, ignore_mask, keypoints, color_aug):
    for channel in range(img.shape[2]):
        img[:, :, channel] = np.clip(img[:, :, channel] * (np.random.random()*color_aug*2 + 1 - color_aug) , 0, 1)
    return img, ignore_mask, keypoints



def normalize(img):
    img = img[:, :, ::-1]
    img = (img - MEAN) / STD
    img = img.transpose(2, 0, 1)
    return img


def denormalize(img):
    img = img.transpose(1, 2, 0)
    img = img * STD + MEAN
    img = img[:, :, ::-1]
    return img


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

# https://github.com/JimmySuen/integral-human-pose/ - Integral pose estimation,
# This paper has very good results on single person pose
def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_augment(img, ignore_mask, keypoints, rot_angle, scale_aug_factor):
    keypoints_useful = keypoints[keypoints[:,:,2]>0]
    if keypoints_useful.ndim == 2:
        keypoints_useful = keypoints_useful.reshape(1, keypoints_useful.shape[0], keypoints_useful.shape[1])
    left_lim = keypoints_useful[:,:,0].min() - 32
    right_lim = keypoints_useful[:,:,0].max() + 32
    top_lim = keypoints_useful[:,:,1].min() - 32
    bot_lim = keypoints_useful[:,:,1].max() + 32
    c_y = img.shape[0]/2
    c_x = img.shape[1]/2
    scale_min = max(max(right_lim-c_x, c_x-left_lim)/c_x, max(c_y - top_lim, bot_lim - c_y)/c_y, 1 - scale_aug_factor)
    scale_max = min(2 - scale_min, 1 + scale_aug_factor)
    scale = (1 + np.clip(np.random.randn(), -1, 1))*(scale_max - scale_min)*0.5 + scale_min
    trans = gen_trans_from_patch_cv(c_x, c_y, img.shape[1], img.shape[0], img.shape[1], img.shape[0], scale, rot_angle)
    img = cv2.warpAffine(img, trans, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    ignore_mask = cv2.warpAffine(ignore_mask, trans, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    affine_trans_keypoints = np.matmul(trans[:,:2], keypoints[:,:,:2].copy().transpose(0,2,1)).transpose(0,2,1)
    affine_trans_keypoints = affine_trans_keypoints + trans[:,2]
    keypoints[:,:,:2] = affine_trans_keypoints
    return img, ignore_mask, keypoints
