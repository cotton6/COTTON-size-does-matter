import numpy as np
import scipy.misc
import imageio
import argparse
import os
import cv2

# def read_masks(filepath):
#     '''
#     Read masks from directory and tranform to categorical
#     '''
#     file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
#     file_list.sort()
#     n_masks = len(file_list)
#     masks = np.empty((n_masks, 640, 480))

#     for i, file in enumerate(file_list):
#         mask = imageio.imread(os.path.join(filepath, file))

#         mask = (mask >= 60).astype(int)
#         mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]

#         masks[i, mask == 7] = 1  # (White: 111) Barren land 
#         masks[i, mask == 0] = 0  # (Black: 000) Unknown 

#     return masks

def read_masks(filepath):
    '''
    Read masks from directory and tranform to categorical
    '''
    colors = [
        (0,0,0),        #background
        (61,245,61),    #neckband
        (250,250,55)    #sleeve
    ]
    file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    file_list.sort()
    n_masks = len(file_list)
    masks = np.empty((n_masks, 640, 480))

    for i, file in enumerate(file_list):
        # mask = imageio.imread(os.path.join(filepath, file))
        mask = cv2.imread(os.path.join(filepath, file))[:, :, ::-1]
        for idx, color in enumerate(colors):
            masks[i, np.all(mask == color, -1)] = idx
    return masks

def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = 0
    CLASS_NUM = 3
    for i in range(CLASS_NUM):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / CLASS_NUM
        print('class #%d : %1.5f'%(i, iou))
    print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--labels', help='ground truth masks directory', type=str)
    parser.add_argument('-p', '--pred', help='prediction masks directory', type=str)
    args = parser.parse_args()

    pred = read_masks(args.pred)
    labels = read_masks(args.labels)

    mean_iou_score(pred, labels)