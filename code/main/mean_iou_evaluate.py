import numpy as np
import scipy.misc
import imageio
import argparse
import os

colormap = [(0,0,0)
        # 0=Background
        ,(255,0,0),(0,0,255),(85,51,0),(255,85,0),(0,119,221)
        # 1=Hat,  2=Hair,    3=Glove, 4=Sunglasses, 5=UpperClothes
        ,(0,0,85),(0,85,85),(51,170,221),(0,255,255),(85,255,170)
        # 6=Dress, 7=Coat, 8=Socks, 9=Pants, 10=Jumpsuits/Neck
        ,(170,255,85),(255,255,0),(255,170,0),(85,85,0)
        ,(128,0,0),(0,85,0)
        # 11=Scarf, 12=Skirt, 13=Face, 14=LeftArm, 15=RightArm
        ,(170,255,85),(255,255,0),(255,170,0)]

c_dict = {}
for idx, c in enumerate(colormap):
    val = c[0]*255**2 + c[1]*255 + c[2]
    c_dict[val] = idx

def read_masks(filepath):
    '''
    Read masks from directory and tranform to categorical
    '''
    # print(c_dict)
    file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    file_list.sort()
    n_masks = len(file_list)
    masks = np.empty((n_masks, 640, 480))

    for i, file in enumerate(file_list):
        mask = imageio.imread(os.path.join(filepath, file)).astype(np.intc)
        # mask = (mask >= 60).astype(int)
        mask = 255**2 * mask[:, :, 0] + 255 * mask[:, :, 1] + mask[:, :, 2]

        for k in c_dict:
            # print(k, (mask==k).sum())
            masks[i, mask==k] = c_dict[k]
    return masks

def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = 0
    valid_class = 0
    for i in range(15):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        if np.isnan(iou):
            continue
        mean_iou += iou
        valid_class += 1
        print('class #%d : %1.5f'%(i, iou))
    print('\nmean_iou: %f\n' % (mean_iou/valid_class))

    return mean_iou/valid_class



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--labels', help='ground truth masks directory', type=str)
    parser.add_argument('-p', '--pred', help='prediction masks directory', type=str)
    args = parser.parse_args()

    pred = read_masks(args.pred)
    labels = read_masks(args.labels)

    mean_iou_score(pred, labels)