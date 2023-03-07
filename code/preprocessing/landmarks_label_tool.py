import cv2
import numpy as np
import glob
import os 
import json
import math
import argparse

def get_dist(a, b):
    return np.linalg.norm(a-b)

def get_limb(a, b, scale):
    # from a point toward b
    vec = (b-a) * scale / get_dist(a,b)
    return a, a+vec

def get_midpoint(a, b):
    return (a+b) / 2

def get_uintVec(a, b):
    # from a point toward b
    return (b-a) / get_dist(a,b)


def draw_bodypose(canvas, keypoints, body_part='top'):
    stickwidth = 4

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for i in range(len(colors)):
        x, y = keypoints[i][0:2]
        cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
    for i in range(len(limbSeq)):
        cur_canvas = canvas.copy()
        Y = keypoints[np.array(limbSeq[i]), 0]
        X = keypoints[np.array(limbSeq[i]), 1]
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    return canvas


def main(opt):

    global useful
    global limbSeq
    global index

    if opt.body == 'top':
        useful = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12]
        limbSeq = [[1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9], [8, 12]]
    elif opt.body == 'bottom':
        useful = [8, 9, 10, 11, 12, 13, 14]
        limbSeq = [[8, 9], [9, 10], [10, 11], [8, 12], [12, 13], [13, 14]]

    images = glob.glob(os.path.join(opt.input_dir, '*.jpg'))
    output_dir = os.path.join(os.path.dirname(opt.input_dir), 'pose_labeled')
    os.makedirs(output_dir, exist_ok=True)

    for image in images:
        imgName = os.path.basename(image).split('.')[0]
        if os.path.isfile(os.path.join(output_dir, '{}_keypoints.json'.format(imgName))):
            continue
        image = cv2.imread(image)
        

        # Create a function based on a CV2 Event (Left button click)
        drawing = False  # True if mouse is pressed
        index = 0

        click_points = np.ones((4, 2))*-1

        # mouse callback function
        def draw_polylines(event, x, y, flags, param):
            global ix, iy, drawing, mode, index
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                click_points[index]= np.array([x, y])
                print("ix, iy: {}, {}".format(click_points[index,0], click_points[index,1]))
                index = index + 1

        cv2.namedWindow('label', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('label', draw_polylines)

        while True:  # Runs forever until we break with Esc key on keyboard
            # Shows the image window
            cv2.imshow('label', image)
            if (cv2.waitKey(1) & 0xFF == 27) or index == 4:
                break

        shoulder_height = click_points[[0,2],1].mean()
        click_points[[0,2], 1] = shoulder_height

        shoulder_width = get_dist(click_points[0], click_points[2])

        label = np.zeros((10, 2))
        label[0] = get_midpoint(click_points[0], click_points[2]) #neck

        label[1] = click_points[0] #right_shoulder
        label[2] = click_points[0] + get_uintVec(click_points[0], click_points[1]) * shoulder_width * 0.809 #right_elbow
        label[3] = label[2] + get_uintVec(click_points[0], click_points[1]) * shoulder_width * 0.809 #right_wrist

        label[4] = click_points[2] #left_shoulder
        label[5] = click_points[2] + get_uintVec(click_points[2], click_points[3]) * shoulder_width * 0.809 #left_elbow
        label[6] = label[5] + get_uintVec(click_points[2], click_points[3]) * shoulder_width * 0.809 #left_wrist

        label[7] = label[0] + np.array([0, shoulder_width * 1.618])
        label[8] = label[7] + np.array([-shoulder_width * 0.309, 0])
        label[9] = label[7] + np.array([shoulder_width * 0.309, 0])


        keypoints = np.zeros((25, 3))
        for idx, useful_idx in enumerate(useful):    
            keypoints[useful_idx, :2] = label[idx]
            keypoints[useful_idx, 2] = 1
        keypoints = keypoints.astype(int)    
        print(keypoints)
        pose_format = {
                "version": 1.3,
                "people": [
                            {
                                "person_id": [-1],
                                "pose_keypoints_2d": keypoints.reshape(-1).tolist(),
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
        with open(os.path.join(output_dir, '{}_keypoints.json'.format(imgName)), 'w') as f:
            json.dump(pose_format, f)
        
        skeleton = draw_bodypose(image, keypoints, body_part='lower')
        cv2.namedWindow('skeleton', cv2.WINDOW_NORMAL)
        cv2.imshow("skeleton", skeleton)
        cv2.waitKey(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
                        type=str,
                        default='Training_Dataset/1024x768/Example_top/product/product')
    parser.add_argument("--body",
                        type=str,
                        default='top'
                        )
    opt = parser.parse_args()

    main(opt)
