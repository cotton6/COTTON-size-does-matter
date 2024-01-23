# Code
## Requirements
Please install package first
```
pip3 install -r requirements.txt
```

## preprocessing
This folder contains all the code for preprocessing.

## main
This folder contains all the code for training and inference.

*Details clarification: We predict human segmentation at lower resolution since it can reduce computations and maintain performance, which is also used by VITON-HD [6] and HR-VITON [21].

[6] Seunghwan Choi, Sunghyun Park, Minsoo Lee, and Jaegul Choo. VITON-HD: High-resolution virtual try-on via misalignment-aware normalization. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

[21] Sangyun Lee, Gyojung Gu, Sunghyun Park, Seunghwan Choi, and Jaegul Choo. High-resolution virtual try-on with misalignment and occlusion-handled conditions. In European Conference on Computer Vision (ECCV), 2022.
