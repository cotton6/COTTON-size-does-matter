# Training and Inference

## Download required weight
Please run the following shell to automatically download the required weights.
```
./download_weights.sh
```
If the shell is not working, you can find the weights on the google drive: https://drive.google.com/drive/folders/1kZVKcurZ49qePDmJABWGcLFnRZs-q7Jv?usp=share_link

## Quick Start
We provide two pretrained weight, which are trained on the self-collect dataset (with face) and the DressCode (face masked) dataset.

Test paired result (./result/Top_1024x768_COTTON/val/43)
```
python main.py --config configs/config_top_COTTON.yaml --mode val
```

Test unpaired result (./result/Top_1024x768_COTTON/test/43)
```
python main.py --config configs/config_top_COTTON.yaml --mode test
```

## Advanced
Size scaling (./result/Top_1024x768_COTTON/val/43)
The following command would scale down the product skeleton, which scale up the clothing on human.
```
python main.py --config configs/config_top_COTTON.yaml --mode test --scale 0.9
```
The following command would scale up the product skeleton, which scale down the clothing on human.
```
python main.py --config configs/config_top_COTTON.yaml --mode test --scale 1.1
```


Clothing untucked (./result/Top_1024x768_COTTON/val/43_untucked)
```
python main.py --config configs/config_top_COTTON.yaml --mode test --untuck
```

## Training Example
Here we provide a simple example for training. Please change the config file and data root for personal use.
```
python main.py --config configs/config_top_Example.yaml
```