# Training and Inference

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

## Training Example
Here we provide a simple example for training. Please change the config file and data root for personal use.
```
python main.py --config configs/config_top_Example.yaml
```