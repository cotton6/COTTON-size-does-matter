import os
import glob
from shutil import copyfile


cloth_types = glob.glob('data/*')
print(cloth_types)

for cloth_type in cloth_types:
    dir_train = os.path.join(cloth_type, 'train')
    dir_test = os.path.join(cloth_type, 'test')
    os.makedirs(dir_train, exist_ok=True)
    os.makedirs(dir_test, exist_ok=True)

    images = glob.glob(os.path.join(cloth_type, '*.jpg'))
    images.sort(key=lambda x:int(os.path.basename(x).split('_')[-1].split('.')[0]))

    train_set = images[:int(len(images)*0.8)]
    test_set = images[int(len(images)*0.8):]

    for img in train_set:
        newName = os.path.join(dir_train, os.path.basename(img))
        os.rename(img, newName)

    for img in test_set:
        newName = os.path.join(dir_test, os.path.basename(img))
        os.rename(img, newName)