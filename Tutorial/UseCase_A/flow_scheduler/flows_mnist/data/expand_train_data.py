"""
用來將train_v2的資料都複製到train_v1，製作第二個版本的訓練資料
"""

import os
import shutil

mnist_dir = os.path.abspath('./MNIST/')
for digit in range(10):
    for data in os.listdir(os.path.join(mnist_dir, 'train_v2', str(digit))):
        shutil.copy(os.path.join(mnist_dir, 'train_v2', str(digit), data), os.path.join(mnist_dir, 'train', str(digit)))
