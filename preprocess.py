import os
from os.path import join
import glob
import numpy as np
from shutil import copyfile
from config import *
from utils import *

assert os.path.exists(RAW_DATA)

# read all raw heatmap images
raw_files = glob.glob(join(RAW_DATA, 'heatmap_*.jpg'))

# shuffle before split
np.random.seed(20)
np.random.shuffle(raw_files)

dirs = ['train', 'valid', 'test']
# split the dataset into 3
# Returns: train, valid, test sets
split_datasets = split_data(raw_files, train_frac=0.8, valid_frac=0.1)

for dirname, data in zip(dirs, split_datasets):
    path = join(DATA_PATH, dirname)
    print('store data into {}'.format(path))
    if not os.path.exists(path):
        os.makedirs(path)

    for filename in data:
        base, fname = os.path.split(filename)
        img = fname.replace('heatmap', 'img')

        copyfile(filename, join(path, fname))
        copyfile(join(base,img), join(path, img))

