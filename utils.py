import os
import glob
import numpy as np
from config import *


def split_data(files, train_frac=0.8, valid_frac=0.1):
    '''
    Split the dataset into train, validation, and test sets
    '''
    n_files = len(files)
    train_idx = int(n_files*train_frac)
    valid_idx = int(n_files*(valid_frac+train_frac))

    return files[:train_idx], files[train_idx:valid_idx], files[valid_idx:]

def load_data():
    train_file_numbers = [k.split('_')[-1].split('.')[0] for k in glob.glob(os.path.join(DATA_PATH, '/train/img_*.jpg'))]
    valid_file_numbers = [k.split('_')[-1].split('.')[0] for k in glob.glob(os.path.join(DATA_PATH, '/valid/img_*.jpg'))]
    test_file_numbers = [k.split('_')[-1].split('.')[0]  for k in glob.glob(os.path.join(DATA_PATH, '/test/img_*.jpg'))]
    print('Number of training set {}, validation set {}, test set {}'.format(len(train_file_numbers), len(valid_file_numbers), len(test_file_numbers)))
    
    return train_file_numbers, valid_file_numbers, test_file_numbers


