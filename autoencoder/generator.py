import os
import numpy as np
from config import DATA_PATH
import tensorflow.keras as keras

def data_generator(path, files, batch_size, input_shape, target_shape, shuffle=True, seed = 10): 
    '''
    Generator to feed into memory
    Scale the image here
    '''

    n_steps = int(np.ceil(len(files)/batch_size))
    if shuffle:
      np.random.seed(seed)
      np.random.shuffle(files)
    while True:
        shift = 0
        for istep in range(n_steps):
            n_data = batch_size if istep < n_steps-1 else (len(files)-(n_steps-1)*batch_size)
            x = np.zeros((n_data, *input_shape, 3))
            y = np.zeros((n_data, *target_shape, 3))
            for i in range(n_data):
              image = keras.preprocessing.image.load_img(os.path.join(DATA_PATH, path, 'heatmap_'+str(files[shift+i])+'.jpg'))
              input_arr = keras.preprocessing.image.img_to_array(image)
              image = keras.preprocessing.image.load_img(os.path.join(DATA_PATH, path, 'img_'+str(files[shift+i])+'.jpg'))
              target_arr = keras.preprocessing.image.img_to_array(image)
              x[i, :, :, :] = input_arr
              y[i, :, :, :] = target_arr
            shift += n_data
            x /= 255.
            y /= 255.
            yield x, y


