import numpy as np
from config import DATA_PATH

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
      for i in range(n_steps):
        n_data = batch_size if i < n_steps-1 else (len(files)-(n_steps-1)*batch_size)
        x = np.zeros((n_data, *input_shape, 3))
        y = np.zeros((n_data, *target_shape, 3))
        for i in range(n_data):
          image = keras.preprocessing.image.load_img(os.path.join(DATA_PATH,'heatmap_'+str(files[i])+'.jpg'))
          input_arr = keras.preprocessing.image.img_to_array(image)
          image = keras.preprocessing.image.load_img(os.path.join(DATA_PATH,'img_'+str(files[i])+'.jpg'))
          target_arr = keras.preprocessing.image.img_to_array(image)
          x[i, :, :, :] = input_arr
          y[i, :, :, :] = target_arr
        x /= 255.
        y /= 255.
        yield x, y


