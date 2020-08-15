import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras import backend as K
from config import *

def conv2dBNleaky(x, filter_size, kernel_size, strides=(1,1), padding='same', act=True):
    '''Conv2D with Batch Normalization and LeakyReLu as an activation function
    Arugments:
      x: input tensor
      filter_size: integer, a filter size for convolution layer
      kerner_size: integer or tuple, size of convolution filter
      strides: strides for conv2D
      act: boolean, applying the activation function or not
    Returns:
      output tensor
    '''
    layers = tf.keras.layers.Conv2D(filter_size, kernel_size, strides=strides, padding=padding)(x)
    layers = tf.keras.layers.BatchNormalization()(layers)
    layers = tf.keras.layers.LeakyReLU()(layers)
    layers = tf.keras.layers.Dropout(0.2)(layers)
    return layers

def conv2dtransposeBNleaky(x, filter_size, kernel_size, strides=(1,1), padding='same', act=True):
    '''Conv2D with Batch Normalization and LeakyReLu as an activation function
    Arugments:
      x: input tensor
      filter_size: integer, a filter size for convolution layer
      kerner_size: integer or tuple, size of convolution filter
      strides: strides for conv2D
      act: boolean, applying the activation function or not
    Returns:
      output tensor
    '''
    layers = tf.keras.layers.Conv2DTranspose(filter_size, kernel_size, strides=strides, padding=padding)(x)
    layers = tf.keras.layers.BatchNormalization()(layers)
    layers = tf.keras.layers.LeakyReLU()(layers)
    layers = tf.keras.layers.Dropout(0.2)(layers)
    return layers

class NN:
    def __init__(self, input_shape = (96, 96), gpu_id=0):
        '''Initialize the Network
        Arguments:
          input_shape: the input shape of images
          gpu_id: the index of gpu to use only one gpu intead of using all available gpus 
        '''
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
                tf.debugging.set_log_device_placement(False)
            except:
                print("Can't config GPU")

        self.input_shape = input_shape
        self.size = input_shape[0]

    def build_model(self, optimizer, learning_rate, momentum, n_steps):
        '''Building Keras Model
        Arguments:
          optimizer: a string to select the optimizer
          learning rate: initial learning rate
          momentum: momentum for SGD
          n_steps: number of steps per epoch for cosine learning rate
        '''
        input_shape = self.input_shape
      
        inputs = tf.keras.Input(shape=input_shape)

        # 96x96
        conv1 = conv2dBNleaky(inputs, 64, 3, strides=1, padding='same')
        conv2 = conv2dBNleaky(conv1, 64, 3, strides=2, padding='same')
        
        # 48x48
        conv3 = conv2dBNleaky(conv2, 128, 5, strides=2, padding='same')
        
        # 24x24
        conv4 = conv2dBNleaky(conv3, 128, 3, strides=1, padding='same')
        conv5 = conv2dBNleaky(conv4, 256, 5, strides=2, padding='same')
        
        # 12x12
        conv6 = conv2dBNleaky(conv5, 512, 3, strides=2, padding='same')
        
        # 6x6
        deconv1 = conv2dtransposeBNleaky(conv6, 512, 3, strides=2, padding='same')
        
        # 12x12
        skip1 = tf.keras.layers.concatenate([deconv1, conv5], name='skip1')
        conv7 = conv2dBNleaky(skip1, 256, 3, strides=1, padding='same')
        deconv2 = conv2dtransposeBNleaky(conv7, 128, 3, strides=2, padding='same')
        
        # 24x24
        skip2 = tf.keras.layers.concatenate([deconv2, conv3], name='skip2')
        conv8 = conv2dBNleaky(skip2, 128, 3, strides=1, padding='same')
        deconv3 = conv2dtransposeBNleaky(conv8, 64, 3, strides=2, padding='same')
        
        # 48x48
        skip3 = tf.keras.layers.concatenate([deconv3, conv2], name='skip3')
        conv9 = conv2dBNleaky(skip3, 64, 3, strides=1, padding='same')
        deconv4 = conv2dtransposeBNleaky(conv9, 64, 3, strides=2, padding='same')
        
        # 96x96
        skip3 = tf.keras.layers.concatenate([deconv4, conv1])
        conv10 = conv2dBNleaky(skip3, 64, 3, strides=1, padding='same')
        deconv5 = conv2dtransposeBNleaky(conv10, 64, 3, strides=2, padding='same')
    
        # 192x192
        conv_final = tf.keras.layers.Conv2D(3, 3, strides=1, padding='same', activation='sigmoid',
                                            name='final_conv')(deconv5)
    
        
        # Enable Cosine Learning Rate
        decay_steps = n_steps*10
        learning_rate = tf.keras.experimental.CosineDecay(learning_rate, decay_steps, alpha=0.2)

        # Enable an optimizer
        # tested with adam and sgd, adam works much better than sgd
        if optimizer.lower() == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate,
                    momentum=momentum)
        else:
            print('Wrong optimizer. Set to ADAM')
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        model = tf.keras.Model([inputs], [conv_final])

        # Compile the Model with the Loss function and the optimizer
        model.compile(loss='mse', optimizer=opt)
      
        return model

def callbacks(output_name='model_out.h5', log_name='keras_log.csv', reduce_lr=False):
    '''Keras callbacks
      ModelCheckpoint: store the best model based on validation loss
      EarlyStopping: Early stopping to prevent overfitting
      ReduceLROnPlateau: Reduce the learning rate if validation loss doesn't improve
    '''
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    cb = []
    cb.append(tf.keras.callbacks.ModelCheckpoint(os.path.join(MODEL_PATH,output_name), 
                                           monitor='val_loss', verbose=0, 
                                           save_best_only=True, mode='auto'))
    cb.append(tf.keras.callbacks.CSVLogger(os.path.join(LOG_PATH, log_name),
                                        append=True))
    #cb.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.1, patience=10))
    if reduce_lr:
        cb.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4,
                                        verbose=1,min_lr = 1e-06, min_delta=0.1))
    return cb

