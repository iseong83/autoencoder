import numpy as np
import argparse
import matplotlib.pyplot as plt
from autoencoder.model import NN, callbacks
from autoencoder.generator import data_generator
from utils import load_data
from config import *

def train():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--optimizer",  help='optimizer', default='adam', type=str)
    parser.add_argument("--epochs",     help='number of epoch', default=50, type=int)
    parser.add_argument("--batch_size", help='batch size', default=16, type=int)
    parser.add_argument("--gpu_id",     help='GPU ID', default=0, type=int)
    parser.add_argument("--lr",         help='learning Rate', default=0.01, type=float)
    parser.add_argument("--momentum",   help='momentum for SGC', default=0.9, type=float)
    parser.add_argument("--output_model", help='output model file name', default='model_out.h5', type=str)

    args = vars(parser.parse_args())
    epochs = args.pop('epochs')
    batch_size = args.pop('batch_size')
    opt = args.pop('optimizer')
    gpu_id = args.pop('gpu_id')
    lr = args.pop('lr')
    mom = args.pop('momentum')

    # load datasets
    train, valid, test = load_data()
    # set number of training and validation samples
    n_train, n_valid, n_test = len(train), len(valid), len(test)

    n_train_steps = int(np.ceil(n_train/batch_size))
    n_valid_steps = int(np.ceil(n_valid/batch_size))
    n_test_steps = int(np.ceil(n_test/batch_size))

    # Initialize the network
    net = NN(input_shape=(96,96,3), gpu_id=gpu_id)
    # build architecture with the given optimizer and learning rate
    model = net.build_model(opt, lr, mom, n_train_steps)
    # print model summary
    print(model.summary())
    print("Number of training samples:{}, validataion sample:{}".format(n_train,n_valid))

    # keras callback
    # This includes Cosine learing rate, model output name
    model_callbacks = callbacks(args.pop('output_model'))


    # use generator to generate training and validation data
    # Images are scaled in the generator
    train_gen = data_generator('train', train, batch_size=batch_size, input_shape = (96,96), target_shape = (192,192), seed=32)
    valid_gen = data_generator('valid', valid, batch_size=batch_size, input_shape = (96,96), target_shape = (192,192), seed=102)

    # training with generator
    # print out losses
    history = model.fit_generator(generator=train_gen, 
            steps_per_epoch=n_train_steps, 
            epochs = epochs, workers = 1,
            validation_data=valid_gen, 
            validation_steps = n_valid_steps, 
            callbacks=model_callbacks)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('losses_{}_{}_{}_{}.png'.format(batch_size,opt,lr,mom))

if __name__ == "__main__":
    train()
