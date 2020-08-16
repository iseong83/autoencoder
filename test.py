# quick test script
import os
import argparse
import tensorflow as tf
import tensorflow.keras as keras
from autoencoder.generator import data_generator
from utils import load_data
import matplotlib.pyplot as plt
from config import MODEL_PATH

SAVE_RESULT = './imgs/'

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--batch_size", help='batch size', default=4, type=int)
parser.add_argument("--model", help='model name', default='model_out.h5', type=str)

args = vars(parser.parse_args())

batch_size = args.pop('batch_size')
MODEL_PATH = os.path.join(MODEL_PATH, args.pop('model'))

if not os.path.exists(SAVE_RESULT):
    os.makedirs(SAVE_RESULT)

# load model
model = keras.models.load_model(MODEL_PATH)

# load the test set
_, _, test = load_data()

# create a generator
test_gen = data_generator('test', test, batch_size=batch_size, input_shape=(96,96), target_shape=(192,192))

fig, ax = plt.subplots(batch_size, 3, figsize=(16,16))
ax = ax.reshape(-1)
avg_ssim = []
for k in range(len(test)//batch_size):
    x, y = next(test_gen)
    results = model.predict(x)

    for i in range(len(y)):
        ax[i*3].imshow(x[i])
        ax[i*3].set_title('Input')
        ax[i*3+1].imshow(y[i])
        ax[i*3+1].set_title('Ground Truth')
        ax[i*3+2].imshow(results[i])
        ax[i*3+2].set_title('Generated Image')

        im1 = tf.image.convert_image_dtype(y[i], tf.float32)
        im2 = tf.image.convert_image_dtype(results[i], tf.float32)
        ssim = tf.image.ssim(im1, im2, max_val=1)
        avg_ssim.append(ssim)
    fig.savefig(f'{SAVE_RESULT}results_{k}.png')
    #input('press any button to check next images')
print('Average SSIM: {:.4f}'.format(sum(avg_ssim)/len(avg_ssim)))
