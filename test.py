# quick test script
import os
import tensorflow.keras as keras
from autoencoder.generator import data_generator
from utils import load_data
import matplotlib.pyplot as plt

SAVE_RESULT = './imgs/'
MODEL_PATH = './saved_model/model_out.h5'
batch_size = 4

if not os.path.exists(SAVE_RESULT):
    os.makedirs(SAVE_RESULT)

model = keras.models.load_model(MODEL_PATH)

_, _, test = load_data()

test_gen = data_generator('test', test, batch_size=batch_size, input_shape=(96,96), target_shape=(192,192))

fig, ax = plt.subplots(batch_size, 3, figsize=(16,16*3))
ax = ax.reshape(-1)

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
    fig.savefig(f'{SAVE_RESULT}results_{k}.png')

    input('press any button to check next images')

