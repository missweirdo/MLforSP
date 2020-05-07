# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt
import random
import glob
import os
import sys
import time
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

TRAIN_DATA_PATH = "C:/Users/Yogi/Documents/Study/MLforSP/train.txt"
TEST_DATA_PATH = "C:/Users/Yogi/Documents/Study/MLforSP/test.txt"
train_data = np.loadtxt(TRAIN_DATA_PATH)
test_data = np.loadtxt(TEST_DATA_PATH)

xtrain = train_data[0, :]
ytrain = train_data[1, :]
xtest = test_data[0, :]
ytest = test_data[1, :]
     

ax1=plt.subplot(2, 2, 1)
plt.plot(xtrain, ytrain)
plt.title('insert title')
plt.xlabel('w0')
plt.ylabel('w1')
ax2=plt.subplot(2, 2, 2)
plt.plot(xtest, ytest)
plt.title('insert title')
plt.xlabel('w0')
plt.ylabel('w1')
plt.show()


def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=None, input_shape=[1]), 
    layers.Dense(64, activation=None),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

checkpoint_path = "practical_2/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

checkpointer = ModelCheckpoint(filepath = checkpoint_path,
                               verbose = 1,
                               save_weights_only =True)
model=build_model()
model.summary()
EPOCHS = 1000
history = model.fit(
  xtrain, ytrain,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[checkpointer])

# Loads the weights
weights = model.load_weights(checkpointer)

# plt.plot(res_exp1, ytest)
# plt.title('insert title')
# plt.xlabel('w0')
# plt.ylabel('w1')
# plt.show()

# # Re-evaluate the model
# loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# model.fit(train_images, 
#           train_labels,  
#           epochs=10,
#           validation_data=(test_images,test_labels),
#           callbacks=[cp_callback])  # Pass callback to training


# plt.subplot()

# mnist = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10)
# ])