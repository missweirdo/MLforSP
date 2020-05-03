# -*- coding: utf-8 -*-
"""
Created on Sun May  3 13:29:54 2020

@author: s164405
"""
#close all 
import keras 
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import optimizers 
import os 
# Load training and test data 
(x_train, y_train) = np.float32(np.loadtxt('train.txt'))
(x_test, y_test) = np.float32(np.loadtxt('test.txt'))
# Plot training and test data 
fig = plt.figure()
plt.subplot(1,2,1)
plt.title('Training Data')
plt.plot(x_train, y_train,'-ok')
plt.subplot(1,2,2)
plt.title('Test Data')
plt.plot(x_test, y_test,'-ok')
plt.show()

# Build a simple fully connected network
# retrieved from: https://www.tensorflow.org/tutorials/estimator/keras_model_to_estimator
# https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
#
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(25, input_shape=(1,)),
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(100),
        tf.keras.layers.Dense(100),
        tf.keras.layers.Dense(1)
        ])

#Number of parameters: 
# layer 1: 25*16 weights + 16 biases = 416
# layer 2: 16*3 weights + 3 biases = 51 
    model.compile(loss='mean_squared_error',
              optimizer='sgd', metrics=['accuracy'])
    keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    return model 

model = create_model()
model.summary()

checkpoint_path = "practical_2/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
# Data preparation 
x_train = (x_train-np.min(x_train))/(np.max(x_train)-np.min(x_train))
x_test = (x_test-np.min(x_test))/(np.max(x_test)-np.min(x_test))
y_train = (y_train-np.min(y_train))/(np.max(y_train)-np.min(y_train))
y_test = (y_test-np.min(y_test))/(np.max(y_test)-np.min(y_test))

#x_train = np.array(x_train).reshape(-1,1)
#x_test = np.array(x_test).reshape(-1,1)
#x_train = sc.fit_transform(x_train)
#x_test = sc.transform(x_test)
print(y_test)
history = model.fit(x_train, y_train,
          batch_size = 32,
          epochs=100,
          verbose=1,
          validation_data=(x_test,y_test),
          callbacks=[cp_callback])  # Pass callback to training
test_prediction = model.predict(x_test, batch_size=4)
score = model.evaluate(x_test, y_test, verbose=0)

fig = plt.figure()
plt.title('Prediction on trained model')
plt.subplot(1,2,1)
plt.title('Test Data')
plt.plot(x_test, y_test,'-ok')
plt.subplot(1,2,2)
plt.title('Prediction')
plt.plot(x_test, test_prediction,'-ok')
plt.show()

print('Test loss:', score[0])
print('Test accuracy:', score[1])

## Create a new basic model instance that has not been trained before
#model = create_model()
## Evaluate the new model
#loss, acc = model.evaluate(x_test, y_test, verbose=2)
#print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
#test_prediction = model.predict(x_test, batch_size=4)
#fig = plt.figure()
#plt.title('Prediction on untrained model')
#plt.subplot(1,2,1)
#plt.title('Test Data')
#plt.plot(x_test, y_test,'-ok')
#plt.subplot(1,2,2)
#plt.title('Prediction')
#plt.plot(x_test, test_prediction,'-ok')
#plt.show()

# Loads the weights
#model.load_weights(checkpoint_path)
#
## Re-evaluate the model
#loss,acc = model.evaluate(x_test, y_test, verbose=2)
#test_prediction = model.predict(x_test, batch_size=4)
#print("Restored model, accuracy: {:5.2f}%".format(100*acc))




