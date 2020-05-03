# -*- coding: utf-8 -*-
"""
Created on Sun May  3 22:34:24 2020

@author: s164405
"""
# Importing the necessary packages 
import numpy as np
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import os 
from sklearn.preprocessing import StandardScaler
# Importing the dataset
(X_train, y_train) = np.float32(np.loadtxt('train.txt'))
(X_test, y_test) = np.float32(np.loadtxt('test.txt'))

# Feature Scaling
sc = StandardScaler()
X_train = np.array(X_train).reshape(-1,1)
X_test = np.array(X_test).reshape(-1,1)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
y_train = np.array(y_train).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1)
y_train = sc.fit_transform(y_train)
y_test = sc.transform(y_test)

# Creating the model 
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, activation='relu',input_shape=(1,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
        ])

    model.compile(loss='mean_squared_error',
              optimizer='sgd') #accuracy only for classification; metrics=['accuracy'])
    keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    return model 

model = create_model()
model.summary()

checkpoint_path = "practical_2/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=0)

# Fitting the NN to the training set
history = model.fit(X_train, y_train, 
                    batch_size = 32, 
                    epochs = 1000,
                    validation_data=(X_test,y_test)) #,
                    #callbacks=[cp_callback]) Do this to create checkpoints of the weights (warning: takes time to save)

y_pred = model.predict(X_test)
plt.figure()
plt.plot(y_test, color = 'red', label = 'y')
plt.plot(y_pred, color = 'blue', label = 'predicted y')
plt.title('Prediction of y via deep NN with ReLU activation function')
plt.legend()
plt.show()

# plot the loss curves 
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

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
