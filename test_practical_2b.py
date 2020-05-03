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

# Fitting the ANN to the Training set
history = model.fit(X_train, y_train, 
                    batch_size = 32, 
                    epochs = 1000,
                    callbacks=[cp_callback])

y_pred = model.predict(X_test)

plt.plot(y_test, color = 'red', label = 'y')
plt.plot(y_pred, color = 'blue', label = 'predicted y')
plt.title('Prediction of y via deep NN with relu activation function')
plt.legend()
plt.show()
