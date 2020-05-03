# -*- coding: utf-8 -*-
"""
Created on Sun May  3 22:34:24 2020

@author: s164405
"""
import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Importing the dataset
#dataset = np.genfromtxt("yacht_hydrodynamics.data", delimiter='')
#X = dataset[:, 5] #:-1]
#y = dataset[:, -1]

# Splitting the dataset into the Training set and Test set
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.08, random_state = 0)
(X_train, y_train) = np.float32(np.loadtxt('train.txt'))
(X_test, y_test) = np.float32(np.loadtxt('test.txt'))


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

#def norm(x):
#    mean = np.mean(X_train)
#    std = np.std(X_train)
#    return (x - mean) / std
#def norm2(x):
#    mean = np.mean(y_train)
#    std = np.std(y_train)
#    return (x - mean) / std
#X_train = norm(X_train)
#X_test = norm(X_test)
#y_train = norm2(y_train)
#y_test = norm2(y_test)
X_train = np.array(X_train).reshape(-1,1)
X_test = np.array(X_test).reshape(-1,1)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
y_train = np.array(y_train).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1)
y_train = sc.fit_transform(y_train)
y_test = sc.transform(y_test)
# Initialising the ANN
model = Sequential()

# Adding the input layer and the first hidden layer
model.add(Dense(32, activation = 'relu', input_dim = 1))

# Adding the second hidden layer
model.add(Dense(units = 32, activation = 'relu'))

# Adding the third hidden layer
model.add(Dense(units = 32, activation = 'relu'))

# Adding the output layer

model.add(Dense(units = 1))

#model.add(Dense(1))
# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size = 32, epochs = 1000)

y_pred = model.predict(X_test)

plt.plot(y_test, color = 'red', label = 'Real data')
plt.plot(y_pred, color = 'blue', label = 'Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()
