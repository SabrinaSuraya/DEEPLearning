# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 15:02:25 2022

@author: sabri
regression problem

EXERCISE2
1- USE IRIS DATA FROM SKLEARN TO CONSTRUCT A NN
2- PERFORM THE NESSESARY DATA PREPARATION STEPS
3- CONSTRUCT A NN USING FUNCTIONAL API WITH 3 HIDDEN LAYERS.


"""

import sklearn.datasets as skdatasets
from sklearn import preprocessing
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

(iris_features, iris_labels)= skdatasets.load_iris(return_X_y=True, as_frame=True)
nIn = iris_features.shape[-1]
nClass= len(np.unique(iris_labels))
#%%
#train & test split

SEED=12345
x_train, x_test, y_train, y_test= train_test_split(iris_features, iris_labels, test_size=0.2, random_state=SEED)
#DAta normalisation
standardizer = StandardScaler()
standardizer.fit(x_train)
x_train= standardizer.transform(x_train)
x_test= standardizer.transform(x_test)

#Define model

inputs= keras.Input(shape=(nIn,))
h1= layers.Dense(64, activation='relu')
h2= layers.Dense(32, activation='relu')
h3= layers.Dense(16, activation='relu')
out= layers.Dense(nClass, activation='softmax')

x= h1(inputs)
x= h2(x)
x= h3(x)
outputs= out(x)

model= keras.Model(inputs=inputs, outputs=outputs) # FUNCTIONAL API


#Compile model

model.compile(optimizer= 'adam', loss= 'sparse_categorical_crossentropy', metrics= ['accuracy'])

#Perform model training
epochs=10
history= model.fit(x_train, y_train, validation_data= (x_test, y_test), batch_size= 16, epochs=epochs)

#Visualisation on model training
import matplotlib.pyplot as plt

training_loss= history.history['loss']
val_loss = history.history['val_loss']
training_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs= history.epoch
plt.plot(epochs, training_loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title("Training vs Validation Loss")
plt.legend()
plt.figure()

plt.plot(epochs, training_acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.figure()

plt.show()