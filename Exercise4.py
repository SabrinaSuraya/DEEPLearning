# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 15:07:20 2022

@author: sabri

REGRESSION PROBLEM
EXERCISE  4
1- USE BOSTON HOUSE PRICE DATASETS FROM SKLEARN
2- CREATE A NN MODEL WITH 6 HIDDEN LAYERS(MAKE IT AS OVERFIT)
3-MODEL TRAINING, USE EARLY SROPPING AND TENSORBNOARD USE EPOCH =500
4- VIEW MODEL TRAINING RESULT IN TENSORBOARD
NO NEED K-FOLD CV
"""
import sklearn.datasets as skdatasets
from sklearn import preprocessing
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, metrics
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime, os

#2. Data Preparation 
#Import the data from sklearn
(boston_features, boston_labels)= skdatasets.load_boston(return_X_y=True)#regressionproblem



#train & test split

SEED=12345
x_train, x_test, y_train, y_test= train_test_split(boston_features, boston_labels, test_size=0.2, random_state=SEED)

#DAta normalisation
standardizer = StandardScaler()
standardizer.fit(x_train)
x_train= standardizer.transform(x_train)
x_test= standardizer.transform(x_test)


#4. Define your model
model= keras.Sequential()
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1))

# compile
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

#, define callback function 
base_log_path= r"C:\Users\sabri\Documents\PYTHON\DL\TensorBoard\tb_callback_boston"
log_path= os.path.join(base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
es= EarlyStopping(monitor='val_loss',patience=20)
tb= TensorBoard(log_dir= log_path)

# Train model
history=model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=16, epochs=500, callbacks=[es,tb])







