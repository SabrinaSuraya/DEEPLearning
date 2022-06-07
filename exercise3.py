# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 11:33:37 2022

@author: sabri

REGRESSION PROBLEM
EXERCISE 3
1- USE BOSTON HOUSE PRICE DATASETS (LOAD_BOSTON) FROM SKLEARN 
2- CONSTRUCT A NN WITH 4 HIDDEN LAYERS
3- PERFORM K-FOLD CROSS VALIDATION WITH K=4
4- PRINT OUT THE AVERAGE EVALUATION RESULTS
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
features = np.array(boston_features)
labels = np.array(boston_labels)

#3. Prepare K-Fold data 
SEED= 12345
kfold = KFold(n_splits=4, shuffle=True, random_state=SEED)

#4. Define your model

model= keras.Sequential()
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1))

"""
for classification problem
nIn = boston_features.shape[-1]
nClass = len(np.unique(boston_labels))
inputs = keras.Input(shape=(nIn,))
h1 = layers.Dense(128, activation='relu')
h2 = layers.Dense(64, activation='relu')
h3 = layers.Dense(32, activation='relu')
h4 = layers.Dense(16, activation='relu')
out = layers.Dense(1)
x = h1(inputs)
x = h2(x)
x = h3(x)
outputs = out(x)
model = keras.Model(inputs=inputs, outputs=outputs)
"""
#%%

#5. Use a for loop to loop through the KFold, for each fold, you will perform 
"""
for classification problem
#model training and evaluation, then save the result for each fold
optimizer = optimizers.Adam(learning_rate=0.0001)
loss = losses.BinaryCrossentropy(from_logits=False)
accuracy = metrics.BinaryAccuracy()
precision = metrics.Precision()
recall = metrics.Recall()

#Create empty list to hold the individual results from each training
loss_list = []
acc_list = []
precision_list = []
recall_list = []
"""
fold_no = 1
loss_list = []
mae_list = []
scaler= StandardScaler()
#%%

for train, test in kfold.split(features, labels):
    #data normalization
    scaler.fit(features[train])
    features[train]= scaler.transform(features[train])
    features[test]= scaler.transform(features[test])
    
    #Compile model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    #Perform model training
    print('###############################################')
    print(f'Training for fold number {fold_no} ...')
    history = model.fit(features[train],labels[train], batch_size=16, epochs=10)
    
    #Generate the scores
    scores = model.evaluate(features[test], labels[test])
    print(f'Score for fold {fold_no}:')
    loss_list.append(scores[0])
    mae_list.append(scores[1])
    
    print(f'Evaluation from fold number {fold_no}: ')
    print('Loss : ', scores[0])
    print('Mae : ', scores[1])
    
    keras.backend.clear_session()
    
#%%  
  
#6. Print the average scores
print('---------------------------------------------------')
print('Average Loss: ', np.mean(loss_list))
print('Average MAE: ', np.mean(mae_list))








