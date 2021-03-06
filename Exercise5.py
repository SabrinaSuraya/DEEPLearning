# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 09:21:31 2022

@author: sabri
EXERCISE 5
WOMEN CHESS PLAYER DATA- CLASSIFICATION PROBLEM- ACTIVE OR INACTIVE
CHECK DATA IF NaN
    1-DATA CLEANING
    2-FEATURE PREPROCESSSING(CATEGORICAL DATA)
    3-TRAIN-TEST SPILT/ K-FOLD
    4- DATA NORMALISATION
NN WITH THAT CAN MAKE PREDICTION
VISUALISE TRAIN WITH TESNSORBOARD
FINE-TUNE MODEL BY ADJUST HYPERPARAMETER BASED ON TENSORBOARD
"""

import sklearn.datasets as skdatasets
from sklearn import preprocessing
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, metrics
#import tensorflow_transform as tft
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime, os
import pandas as pd

#read data from csv
file_path= r"C:\Users\sabri\Documents\PYTHON\DL\top_women_chess_players_aug_2020.csv"
chess_data= pd.read_csv(file_path, sep=',', header= 0 )

#%%
print(chess_data.isna().sum())

#%%
#DATA CLEANING

# remove irrelevent data
chess_data=chess_data.drop(['Fide id', 'Name', 'Gender'], axis= 1)
#remove data of Column Year of Birth
chess_data=chess_data.dropna(subset=['Year_of_birth'])
#update the data value
chess_data['Inactive_flag'].replace(to_replace=np.nan, value='active')
chess_data['Inactive_flag'].replace(to_replace='wi', value='inactive')
#replace data in column title with other
chess_data=chess_data.fillna(value={'Title': 'Other', 'Rapid_rating': 0, 'Blitz_rating': 0})

print(chess_data.isna().sum())

#%%
#DATA PREPROCESSING
#Label encode the federation column
encoder= LabelEncoder()
encoder.fit(chess_data['Federation'])
chess_data['Federation']= encoder.transform(chess_data['Federation'])

#One-hot encode the Title
chess_data=pd.get_dummies(chess_data)

#%%

#split into features and labels
label_name= ['Inactive_flag_active', 'Inactive_flag_inactive']
features = chess_data.copy()
labels= pd.concat([chess_data.pop(x) for x in label_name], axis=1)

#%%

#Prepare train_test split 
SEED=12345
x_train, x_test, y_train, y_test= train_test_split(features, labels, test_size=0.2, random_state=SEED)

#DAta normalisation
standardizer = StandardScaler()
standardizer.fit(x_train)
x_train= standardizer.transform(x_train)
x_test= standardizer.transform(x_test)

##DATA PREPARATION IS COMPLETED
#%%
#4. Construct NN model

nIn=x_train.shape[-1]
nClass= y_train.shape[-1]

#use functional API
inputs= keras.input(shape=(nIn,))
h1=layers.Dense(512, activation='relu')
h2=layers.Dense(256, activation='relu')
h3=layers.Dense(128, activation='relu')
h4=layers.Dense(64, activation='relu')
h5=layers.Dense(32, activation='relu')
h6=layers.Dense(16, activation='relu')
out= layers.Dense(nClass, activation='softmax')

x= h1(inputs)
x= h2(x)
x= h3(x)
x= h4(x)
x= h5(x)
x= h6(x)
outputs= out(x)

# compile
model= keras.Model(inputs=inputs, outputs=outputs)
model.summary()

#, define callback function 
base_log_path= r"C:\Users\sabri\Documents\PYTHON\DL\TensorBoard\tb_callback_boston"
log_path= os.path.join(base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
es= EarlyStopping(monitor='val_loss',patience=20)
tb= TensorBoard(log_dir= log_path)

# Train model

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_text, y_test), batch_size=16, epochs=10, callbacks=tb)

#%%

#make prediction
prediction= tf.argmax(model(x_test), axis= 1)
confusion_matrix= tf.math.confusion_matrix(y_test, prediction)
print(confusion_matrix)

