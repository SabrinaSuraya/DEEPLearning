# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 14:18:44 2022

@author: sabri

Create a RNN for sentiment analysis.
Sentiment analysis is a type of  test classification task, whereby w eclassify the emotions behind a piece of texts

use dataset called IMND reviews, tensorflow has the dataset readied for use straight away

"""

#Import the packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers
import tensorflow_datasets as tfds
import numpy as np
import io
import matplotlib.pyplot as plt

#load data  from tensorflow datasets
(train_data, test_data), info= tfds.load('imdb_reviews', split=(tfds.Split.TRAIN, tfds.Split.TEST), with_info=True, as_supervised=True)

for example, label in train_data.take(1):
    print(f'Text: {example.numpy()}')
    print(f'Label: {example.numpy()}')

#%%

#Change the settings of the pretech dataset
BUFFER_SIZE= 1000
BATCH_SIZE= 64

train_dataset= train_data.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
train_dataset= train_dataset.cache()
test_dataset= test_data.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset= test_dataset.cache().repeat()

#Now we have 64 data pr batch
"""
HERE we are essentislly doing Natural Language Processing (NLP) tasks, For NLP task, there are general process incocled.

1. texts are in strings, we need to convert them into numerical representation. This process called tokenization
2. After tokenization, the process that convert individual words into numbers is calles word embedding
3. word embedding are in number, which is something that neutral network can accept.

for tokenization:
    a- vocabulary: a list of words that we will select to be converted into its own numerical representation.
    Any words that is not inluded will be teated as out of vacabulary (OOV)
    b- in tokenization, you will define teh size of vocabulary, that will determine on how many will get their own special representation.
    

"""
#Start with tokenization
VOCAB_SIZE= 1000
encoder= layers.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label : text))

#Show some example of vocabs
vocab= np.array(encoder.get_vocabulary())
print(vocab[:30])

# we can pass some data into tokenization and see the process happens

for example, label in train_dataset.take(1):
    print(f'Text: {example.numpy()}')
    print(f'Label: {example.numpy()}')

print("Reviews: ", example[:3])
encoded_example= encoder(example)[:3].numpy()
print("Encoded examples; ", encoded_example)

#%%

embedding= layers.Embedding(input_dim=len(encoder.get_vocabulary()), output_dim=64, mask_zero=True)
embedding_example= embedding(encoded_example).numpy
print(embedding_example)


#%%

"""
Now we wil start to build teh model. this model will included the tokenizatuon
follow by word embedding layer, then follow by a RNN that will 
output at the end.

Input(Text)--> Tokenization(TextVectorization)--> Embedding --> RNN --> Output

"""
# Create teh model
model= keras.Sequential()
model.add(encoder)
model.add(embedding)
model.add(layers.Bidirectional(layers.LSTM(64)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))

#%%

#Compile and train your model

rp_train= train_dataset
rp_test= test_dataset.repeat()


optimizer= optimizers.Adam(0.0001)#0.0001 is learning rate
loss= losses.BinaryCrossentropy(from_logits= True)

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

history= model.fit(rp_train, validation_data= rp_test, validation_steps=30, epochs=10 )

































# Start with tokenization
 