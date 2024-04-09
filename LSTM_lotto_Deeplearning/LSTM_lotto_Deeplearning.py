
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import os

from tensorflow.python.saved_model.load import metrics
import chardet
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

dir = os.getcwd()
print(dir)

with open("./lotto.csv", 'rb') as f:
    rawdata = f.read()
    result = chardet.detect(rawdata)
    encoding = result['encoding']
#rows = pd.read_csv("./lotto.csv", encoding='utf-8')
rows = np.genfromtxt("./lotto.csv", delimiter=",", dtype='str', encoding='UTF8')
#rows = np.loadtxt("./lotto.csv", delimiter=",", encoding='UTF8')
                  
row_count = len(rows)
print(row_count)

def numbers2ohbin(numbers):
    ohbin = np.zeros(45) 
    for i in range(6): 
        ohbin[int(numbers[i]) - 1] = 1 
        
    return ohbin

def ohbin2numbers(ohbin):
    numbers = []
    for i in range(len(ohbin)):
        if ohbin[i] == 1.0:
            numbers.append(i+1)
        
    return numbers

numbers = rows[:, 1:7]
ohbins = list(map(numbers2ohbin, numbers))

x_samples = ohbins[0:row_count - 1]
y_samples = ohbins[1:row_count]

print("ohbins")
print ("X[0]: " + str(x_samples[0]))
print ("Y[0]: " + str(y_samples[0]))

print("numbers")
print("X[0]: " + str(ohbin2numbers(x_samples[0])))
print("Y[0]: " + str(ohbin2numbers(y_samples[0])))

train_idx = (0, 800)
val_idx = (801, 900)
test_idx = (901, len(x_samples))

print("train : {0}, val : {1}, test : {2}".format(train_idx,val_idx,test_idx))

#####################################################################################
# 모델 생성 및 학습
#####################################################################################

model = keras.Sequential([
    keras.layers.LSTM(128, batch_input_shape=(1,1,45), return_sequences=False, stateful=True),
    keras.layers.Dense(45, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

train_loss = []
train_acc = []
val_loss = []
val_acc = []

for epoch in range(100):
    model.reset_states()
    
    batch_train_loss = []
    batch_train_acc = []
    
    for i in range(train_idx[0], train_idx[1]):
        xs = x_samples[i].reshape(1,1,45)
        ys = y_samples[i].reshape(1,45)
        
        loss, acc = model.train_on_batch(xs, ys)
        
        batch_train_loss.append(loss)
        batch_train_acc.append(acc)
        
    train_loss.append(np.mean(batch_train_loss))
    train_acc.append(np.mean(batch_train_acc))
    
    batch_val_loss = []
    batch_val_acc = []
    
    for i in range(val_idx[0], val_idx[1]):
        xs = x_samples[i].reshape(1,1,45)
        ys = y_samples[i].reshape(1,45)
        
        loss, acc = model.test_on_batch(xs,ys)
        batch_val_loss.append(loss)
        batch_val_acc.append(acc)
        
    val_loss.append(np.mean(batch_val_loss))
    val_acc.append(np.mean(batch_val_acc))
    

