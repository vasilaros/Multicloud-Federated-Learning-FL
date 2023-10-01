#https://stackabuse.com/time-series-analysis-with-lstm-using-pythons-keras-library/

# -*- coding: utf-8 -*-
"""
Created on Fri Oct  16 17:05:47 2020

@author: stefanidis
"""

#Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
path = os.path.join(os.getcwd(),'w_2_metric')
files = os.listdir(path)

# Find the csv files Dataset

csv_files = [f for f in files if f.split('.')[-1] == 'csv' ]

# Read the csv files in pandas

def read_csv(fname, path=os.getcwd()):
    '''
    Reads a csv file and returns a pandas dataframe
    '''
    df = pd.read_csv(os.path.join(path,fname), header=None)
    df.columns = [fname.split('.')[0]+str(i) for i in df.columns]
    return df

df_total = None
for f in csv_files:
    df = read_csv(f, path='w_2_metric')
    if df_total is None:
        df_total = df
    else:
        df_total = df_total.join(df)

#from sklearn.model_selection import train_test_split
df_ = df_total.abrupt1

# Diverse the train and test data of Dataset
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
train, test = df_.iloc[:1500], df_.iloc[1500:]

#train_processed = train.iloc[:1500].values

#train_processed = train.reshape(1,-1)

#apple_training_processed = apple_training_complete.iloc[:, 1:2].values

#Import Dataset
#apple_training_complete = pd.read_csv(r'E:\Datasets\apple_training.csv')
#apple_training_processed = apple_training_complete.iloc[:, 1:2].values

#Data Normalization on Training Data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))
#apple_training_scaled = scaler.fit_transform(apple_training_processed)
train_scaled = scaler.fit_transform(train.values.reshape(-1,1))

#Convert Training Data to Right Shape
features_set = []
labels = []
for i in range(60, 1500):
    features_set.append(train_scaled[i-60:i, 0])
    labels.append(train_scaled[i, 0])

features_set, labels = np.array(features_set), np.array(labels)
features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))

#Training The LSTM and creating 4 layers model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
model = Sequential()

#Creating LSTM and Dropout Layers
model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

#Creating Dense Layer
model.add(Dense(units = 1))

#Model Compilation
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Algorithm Training based on the previous defintion
model.fit(features_set, labels, epochs = 100, batch_size = 32)

#Testing our LSTM
#apple_testing_complete = pd.read_csv(r'E:\Datasets\apple_testing.csv')
#apple_testing_processed = apple_testing_complete.iloc[:, 1:2].values

#Converting Test Data to Right Format
#apple_total = pd.concat((apple_training_complete['Open'], apple_testing_complete['Open']), axis=0)



#test_inputs = apple_total[len(apple_total) - len(apple_testing_complete) - 60:].values
test_inputs = df_[len(df_) - 500 - 60:].values

test_inputs = test_inputs.reshape(-1,1)
test_inputs = scaler.transform(test_inputs)

test_features = []
for i in range(60, 560):
    test_features.append(test_inputs[i-60:i, 0])
test_features = np.array(test_features)
test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))

#Making Predictions
predictions = model.predict(test_features)
predictions = scaler.inverse_transform(predictions)

plt.figure(figsize=(10,6))
plt.plot(test, color='blue', label='Actual CPU')
plt.plot(predictions , color='red', label='Predicted CPU')
plt.title('CPU Prediction')
plt.xlabel('Time')
plt.ylabel('CPU Consmp')
plt.legend()
plt.show()


###  OLD  plot
#model = SimpleExpSmoothing(train+1e-3).fit(smoothing_level=0.2,optimized=False)

#pred = model.predict(start=test.index[0], end=test.index[-1])

#plt.plot(train.index, train, label='Train')
#plt.plot(test.index, test, label='Test')
#plt.plot(pred.index, pred, label='SimpleExpSmoothing')
#plt.legend(loc='best')


#print(model.params['smoothing_level'])
#print(model.params)