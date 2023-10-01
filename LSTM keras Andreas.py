#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, SimpleRNN
from tensorflow.keras.layers import Dropout

time_dimension  = 60
train_size      = 1500
test_size       = 500

learning_rate   = 0.01
optimizer       = tf.train.AdagradOptimizer
n_epochs        = 10
n_embeddings    = 50


model1 = Sequential()
#Creating LSTM and Dropout Layers
model1.add(SimpleRNN(units=n_embeddings, input_shape=(time_dimension, 1)))
#model1.add(LSTM(units=50, return_sequences=True))
#model1.add(LSTM(units=50))
model1.add(Dense(units = 1))

#Model Compilation
model1.compile(optimizer = 'adam', loss = 'mean_squared_error')
# #Algorithm Training based on the previous defintion
# model1.fit(features_set, labels, epochs = 10, batch_size = 32)


#test_inputs = apple_total[len(apple_total) - len(apple_testing_complete) - 60:].values
test_inputs = df_[len(df_) - 500 - 60:].values

test_inputs = test_inputs.reshape(-1,1)
test_inputs = scaler.transform(test_inputs)
test_features = []
for i in range(60, 560):
    test_features.append(test_inputs[i-60:i, 0])
test_features = np.array(test_features)
test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))

x_ = x_test[:10]
y_ = y_test[:10]



with tf.GradientTape() as tape:
    yp = model1(x_)
    # Compute the loss value for this minibatch.
    loss_ = tf.reduce_mean(tf.square(y_-yp))
grads = tape.gradient(loss_, model1.trainable_weights)


#Making Predictions
predictions = model1.predict(test_features)
predictions = scaler.inverse_transform(predictions)

plt.figure(figsize=(10,6))
plt.plot(test, color='blue', label='Actual CPU')
plt.plot(predictions , color='red', label='Predicted CPU')
plt.title('CPU Prediction')
plt.xlabel('Time')
plt.ylabel('CPU Consmp')
plt.legend()
plt.show()




