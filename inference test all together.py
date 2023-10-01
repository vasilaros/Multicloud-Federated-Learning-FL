# -*- coding: utf-8 -*-
"""
Created on Sun May  1 14:14:54 2022

@author: VasilAngelo
"""

time_dimension = 60
n_embeddings = 50
learning_rate=0.001

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, SimpleRNN
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import mean_absolute_error
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.losses import mean_absolute_percentage_error
from keras.models import load_model
# Import Keras backend
import keras.backend as K
import numpy as np

# Create a model using keras
model = Sequential()         
model.add(LSTM(units=n_embeddings,return_sequences=False, input_shape=(time_dimension, 1)))
model.add(Dropout(0.2))
model.add(Dense(units = 1))
opt = Adam(learning_rate=learning_rate)
model.compile(optimizer = opt, loss = 'mean_squared_error')
        



#model.fit(x=[-1, 0, 1], y=[-3, -1, 1], epochs=5) # train the model


# (to generate a SavedModel) tf.saved_model.save(model, "saved_model_keras_dir")
print("keras model: ", model)
model.summary() 
# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.allow_custom_ops=True
converter.experimental_new_converter = True
tflite_model = converter.convert()

  
# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_content=tflite_model)
#interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()


# Get input and output tensors.
input_details = interpreter.get_input_details()
print(input_details)
output_details = interpreter.get_output_details()


# Test the model on random input data.
input_shape = input_details[0]['shape']
print(input_shape)
x=x_test

y = np.zeros(x.shape[0])

        #for ix, *x_ in x:
            
for ix in range(x.shape[0]):
    x_=x[ix]  
    x_ = np.expand_dims(x_, axis=0)
    x_ = np.array(x_, dtype=np.float32)
    #print("ix,x_shape:",ix,x_.shape)
    
    interpreter.set_tensor(input_details[0]['index'], x_)
   
    interpreter.invoke()
    y_ = interpreter.get_tensor(output_details[0]['index'])
    y[ix] = y_
        




input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
print("input data:", input_data)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print("output data:",output_data)

#tflite_model.summary() 