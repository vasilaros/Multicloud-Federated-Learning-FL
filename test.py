# -*- coding: utf-8 -*-
"""
Created on Sat May  7 21:00:32 2022

@author: VasilAngelo
"""
import os, sys


import tensorflow as tf
from models.rnn_keras_abstract import ModelRNNAbstract
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


adj = ["red", "big", "tasty"]
fruits = ["apple", "banana", "cherry"]

for x in adj:
  for y in fruits:
    print(x, y)
    
    