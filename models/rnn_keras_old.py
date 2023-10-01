# -*- coding: utf-8 -*-
"""
Created on Sat May  7 18:54:05 2022

@author: VasilAngelo
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

from time import time
import os

#added wih marios


from models.get_model import get_model
from data_reader.data_reader import get_data
from config import *

# Import Keras backend
import keras.backend as K

import numpy as np

time_dimension = 60
n_embeddings = 50

class RNN(ModelRNNAbstract):
    def __init__(self):
        super().__init__()
        pass

    def create_graph(self, learning_rate=None):
        # Model definition
        self.model = Sequential()         
        self.model.add(LSTM(units=n_embeddings,return_sequences=False, input_shape=(time_dimension, 1)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units = 1))
        self.all_weights = np.array(self.model.get_weights())

        self.opt = Adam(learning_rate=learning_rate)
        
        self.model.compile(optimizer = self.opt, loss = 'mean_squared_error')
        
        #self.grad = self.optimizer.compute_gradients(self.mse, var_list=self.all_weights)
        self.graph_created = True
        #self.model.save('model_keras.h5')
       
        self.model.summary()
        
    def get_weight_dimension(self, imgs=None, labels=None):    
        return self.model.count_params()
    
    def get_init_weight_non_flattened(self, dim, rand_seed=None):
        return np.array(self.model.get_weights())
    
    def get_init_weight(self, dim, rand_seed=None):
        if not self.graph_created:
            raise Exception('Graph is not created. Call create_graph() first.')
        weight_flatten_list = []
        for weight in (self.model.get_weights()):
            weight_flatten_list.append(np.reshape(weight, weight.size))
        weight_flatten_array = np.hstack(weight_flatten_list)
        return np.array(weight_flatten_array)  
    

    #new added function
    def gradient_non_flattened(self, x, y, w, sample_indices):
        #if sample_indices is None:
        #   sample_indices = [i for i in range( len(y))]
        #with tf.GradientTape() as tape:
        with tf.GradientTape() as tape:    
            loss_ = self.loss(x[sample_indices], y[sample_indices])
        grads = tape.gradient(loss_, self.model.trainable_weights)
        return grads
    
    

   
   #new added function
    def gradient(self, x, y, w, sample_indices): 
         self.assign_flattened_weight(w)
   #    if sample_indices is None: 
   #        sample_indices = [i for i in range( len(y))]
         with tf.GradientTape() as tape:
             loss_ = self.loss(x, y, w, sample_indices)
         grads = tape.gradient(loss_, self.model.trainable_weights)
         grad_flatten_list = []
         for g in (grads):
             g = np.array(g)
             grad_flatten_list.append(np.reshape(g, g.size))
         grad_flatten_list = np.hstack(grad_flatten_list)
         return grad_flatten_list
   
   
   
    def assign_flattened_weight(self, w):
        start_index = 0
        weights_list = []
        for k in range(0, len(self.all_weights)):
            weight = self.all_weights[k]

            tmp = 1
            l = weight.shape
            for i in range(0, len(l)):
                tmp *= l[i]

            weight_var = np.reshape(w[start_index : start_index+tmp], l)
            weights_list.append(weight_var)
            del weight_var
            start_index = start_index + tmp
        self.model.set_weights(weights_list)
           
    def start_consecutive_training(self, w_init):
        if not self.graph_created:
            raise Exception('Graph is not created. Call create_graph() first.')
        self.assign_flattened_weight(w_init)   
        
    def end_consecutive_training_and_get_weights(self):
        weight_flatten_list = []
        for weight in self.all_weights:
            weight_var = weight
            weight_flatten_list.append(np.reshape(weight_var, weight_var.size))

        weight_flatten_array = np.hstack(weight_flatten_list)

        return weight_flatten_array    
        
    # Define loss function as MSE    
    #def loss(self, x, y, w=None, sample_indices=None): Vasilis Removed
    def loss(self, x, y, w, sample_indices):
        #if sample_indices is None:
        #    sample_indices = [i for i in range( len(y))]
        y_pred = self.predict(x[sample_indices])
        print(len(y_pred))
        return tf.reduce_mean(tf.square(y[sample_indices]-y_pred))
    
    # Define mean absolute error loss function  --- Stefanidis Removed  
    #def mae(self, x, y, w=None, sample_indices=None):
    #    if sample_indices is None:
    #        sample_indices = [i for i in range( len(y))]
    #    y_pred = self.predict(x)
    #
    #    mae = tf.keras.losses.mean_absolute_error(y, y_pred)   
    #    return (sum(mae)/len(mae))
    def mae(self, x, y, w=None, sample_indices=None):
        if sample_indices is None:
            sample_indices = [i for i in range( len(y))]
        y_pred = self.predict(x[sample_indices])
    
        mae = tf.keras.losses.mean_absolute_error(y[sample_indices], y_pred)   
        return (sum(mae)/len(mae))
    
    def maelite(self, x, y, w=None, sample_indices=None):
        
        #self.convert_tflite()
        if sample_indices is None:
            sample_indices = [i for i in range( len(y))]
        
        y_pred_lite = self.predict_lite(x[sample_indices])
    
        mae = tf.keras.losses.mean_absolute_error(y[sample_indices], y_pred_lite)   
        return (sum(mae)/len(mae))
    
    # Define mean square error loss function
    def mse(self, x, y, w=None, sample_indices=None):
        if sample_indices is None:
            sample_indices = [i for i in range( len(y))]
        y_pred = self.predict(x[sample_indices])
    
        mse = tf.keras.losses.mean_squared_error(y[sample_indices], y_pred)
        return (sum(mse)/len(mse))
    
    
        # Define mean square error loss function
    def mselite(self, x, y, w=None, sample_indices=None):
        if sample_indices is None:
            sample_indices = [i for i in range( len(y))]
        y_pred_lite = self.predict_lite(x[sample_indices])
    
        mse = tf.keras.losses.mean_squared_error(y[sample_indices], y_pred_lite)
        return (sum(mse)/len(mse))
    
    # Define root mean square loss function
    def rmse(self, x, y, w=None, sample_indices=None):
        if sample_indices is None:
            sample_indices = [i for i in range( len(y))]
        y_pred = self.predict(x[sample_indices])
    
        rmse = tf.sqrt(tf.keras.losses.mean_squared_error(y[sample_indices], y_pred))
        return (sum(rmse)/len(rmse))
    
    
        # Define root mean square loss function
    def rmselite(self, x, y, w=None, sample_indices=None):
        if sample_indices is None:
            sample_indices = [i for i in range( len(y))]
        y_pred_lite = self.predict_lite(x[sample_indices])
    
        rmse = tf.sqrt(tf.keras.losses.mean_squared_error(y[sample_indices], y_pred_lite))
        return (sum(rmse)/len(rmse))


    # Define MAPE loss function
    def mape(self, x, y, w=None, sample_indices=None):
        if sample_indices is None:
            sample_indices = [i for i in range( len(y))]
        y_pred = self.predict(x[sample_indices])
        
        mape = tf.keras.losses.mean_absolute_percentage_error(y[sample_indices], y_pred)
        return (sum(mape)/len(mape))
    
    
    # Define MAPE loss function
    def mapelite(self, x, y, w=None, sample_indices=None):
        if sample_indices is None:
            sample_indices = [i for i in range( len(y))]
        y_pred_lite = self.predict_lite(x[sample_indices])
        
        mape = tf.keras.losses.mean_absolute_percentage_error(y[sample_indices], y_pred_lite)
        return (sum(mape)/len(mape))

    
    # Define SMAPE loss function just adds epsilon value for some kind of regularization
    def smape(self, x, y, w=None, sample_indices=None,epsilon=0.1):
        if sample_indices is None:
            sample_indices = [i for i in range( len(y))]
        y_pred = self.predict(x[sample_indices])
             
        summ = K.maximum(K.abs(tf.cast(y[sample_indices],tf.float32)) + K.abs(tf.cast(y_pred,tf.float32)) + tf.cast(epsilon,tf.float32), 0.5 + tf.cast(epsilon,tf.float32))
        smape = (K.abs(tf.cast(y_pred,tf.float32) - tf.cast(y[sample_indices],tf.float32)) / summ) * 2.0
        return (sum(smape)/len(smape))
   
    # Define SMAPE loss function just adds epsilon value for some kind of regularization
    def smapelite(self, x, y, w=None, sample_indices=None,epsilon=0.1):
        if sample_indices is None:
            sample_indices = [i for i in range( len(y))]
        y_pred_lite = self.predict_lite(x[sample_indices])
             
        summ = K.maximum(K.abs(tf.cast(y[sample_indices],tf.float32)) + K.abs(tf.cast(y_pred_lite,tf.float32)) + tf.cast(epsilon,tf.float32), 0.5 + tf.cast(epsilon,tf.float32))
        smape = (K.abs(tf.cast(y_pred_lite,tf.float32) - tf.cast(y[sample_indices],tf.float32)) / summ) * 2.0
        return (sum(smape)/len(smape))
    
        # Define SMAPE loss function quick definition
    def smapee(self, x, y, w=None, sample_indices=None):
        if sample_indices is None:
            sample_indices = [i for i in range( len(y))]
        y_pred = self.predict(x[sample_indices])
        
        smapee = 100/len(y[sample_indices]) * np.sum(2 * np.abs(y_pred - y[sample_indices]) / (np.abs(y[sample_indices]) + np.abs(y_pred)))


     
        return smapee
    
     # Define SMAPE loss function quick definition
    def smapeelite(self, x, y, w=None, sample_indices=None):
        if sample_indices is None:
            sample_indices = [i for i in range( len(y))]
        y_pred_lite = self.predict_lite(x[sample_indices])
        
        smapee = 100/len(y[sample_indices]) * np.sum(2 * np.abs(y_pred_lite - y[sample_indices]) / (np.abs(y[sample_indices]) + np.abs(y_pred_lite)))


     
        return smapee
    
     # Define MASE loss function 
    def mase(self, x, y, w=None, sample_indices=None):
        if sample_indices is None:
            sample_indices = [i for i in range( len(y))]
        y_pred = self.predict(x[sample_indices])
        
        denom = np.sum(np.abs(np.diff(y[sample_indices])))
        #norms = np.where(norms != 0 )
        nom = np.sum(np.abs(y[sample_indices] - y_pred))
        mase = ((len(y[sample_indices])-1)/len(y[sample_indices])) * (nom / denom)

        
        
        #summ = K.maximum(K.abs(tf.cast(y,tf.float32)) + K.abs(tf.cast(y_pred,tf.float32)) + tf.cast(epsilon,tf.float32), 0.5 + tf.cast(epsilon,tf.float32))
        #smape = (K.abs(tf.cast(y_pred,tf.float32) - tf.cast(y,tf.float32)) / summ) * 2.0
        #return (sum(smapee)/len(smapee))
     
        return mase  
    
     # Define MASE loss function 
    def maselite(self, x, y, w=None, sample_indices=None):
        if sample_indices is None:
            sample_indices = [i for i in range( len(y))]
        y_pred_lite = self.predict_lite(x[sample_indices])
        
        denom = np.sum(np.abs(np.diff(y[sample_indices])))
        #norms = np.where(norms != 0 )
        nom = np.sum(np.abs(y[sample_indices] - y_pred_lite))
        mase = ((len(y[sample_indices])-1)/len(y[sample_indices])) * (nom / denom)

        
        
        #summ = K.maximum(K.abs(tf.cast(y,tf.float32)) + K.abs(tf.cast(y_pred,tf.float32)) + tf.cast(epsilon,tf.float32), 0.5 + tf.cast(epsilon,tf.float32))
        #smape = (K.abs(tf.cast(y_pred,tf.float32) - tf.cast(y,tf.float32)) / summ) * 2.0
        #return (sum(smapee)/len(smapee))
     
        return mase  
    
    def losses(self, x, y, w=None, sample_indices=None):
        if sample_indices is None:
            sample_indices = range(0, len(y))
        y_pred = self.predict(x[sample_indices])
        return np.square(y[sample_indices]-y_pred)
    
    def predict(self, x):
        
        #self.model.save('model_keras.h5')
        
        return self.model(x)
    
    def save(self):
        
        self.model.save('model_keras.h5')
        
        
    
    def summary(self):
        
        #self.model.save('model_keras.h5')
        
        return self.model.summary()

    # Defining the representative dataset from training images.

    #def representative_data_gen():
    #   for input_value in tf.data.Dataset.from_tensor_slices(sample_indices).take(100):
    #         yield [input_value]
    def representative_dataset():
      for _ in range(100):
        data = np.random.rand(1, 244, 244, 3)
        yield [data.astype(np.float32)]
    

    def convert_tflite(self):
        '''
        TF Lite is saved in self.tflite_model
        '''
        
        self.model.save('model_keras.h5')
        print('Model keras Saved!')
        self.converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        self.converter.allow_custom_ops=True
        self.converter.experimental_new_converter = True
        #----self.converter._experimental_new_quantizer = True
        #---self.converter.target_spec.supported_ops = [
        #  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        #  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
        #]
        
        total_data = 2000
        n_train = 1500
        n_test = 500
        time_dimension = 60
        n_embeddings = 50
        x_train, y_train, x_test, y_test, _ = get_data(dataset, total_data)
        
        # Defining the representative dataset from training values.
        #--def representative_data_gen():
        #  for input_value in tf.data.Dataset.from_tensor_slices(y_train).take(100):
        #   yield [input_value]
        
        #---self.converter.representative_dataset = representative_data_gen
        #model quantization
        #--self.converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        #-------------self.converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        
        #def representative_dataset():
        # for _ in range(100):
        #    data = np.random.rand(1, 244, 244, 3)
        #    yield [data.astype(np.float32)]
            
        #def representative_dataset():
        #  for data in tf.data.Dataset.from_tensor_slices((y_train)).batch(1).take(100):
        #    yield [tf.dtypes.cast(data, tf.float32)] 
        
        #self.converter.optimizations = [tf.lite.Optimize.DEFAULT]
        #self.converter.representative_dataset = representative_dataset
        #---self.converter.target_spec.supported_types = [tf.float16]
        #self.converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
        # Using float-16 quantization.
        #---self.converter.optimizations = [tf.lite.Optimize.DEFAULT]
        #----self.converter.target_spec.supported_types = [tf.float16]
        # Defining the representative dataset from training images.
        #self.converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        #self.converter.inference_input_type = tf.uint8
        #self.converter.inference_output_type = tf.uint8
        
        #int 8 added

        #self.converter.optimizations = [tf.lite.Optimize.DEFAULT]
        #self.converter.representative_dataset = representative_data_gen

        # Using Integer Quantization.

        #self.converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        # Setting the input and output tensors to uint8.

        #self.converter.inference_input_type = tf.uint8

        #self.converter.inference_output_type = tf.uint8

        # end int 8 addition
        
        #coninue
        self.tflite_model = self.converter.convert()
        with open('model.tflite', 'wb') as f:
           f.write(self.tflite_model)
        #open("/model_h5_to_tflite.tflite", "wb").write(self.tflite_model)
        print('Model tflite Saved!')
        
        #with open('model.tflite', 'wb') as f:
        #   f.write(self.tflite_model)
        
        
        # Load the TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_content=self.tflite_model)
        self.interpreter.allocate_tensors()
        
        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        

        
    def predict_lite(self,x):
        '''
        input: x is a tensor
        In a tflite model a tensor cannot be applied directly to the model. 
        Each vector of the tensor is calculated in a loop.
        returns the output of the model as an array
        '''
        # Allocate the size for the output array
        y = np.zeros(x.shape[0])

        for ix in range(x.shape[0]):
            x_=x[ix]  
            x_ = np.expand_dims(x_, axis=0)
            x_ = np.array(x_, dtype=np.float32)
            #print("ix,x_shape:",ix,x_.shape)
            
            self.interpreter.set_tensor(self.input_details[0]['index'], x_)
            
            
            time_before_tflite=time()
            # here is the prediction time consumed         
            self.interpreter.invoke()
            time_after_tflite=time()
            total_tflite_time=time_after_tflite-time_before_tflite
            
            
            y_ = self.interpreter.get_tensor(self.output_details[0]['index'])
            y[ix] = y_
        
        print('execution time of tflite:', total_tflite_time)
        return y    

    
    def precision(self,x,y, w=None, sample_indices=None,thresh=0.05):
        if sample_indices is None:
            sample_indices = range(0, len(y))
        y_pred = self.predict(x[sample_indices])       
        hit =  np.mean(np.array((y[sample_indices]-y_pred)/y[sample_indices] < thresh).astype(int))
        return hit
    
    def precisionlite(self,x,y, w=None, sample_indices=None,thresh=0.05):
        if sample_indices is None:
            sample_indices = range(0, len(y))
        y_pred_lite = self.predict_lite(x[sample_indices])       
        hit =  np.mean(np.array((y[sample_indices]-y_pred_lite)/y[sample_indices] < thresh).astype(int))
        return hit
    
    #def recall(self,x,y, w=None, sample_indices=None,thresh=0.05):
    #    if sample_indices is None:
    #        sample_indices = range(0, len(y))
    #    y_pred = self.predict(x)       
    #    hitt =  np.mean(np.array((y-y_pred)/y_pred < thresh).astype(int))
    #    return hitt
    
    
    def recall(self,x,y, w=None, sample_indices=None,thresh=0.05):
        if sample_indices is None:
            sample_indices = range(0, len(y))
        y_pred = self.predict(x[sample_indices])       
        hitt =  np.mean(np.array((y[sample_indices]-y_pred)/y_pred < thresh).astype(int))
        return hitt
    
    
    def recalllite(self,x,y, w=None, sample_indices=None,thresh=0.05):
        if sample_indices is None:
            sample_indices = range(0, len(y))
        y_pred_lite = self.predict_lite(x[sample_indices])       
        hitt =  np.mean(np.array((y[sample_indices]-y_pred_lite)/y_pred_lite < thresh).astype(int))
        return hitt
    
#### Statistical definitions ####

def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))

def mse(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    differences = np.subtract(y_true, predictions)
    squared_differences = np.square(differences)
    return squared_differences.mean()

def rmse(y_true, predictions):
    return np.sqrt(((predictions - y_true) ** 2).mean())

def mape(y_true, predictions):
    mape = tf.keras.losses.mean_absolute_percentage_error(y_true, predictions)
    return (sum(mape)/len(mape))

def smapee(y_true, predictions):
    smapee = 100/len(y_true) * np.sum(2 * np.abs(predictions - y_true) / (np.abs(y_true) + np.abs(predictions)))
    return smapee

def mase(y_true, predictions):
    denom = np.sum(np.abs(np.diff(y_true)))
    nom = np.sum(np.abs(y_true - predictions))
    mase = ((len(y_true)-1)/len(y_true)) * (nom / denom) 
    return mase  

def precision(y_true, predictions,thresh=0.05):   
    hit = np.mean(np.array((y_true-predictions)/y_true < thresh).astype(int))
    return hit

def recall(y_true, predictions,thresh=0.05):
    hitt =  np.mean(np.array((y_true-predictions)/predictions < thresh).astype(int))
    return hitt
    