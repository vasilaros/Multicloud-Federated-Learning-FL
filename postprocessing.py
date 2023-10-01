import numpy as np
from models.get_model import get_model
from data_reader.data_reader import get_data
import matplotlib.pyplot as plt
# Configurations are in a separate config.py file
import tensorflow as tf
from config import *
from time import time
import os

#### Statistical definitions ####

def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))

def mse(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    differences = np.subtract(y_true, predictions)
    squared_differences = np.square(differences)
    return squared_differences.mean()

#def rmse(y_true, predictions):
#    return np.sqrt(((predictions - y_true) ** 2).mean())
#
    # Define root mean square loss function
def rmse(y_true, predictions):

    
        rmse = tf.sqrt(tf.keras.losses.mean_squared_error(y_true, predictions))
        return (sum(rmse)/len(rmse))

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


def postprocessing(w_eval):
    
    model_eval = get_model(model_name)

    
    model_eval.create_graph(learning_rate=step_size)
    
    model_eval.assign_flattened_weight(w_eval)
    
    total_data = 2000
    n_train = 1500
    n_test = 500
    time_dimension = 60
    n_embeddings = 50
    x_train, y_train, x_test, y_test, _ = get_data(dataset, total_data)
    decision=0.0
    
    decision= (0.15 * sizemodel) + (0.15 * timemodel) + (0.15 * resourcemodel) + (0.55 * accuracymodel)
    
    if (decision > 0.5):
    
      time_before_keras=time()
      y_pred = model_eval.predict(x_test)
      time_after_keras=time()
      total_keras_time=time_after_keras-time_before_keras
      print('execution time of keras:', total_keras_time)
     
      model_eval.summary()
      #model_eval.save('./saves')
      #print('weights saved')
      # save model
      #model_eval.save
      model_eval.save()
	  
      #with open('model.h5', 'wb') as f:
      #   f.write(model_eval)
      print('Model keras Saved!')
      #print('Model h5 Saved!')
      #print('Model h5 Saved!')
    
      print(f'mae initial: {mae(y_test,y_pred)}')
     
      print(f'mse initial: {mse(y_test,y_pred)}')
    
      print(f'rmse initial: {rmse(y_test,y_pred)}')
   
      #print(f'mse initial: {model_eval.mse(x_test,y_test)}')
    
      #print(f'rmse initial: {model_eval.rmse(x_test,y_test)}')
   
      print(f'mape initial: {mape(y_test,y_pred)}')
    
      #print(f'smape initial: {model_eval.smape(x_test,y_test)}')
    
      #print(f'smapee initial: {model_eval.smapee(x_test,y_test)}')
    
      #print(f'Precision initial: {model_eval.precision(x_test,y_test)}')
   
      #print(f'Recall initial: {model_eval.recall(x_test,y_test)}')
    
        
      plt.plot(np.arange(time_dimension,n_train), y_train.reshape(-1,1), label='Actual Train')
      plt.plot(np.arange(n_train,n_train+n_test), y_test,  label='Actual Test')
      plt.plot(np.arange(n_train,n_train+n_test),y_pred, label='Predicted')
   
      #plt.plot(np.arange(n_train,n_train+n_test),y_pred_lite, label='tflite Predicted')
      #plt.plot(np.arange(n_train,n_train+n_test),y_pred_n, label='Predictedtflite')
      plt.title('CPU Consumption')
      plt.xlabel('Time Points')
      plt.ylabel('CPU Consumption')
      plt.legend()
      plt.show()
      print('executed normal model of keras!')
   
      '''
      print('w_eval')
      print(w_eval)

      print(y_test)
      print('---')
      print(y_pred)
      print('len')
      print(len(y_pred))
      print('len')
      print(len(y_pred))
      '''
      
    else:
   
    
      # Create the tf lite version of the model, added y_train
      #model_eval.convert_tflite(y_train)
      model_eval.convert_tflite()
    
      # Make same predictions in TFlite 
      y_pred_lite = model_eval.predict_lite(x_test)
    

    
    
      model_eval.summary()
    
    
      #with open('model.tflite', 'wb') as f:
      #    f.write(self.tflite_model)
      #print('Model tflite Saved!')
        
      #model_eval.save_weights('./saves')
      #print('tflite weights saved')
    
      #print("tflite prediction values:", y_pred_lite)
      #print("keras prediction values:", y_pred)


    
      plt.plot(np.arange(time_dimension,n_train), y_train.reshape(-1,1), label='Actual Train')
      plt.plot(np.arange(n_train,n_train+n_test), y_test,  label='Actual Test')
      plt.plot(np.arange(n_train,n_train+n_test), y_pred_lite, label='tflite Predicted')
    
      plt.title('CPU Consumption')
      plt.xlabel('Time Points')
      plt.ylabel('CPU Consumption')
      plt.legend()
      plt.show()
    
      '''
      print('w_eval')
      print(w_eval)
      
      print(y_test)
      print('---')
      print(y_pred_lite)
      print('len')
      print(len(y_pred_lite))
      '''

      tensor = tf.convert_to_tensor(y_pred_lite)
      new_shape = tf.reshape(tensor, [500,1])
      #print(new_shape)

      '''
      print(f'mae tflite: {mae(y_test,y_pred_lite)}')
    
      print(f'mse tflite: {mse(y_test,y_pred_lite)}')
    
      print(f'rmse tflite: {rmse(y_test,y_pred_lite)}')
      #print(f'mse tflite: {model_eval.mselite(x_test,y_test)}')
      #print(f'rmse tflite: {model_eval.rmselite(x_test,y_test)}')
      print(f'mape tflite: {mape(y_test,y_pred_lite)}')
      '''

      print(f'mae tflite: {mae(y_test,new_shape)}')
      print(f'mse tflite: {mse(y_test,new_shape)}')
      print(f'rmse tflite: {rmse(y_test,new_shape)}')
      print(f'mape tflite: {mape(y_test,new_shape)}')

      #print(f'smape tflite: {model_eval.smapelite(x_test,y_test)}')
      #print(f'smapee tflite: {model_eval.smapeelite(x_test,y_test)}')
      #print(f'mase: {model_eval.maselite(x_test,y_test)}')
      #print(f'Precision tflite: {model_eval.precisionlite(x_test,y_test)}')
      #print(f'Recall initial: {model_eval.recalllite(x_test,y_test)}')
      print('executed Lite model of Edge!')