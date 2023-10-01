import numpy as np
from models.get_model import get_model
from data_reader.data_reader import get_data
import matplotlib.pyplot as plt
# Configurations are in a separate config.py file
import tensorflow as tf
from config import *
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

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
from keras import *




def ploting(w_eval):

    model_eval = get_model(model_name)

    
    model_eval.create_graph(learning_rate=step_size)
    
    model_eval.assign_flattened_weight(w_eval)
    
    total_data = 2000
    n_train = 1500
    n_test = 500
    time_dimension = 60
    n_embeddings = 50
    x_train, y_train, x_test, y_test, _ = get_data(dataset, total_data)
   
    y_pred = model_eval.predict(x_test)
    model_eval.summary()
   
    plt.plot(np.arange(time_dimension,n_train), y_train.reshape(-1,1), label='Actual Train')
    plt.plot(np.arange(n_train,n_train+n_test), y_test,  label='Actual Test')
    plt.plot(np.arange(n_train,n_train+n_test),y_pred, label='Predicted add')
    #plt.plot(np.arange(n_train,n_train+n_test),y_pred_lite, label='tflite Predicted')
    #plt.plot(np.arange(n_train,n_train+n_test),y_pred_n, label='Predictedtflite')
    plt.title('CPU Consumption')
    plt.xlabel('Time Points')
    plt.ylabel('CPU Consumption')
    plt.legend()
    plt.show()
    
    #model_eval.save('keras_model.h5')
    #print("Saved model to disk")
    #del model_eval
    # load model
    #model_eval = load_model('keras_model.h5')
    
    # serialize model to JSON
    #model_json = model_eval.to_json()
    #with open("model.json", "w") as json_file:
    #     json_file.write(model_json)
    # serialize weights to HDF5
    #model_eval.save_weights("model.h5")
    #print("Saved model to disk")
 
    # later...
 
    # load json and create model
    #json_file = open('model.json', 'r')
    #loaded_model_json = json_file.read()
    #json_file.close()
    #model_eval = model_from_json(loaded_model_json)
    # load weights into new model
    #model_eval.load_weights("model.h5")
    #print("Loaded model from disk")
    
    
    
    
    # summarize model.
    model_eval.summary()
    #model_eval = keras.models.load_model('keras_model.h5')
    #tf.saved_model.save(model_eval, "saved_model_keras_dir")
    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model_eval)
    converter.allow_custom_ops=True
    converter.experimental_new_converter = True
    tflite_model = converter.convert()


    #print("TFlite model: ", tflite_model)
    # Save the model.
    with open('model.tflite', 'wb') as f:
             f.write(tflite_model)
  
  
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
   
    y = np.zeros(x_test.shape[0])

    for ix in range(x_test.shape[0]):
            x_=x_test[ix]  
            x_ = np.expand_dims(x_, axis=0)
            x_ = np.array(x_, dtype=np.float32)
            #print("ix,x_shape:",ix,x_.shape)
            
            interpreter.set_tensor(input_details[0]['index'], x_)
           
            interpreter.invoke()
            y_ = interpreter.get_tensor(output_details[0]['index'])
            y[ix] = y_
    
      
   
    
   
    
   
    
   
    
   
    # Create the tf lite version of the model
    #model_eval.convert_tflite()
    # Make same predictions in TFlite 
    #y_pred_lite = model_eval.predict_lite(x_test)
    model_eval.summary()
    #print("tflite prediction values:", y_pred_lite)
    #print("keras prediction values:", y_pred)


    
    plt.plot(np.arange(time_dimension,n_train), y_train.reshape(-1,1), label='Actual Train')
    plt.plot(np.arange(n_train,n_train+n_test), y_test,  label='Actual Test')
    plt.plot(np.arange(n_train,n_train+n_test),y, label='tflite Predicted final')
    
    plt.title('CPU Consumption')
    plt.xlabel('Time Points')
    plt.ylabel('CPU Consumption')
    plt.legend()
    plt.show()
    
    