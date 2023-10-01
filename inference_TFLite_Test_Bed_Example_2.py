# -*- coding: utf-8 -*-
"""
Created on Sun May  1 18:32:39 2022

@author: VasilAngelo
"""



import tensorflow as tf
import numpy as np

# Create a model using high-level tf.keras.* APIs
class TestModel(tf.Module):
  def __init__(self):
    super(TestModel, self).__init__()

  @tf.function(input_signature=[tf.TensorSpec(shape=[1, 10], dtype=tf.float32)])
  def add(self, x):
    '''
    Simple method that accepts single input 'x' and returns 'x' + 4.
    '''
    # Name the output 'result' for convenience.
    return {'result' : x + 4}




# Save the model
model = TestModel()

print("keras model: ", model)
#model.summary() 
# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)

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
output_details = interpreter.get_output_details()


# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
#print("skata", model.tflite)
#tflite_model.summary() 