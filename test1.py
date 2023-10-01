# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 16:38:39 2021

@author: VasilAngelo
"""



# importing the library 
import tensorflow as tf 
  
# Initializing Input 
#value = tf.constant([1, 15, 10], dtype = tf.float64) 
  
# Printing the Input 
#print("Value: ", value) 
  
# Converting Tensor to TensorProto 
#proto = tf.make_tensor_proto(loss_local_last_global) 
  
# Generating numpy array 
res1 = tf.make_ndarray(tf.make_tensor_proto(loss_local_last_global) ) 
  
# Printing the resulting numpy array 
print("Result: ", res1)


 
  
# Generating numpy array 
res2 = tf.make_ndarray(tf.make_tensor_proto(loss_last_global)) 
  
# Printing the resulting numpy array 
print("Result: ", ress)

import numpy as np
# importing the library 
import tensorflow as tf 

print(loss_last_global.numpy())