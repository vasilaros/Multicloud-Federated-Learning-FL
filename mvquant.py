#import tensorflow_model_optimization as tfmot
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential

from config import *
from postprocessing import postprocessing
from models.get_model import get_model
from data_reader.data_reader import get_data

#quantize_model = tfmot.quantization.keras.quantize_model

model = get_model(model_name)
model.create_graph(learning_rate=step_size)

total_data = 2000
x_train, y_train, x_test, y_test, _ = get_data(dataset, total_data)
dim_w = model.get_weight_dimension(x_train, y_train)
w_global_init = model.get_init_weight(dim_w, rand_seed=0)
print(w_global_init)

postprocessing(w_global_init)
