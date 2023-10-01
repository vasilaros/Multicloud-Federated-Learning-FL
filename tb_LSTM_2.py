import numpy as np
from control_algorithm.adaptive_tau import ControlAlgAdaptiveTauServer
from data_reader.data_reader import get_data
from models.get_model import get_model
from statistic.collect_stat import CollectStatistics
from util.utils import send_msg, recv_msg, get_indices_each_node_case
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
# Configurations are in a separate config.py file
from config import *


x_train, y_train, x_test, y_test, _ = get_data(dataset, total_data)

################################################
step_size = 1e-1
model = get_model(model_name)
if hasattr(model, 'create_graph'):
    model.create_graph(learning_rate=step_size)
################################################

n_train = 1500
n_test = 500
time_dimension = 60
n_embeddings = 50

############################################################
epochs = 1

batch_size = 8
w = model.get_init_weight(0)

model.assign_flattened_weight(w)

x_ = x_test[:10]
y_ = y_test[:10]

yp = model.predict(x_)
model.loss(x_, y_)
grad1 = model.gradient_non_flattened(x_, y_)
w1 = model.get_init_weight(0)
grad = model.gradient(x_,y_,w1)


for e in tqdm.tqdm(range(epochs)):
    j = 1
    while (j < (n_train-time_dimension)//batch_size):
        train_indices = np.arange((j-1)*batch_size, j*batch_size)
        grad = model.gradient(x_train, y_train, w, train_indices)
        w = w - step_size * grad
        j += 1
        
       
y_pred = model.predict(x_test)


print(f'mae: {model.mae(x_test,y_test)}')
print(f'mse: {model.mse(x_test,y_test)}')
print(f'rmse: {model.rmse(x_test,y_test)}')
print(f'mape: {model.mape(x_test,y_test)}')
print(f'smape: {model.smape(x_test,y_test)}')
print(f'smapee: {model.smapee(x_test,y_test)}')
print(f'precision: {model.precision(x_test,y_test)}')
print(f'recall: {model.recall(x_test,y_test)}')
#print(f'mase: {model.mase(x_test,y_test)}')




#print(f'Precision: {model.precision(x_test,y_test)}')
print(f'Loss: {model.loss(x_test,y_test)}')
#print(f'Recall: {model.recall(x_test,y_test)}')

plt.plot(np.arange(time_dimension,n_train), y_train.reshape(-1,1), label='Actual Train')
plt.plot(np.arange(n_train,n_train+n_test), y_test,  label='Actual Test')
plt.plot(np.arange(n_train,n_train+n_test),y_pred, label='Predicted')
plt.title('CPU Consumption')
plt.xlabel('Time Points')
plt.ylabel('CPU Consumption')
plt.legend()
plt.show()