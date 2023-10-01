import numpy as np
from control_algorithm.adaptive_tau import ControlAlgAdaptiveTauServer
from data_reader.data_reader import get_data
from models.get_model import get_model
from statistic.collect_stat import CollectStatistics
from util.utils import send_msg, recv_msg, get_indices_each_node_case


# Configurations are in a separate config.py file
from config import *

model = ModelLinear()
X_train, Y_train, X_test, Y_test, train_label_orig =  get_data(dataset, 0)

w_ = np.array([0, 1e-6])
model.gradient(X_train, Y_train, w_, None)

model.loss(X_train, Y_train, w_, None)
model.loss_from_prev_gradient_computation()


for i in range(100):
    w_ = w_ - 2e-10*model.gradient(X_test, Y_test, w_, np.arange(10000))
    if (i % 10 == 0);
        print(w_)
    
#model.loss(X_test, Y_test, w_, np.arange(10000))





x0 = np.arange(10000)
y = 1e-6*x0 + 1.25-6

b = np.ones(x0.shape[0])
x = np.column_stack((x0,b))



model.set_w([0,0])
model.calc(x)


w_ = np.array([0,0])
for i in range(1000):
    w_ = w_ - 1e-9*model.gradient(x, y, w_, x0)
    print(w_)
    
