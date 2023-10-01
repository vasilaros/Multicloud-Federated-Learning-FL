import numpy as np
import matplotlib.pyplot as plt


n = 200
x0 = np.arange(n)
y = 5*x0+15

b = np.ones(x0.shape[0])
x = np.column_stack((x0,b))

a_0 = 10
a_1 = 2
alpha = 1e-4

epochs = 0
while(epochs < 100000):
    y_ = a_0 + a_1 * x0
    error = y_ - y
    mean_sq_er = np.sum(error**2)
    mean_sq_er = mean_sq_er/n
    mean_sq_er_sum = np.min([np.sum(error)/n, 100])
    a_0 = a_0 - alpha * 2 * mean_sq_er_sum
    a_1 = a_1 - alpha * 2 * mean_sq_er_sum/n
    epochs += 1
    if(epochs%10 == 0):
        print(mean_sq_er)


        
from sklearn.linear_model import LinearRegression
from data_reader.data_reader import get_data
import numpy as np
from config import *

X_train, Y_train, X_test, Y_test, train_label_orig =  get_data(dataset, 0)

x0 = np.arange(10000)
y = 5*x0+15

b = np.ones(x0.shape[0])
x = np.column_stack((x0,b))

lr = LinearRegression()  # create object for the class
lr.fit(x, y)  # perform linear regression

print(lr.coef_, lr.intercept_)