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


################################################
step_size = 1e-3
model = get_model(model_name)
if hasattr(model, 'create_graph'):
    model.create_graph(learning_rate=step_size)
################################################

path = os.path.join(os.getcwd(),'w_2_metric')
files = os.listdir(path)
csv_files = [f for f in files if f.split('.')[-1] == 'csv' ]

# Read the csv files in pandas

def read_csv(fname, path=os.getcwd()):
    '''
    Reads a csv file and returns a pandas dataframe
    '''
    df = pd.read_csv(os.path.join(path,fname), header=None)
    df.columns = [fname.split('.')[0]+str(i) for i in df.columns]
    return df

df_total = None
for f in csv_files:
    df = read_csv(f, path='w_2_metric')
    if df_total is None:
        df_total = df
    else:
        df_total = df_total.join(df)

#from sklearn.model_selection import train_test_split
data = df_total.abrupt1
data = data/data.max()

n_train = 1500
n_test = 500
time_dimension = 60
n_embeddings = 50

x_train = data[:time_dimension]
y_train = data[time_dimension]
for i in range(time_dimension+1, n_train):
    x_train = np.vstack((x_train, data[i-time_dimension:i]))
    y_train = np.vstack((y_train, data[i]))

x_train = x_train[:,:, np.newaxis]

x_test = data[n_train:n_train+time_dimension]
y_test = data[n_train+time_dimension]
for i in range(n_train+time_dimension+1, n_train+n_test):
    x_test = np.vstack((x_test, data[i-time_dimension:i]))
    y_test = np.vstack((y_test, data[i]))

x_test = x_test[:,:, np.newaxis]

############################################################
epochs = 40
step_size = 1e-1

batch_size = 32
w = model.get_init_weight(0)

for e in tqdm.tqdm(range(epochs)):
    j = 1
    while (j < (n_train-time_dimension)//batch_size):
        train_indices = np.arange((j-1)*batch_size, j*batch_size)
        grad = model.gradient(x_train, y_train, w, train_indices)
        w = w - step_size * grad
        j += 1
    
y_pred = model.predict(x_test, w)

plt.plot(y_test)
plt.plot(y_pred)