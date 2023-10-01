import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd



def linear_extract_samples(sample_list, is_train=True, file_path=os.path.dirname(__file__)):


    # data = pd.from_excel(file_path+'/linear/data.xlsx')
    data = pd.read_excel('datasets\\linear\\data.xlsx')
    X = data.X
    Y = data.Y

    if is_train: 
        X = X[:6000]
        Y = Y[:6000]
    else:
        X = X[-1000:]
        Y = Y[-1000:]
        
    return X, Y


def linear_extract( is_train=True, file_path=os.path.dirname(__file__)):
    # sample_list = range(start_sample_index, start_sample_index + num_samples)
    sample_list = 0 # dummy
    return linear_extract_samples(sample_list, is_train, file_path)