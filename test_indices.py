# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 20:46:11 2020

@author: VasilAngelo
"""

import socket
import time

import numpy as np

#from control_algorithm.adaptive_tau import ControlAlgAdaptiveTauServer
from data_reader.data_reader import get_data
from models.get_model import get_model
from statistic.collect_stat import CollectStatistics
from util.utils import send_msg, recv_msg, get_indices_each_node_case

# Configurations are in a separate config.py file
from config import *

train_image, train_label, test_image, test_label, train_label_orig = get_data(dataset, total_data, dataset_file_path)


    # This function takes a long time to complete,
    # putting it outside of the sim loop because there is no randomness in the current way of computing the indices
    
indices_each_node_case = get_indices_each_node_case(n_nodes, MAX_CASE, train_label_orig)