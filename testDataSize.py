# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 23:33:20 2021

@author: VasilAngelo
"""
import numpy as np

n_nodes=5
indices_each_node_case=[]
total_data = 1500

samples = np.arange(total_data)
np.random.shuffle(samples)
# pick the points for splitting
splitting_points = np.sort(np.random.choice(samples, size=4))
indices_each_node_case = np.array_split(samples, splitting_points)
