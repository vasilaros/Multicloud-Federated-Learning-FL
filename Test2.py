# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 22:18:22 2021

@author: VasilAngelo
"""

import numpy as np

n_nodes = 4
total_data = 1440
a = np.random.rand(n_nodes)
a = (a/np.sum(a)*total_data).astype(int)

sum_a = a.sum()
mean_a = a.mean()
std_a = np.std(a)
half=0
if std_a > mean_a:
    half=1
    std_a /=2
thresh = mean_a - 1*std_a

reject_nodes = (a < thresh).astype(int)
