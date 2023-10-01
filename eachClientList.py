# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 08:28:13 2021

@author: VasilAngelo
"""

#import numpy
import numpy as np
array = np.arange(20)

#using numpy.array_split() method

gfg = np.array_split (array, [1,6,9,11])
gfg2 = []


#how to use append and make a list of lists as requested in Salonidis utils module --for each client a specific dataset
for arr in gfg :
    gfg2.append (arr.tolist())
    
    