import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from numpy import linalg


class ModelLinear:
    def __init__(self):
        self.w = [0,0]

    def set_w(self, w):
        self.w = w
        
    def calc(self, x):
        return np.inner(self.w, x)

    def get_weight_dimension(self, imgs, labels):

        return len(imgs[0])  # Assuming all images have the same size

    def get_init_weight(self, dim, rand_seed=None):
        return np.zeros(dim)

    # labels should be 1 or -1
    def gradient(self, imgs, labels, w, sample_indices):
        # https://towardsdatascience.com/introduction-to-machine-learning-algorithms-linear-regression-14c4e325882a
        self.w = w
        self.loss_hist = 0

        grad = np.zeros(len(w))
        if sample_indices is None:
            sample_indices = range(0, len(labels))
        
        self.loss_hist = self.loss(imgs, labels, w, sample_indices)

        
        for i in sample_indices:
            grad = grad + (np.inner(w, imgs[i]) - labels[i])*imgs[i]
                
        grad = 2/len(labels)*grad
        return grad

    def loss(self, imgs, labels, w, sample_indices = None):

        if sample_indices is None:
            sample_indices = range(0, len(labels))
        val = 0
        for i in sample_indices:
            val = val + 0.5*pow(linalg.norm(labels[i] - np.inner(w, imgs[i])), 2)

        return val

    def loss_from_prev_gradient_computation(self): # to check
        if (self.loss_hist is None) or (self.w is None):
            raise Exception('No previous gradient computation exists')

        # grad = np.zeros(len(self.w))
        # for i in range(0, len(self.loss_hist)):
        #     loss_ = self.loss_hist[i]
        #     grad = grad + loss_
                
        # grad = 2/len(self.loss_hist)*grad
        # return grad
    
        ## temp implementation
        
        return self.loss_hist
    
    

    def accuracy(self, imgs, labels, w):
        val = 0
        for i in range(1, len(labels)):
            if labels[i] * np.inner(w, imgs[i]) > 0:
                val += 1
        val /= len(labels)

        return val