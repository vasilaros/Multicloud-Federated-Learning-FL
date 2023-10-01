import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow as tf
import abc

LOSS_ACC_BATCH_SIZE = 32  # When computing loss and accuracy, use blocks of LOSS_ACC_BATCH_SIZE

class ModelRNNAbstract(abc.ABC):
    def __init__(self):
        self.graph_created = False
        pass

    @abc.abstractmethod
    def create_graph(self, learning_rate=None):
        # The below variables need to be defined in the child class
        self.all_weights = None
        self.x = None
        self.y_ = None
        self.y = None
        self.model = None
        self.opt = None
        self.mse = None
        self.acc = None
        self.init = None
        self.all_assignment_placeholders = None
        self.all_assignment_operations = None
        self._optimizer_init(learning_rate=learning_rate)
        self.grad = None

        self.session = None  # Used for consecutive training

    def _optimizer_init(self, learning_rate=None):
        pass

    def _assignment_init(self):
        pass

    def _session_init(self):
        self.session = tf.Session()


    # def accuracy(self, x, y, w, sample_indices=None):
    #     if not self.graph_created:
    #         raise Exception('Graph is not created. Call create_graph() first.')

    #     self.assign_flattened_weight(w)

    #     if sample_indices is None:
    #         sample_indices = range(0, len(labels))

    #     val = 0
    #     l = []
    #     for k in range(0, len(sample_indices)):
    #         l.append(sample_indices[k])

    #         if len(l) >= LOSS_ACC_BATCH_SIZE or k == len(sample_indices) - 1:

    #             val += self.session.run(self.acc, feed_dict={self.x: [imgs[i] for i in l],
    #                                                          self.y_: [labels[i] for i in l]}) \
    #                    * float(len(l)) / len(sample_indices)

    #             l = []

    #     return val
    def accuracy(self, x, y, w, sample_indices=None):
        return 1

    def run_one_step_consecutive_training(self, imgs, labels, sample_indices):
        self.session.run(self.optimizer_op, feed_dict={self.x: [imgs[i] for i in sample_indices], self.y_: [labels[i] for i in sample_indices]})


