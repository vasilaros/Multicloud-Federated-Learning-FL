import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
from tensorflow.contrib import rnn
from models.rnn_abstract import ModelRNNAbstract

time_dimension = 60
n_embeddings = 50

def rnn_layer(x, weights, biases):
    x_ = tf.unstack(x, time_dimension, 1)
    
    lstm_cell = rnn.BasicLSTMCell(n_embeddings, reuse=tf.AUTO_REUSE)
#    lstm_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_embeddings),rnn.BasicLSTMCell(n_embeddings)])
    outputs, _ = rnn.static_rnn(lstm_cell, x_, dtype=tf.float32)
    return tf.add(biases, tf.matmul(outputs[-1], weights))

class RNN(ModelRNNAbstract):
    def __init__(self):
        super().__init__()
        tf.reset_default_graph()
        pass

    def create_graph(self, learning_rate=None):
        self.x = tf.placeholder(tf.float32, shape=[None, time_dimension,1], name='X')
        self.y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y')
        self.weights = tf.Variable(tf.truncated_normal([n_embeddings, 1], mean=0.0, stddev=1.0), name="weights")
        self.biases = tf.Variable(tf.zeros([1]), name="bias")
        self.y = rnn_layer(self.x, self.weights, self.biases)
        
        
        self.mse = tf.reduce_mean(tf.square(self.y - self.y_))

        self.all_weights = [self.weights, self.biases]

        self._assignment_init()

        self._optimizer_init(learning_rate=learning_rate)
        self.grad = self.optimizer.compute_gradients(self.mse, var_list=self.all_weights)

        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.acc = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self._session_init()
        self.graph_created = True