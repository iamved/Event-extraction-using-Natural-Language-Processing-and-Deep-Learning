#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *

class SoftmaxLayer(object):
    def __init__(self, shape, X, batch_size = 1):
        prefix = "Softmax_"
        self.in_size, self.out_size = shape
        self.W = init_weights(shape, prefix + "W", sample = "xavier")
        self.b = init_bias(self.out_size, prefix + "b")
        self.params = [self.W, self.b]
      
        a = T.dot(X, self.W) + self.b
        a_shape = a.shape
        a = T.nnet.softmax(T.reshape(a, (a_shape[0] * a_shape[1], a_shape[2])))
        self.activation = T.reshape(a, a_shape)


        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b] # parameters of the model


    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

