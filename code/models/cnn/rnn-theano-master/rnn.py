#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T

from softmax import *
from gru import *
from lstm import *
from logistic import *
from updates import *

class RNN(object):
    def __init__(self, in_size, out_size, hidden_size,
                 cell = "gru", optimizer = "rmsprop", p = 0.5):
        self.X = T.tensor3("X")
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.cell = cell
        self.drop_rate = p
        self.is_train = T.iscalar('is_train') # for dropout
        self.batch_size = T.iscalar('batch_size') # for mini-batch training
        self.maskX = T.matrix("maskX")
        self.maskY = T.matrix("maskY")
        self.optimizer = optimizer
        self.define_layers()
        self.define_train_test_funcs()
        
    def define_layers(self):
        self.layers = []
        self.params = []
        rng = np.random.RandomState(1234)
        # hidden layers
        for i in xrange(len(self.hidden_size)):
            if i == 0:
                layer_input = self.X
                shape = (self.in_size, self.hidden_size[0])
            else:
                layer_input = self.layers[i - 1].activation
                shape = (self.hidden_size[i - 1], self.hidden_size[i])

            
            hidden_layer = LSTMLayer(rng, str(i), shape, layer_input,
                                         self.maskX, self.is_train, self.batch_size, self.drop_rate)
            
            self.layers.append(hidden_layer)
            self.params += hidden_layer.params

        # output layer
        output_layer = SoftmaxLayer((hidden_layer.out_size, self.out_size),
                                    hidden_layer.activation, self.batch_size)
        self.layers.append(output_layer)
        self.params += output_layer.params
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b] # parameters of the model
   
    # https://github.com/fchollet/keras/pull/9/files
        self.epsilon = 1.0e-15


    def errors(self, y):
        """Return a float representing the number of errors in the minibatch ;
    zero one loss over the size of the minibatch
    :type y: theano.tensor.TensorType
    :param y: corresponds to a vector that gives for each example the
    correct label
    """
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def categorical_crossentropy(self, y_pred, y_true):
        y_pred = T.clip(y_pred, self.epsilon, 1.0 - self.epsilon)
        m = T.reshape(self.maskY, (self.maskY.shape[0] * self.batch_size, 1))
        ce = T.nnet.categorical_crossentropy(y_pred, y_true)
        ce = T.reshape(ce, (self.maskY.shape[0] * self.batch_size, 1))
        return T.sum(ce * m) / T.sum(m)

    def define_train_test_funcs(self):
        activation = self.layers[len(self.layers) - 1].activation
        self.Y = T.tensor3("Y")
        pYs = T.reshape(activation, (self.maskY.shape[0] * self.batch_size, self.out_size))
        tYs =  T.reshape(self.Y, (self.maskY.shape[0] * self.batch_size, self.out_size))
        cost = self.categorical_crossentropy(pYs, tYs)
        
        gparams = []
        for param in self.params:
            #gparam = T.grad(cost, param)
            gparam = T.clip(T.grad(cost, param), -10, 10)
            gparams.append(gparam)

        lr = T.scalar("lr")
        # eval(): string to function
        optimizer = eval(self.optimizer)
        updates = optimizer(self.params, gparams, lr)
        
        self.train = theano.function(inputs = [self.X, self.maskX, self.Y, self.maskY, lr, self.batch_size],
                                               givens = {self.is_train : np.cast['int32'](1)},
                                               outputs = cost,
                                               updates = updates)
        self.predict = theano.function(inputs = [self.X, self.maskX, self.batch_size],
                                                 givens = {self.is_train : np.cast['int32'](0)},
                                                 outputs = activation)
  
        #theano.printing.pydotprint(self.train, outfile="./model/train.png", var_with_name_simple=True) 
