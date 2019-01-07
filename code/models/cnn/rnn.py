mport numpy as np
import theano
import theano.tensor as T

from softmax import *
from lstm import *
from logistic import *
from updates import *
from utils_pg import *

class RNN(object):
    def __init__(self, in_size, out_size, hidden_size,
                 cell = "lstm", optimizer = "rmsprop", p = 0.5):
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
        #self.define_layers()
        self.define_train_test_funcs()
        

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
            # if self.cell == "gru":
            #     hidden_layer = GRULayer(rng, str(i), shape, layer_input,
            #                             self.maskX, self.is_train, self.batch_size, self.drop_rate)
            #elif self.cell == "lstm":
            hidden_layer = LSTMLayer(rng, str(i), shape, layer_input,
                                         self.maskX, self.is_train, self.batch_size, self.drop_rate)
            self.layers.append(hidden_layer)
            self.params += hidden_layer.params
        # output layer
        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        # self.logRegressionLayer = LogisticRegression(
        #     input=self.hiddenLayer.output,
        #     n_in=n_hidden,
        #     n_out=n_out)
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        # self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
         # same holds for the function computing the number of errors
        # self.errors = self.logRegressionLayer.errors
         # the parameters of the model are the parameters of the two layer it is
         # made out of
        # self.params = self.hiddenLayer.params + self.logRegressionLayer.params


        output_layer = SoftmaxLayer((hidden_layer.out_size, self.out_size),
                                    hidden_layer.activation, self.batch_size)
        self.layers.append(output_layer)
        self.params += output_layer.params
        self.epsilon = 1.0e-15
        
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
        self.p_y_given_x = T.nnet.softmax(T.dot(X, self.W) + self.b)
        self.activation = T.reshape(a, a_shape)self.W) + self.b)
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean((T.log(self.p_y_given_x) * w)[T.arange(y.shape[0]), y]) 

class LogisticLayer(object):
    def __init__(self, layer_id, n_in, n_out):
        prefix = "Logistic_"
        layer_id = "_" + layer_id
        self.in_size, self.out_size = shape
        self.W = init_weights(shape, prefix + "W" + layer_id)
        self.b = init_bias(self.out_size, prefix + "b" + layer_id)
        self.X = X
        self.activation = T.nnet.sigmoid(T.dot(self.X, self.W) + self.b)
        self.params = [self.W, self.b]

        self.p_y_x = T.nnet.softmax(T.dot(x, self.W) + self.b)

        # Symbolic expression for computing digit prediction
        self.y_pred = T.argmax(self.p_y_x, axis=1)

        # Model parameters
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

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

class LSTMLayer(object):
    def __init__(self, rng, layer_id, shape, X, mask, is_train = 1, batch_size = 1, p = 0.5):
        prefix = "LSTM_"
        layer_id = "_" + layer_id
        self.in_size, self.out_size = shape
        
        self.W_x_ifoc = init_weights_4((self.in_size, self.out_size), prefix + "W_x_ifoc" + layer_id, sample = "xavier")
        self.W_h_ifoc = init_weights_4((self.out_size, self.out_size), prefix + "W_h_ifoc" + layer_id, sample = "ortho")
        self.b_ifoc = init_bias(self.out_size * 4, prefix + "b_ifoc" + layer_id)

        self.params = [self.W_x_ifoc, self.W_h_ifoc,self.b_ifoc]

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim : (n + 1) * dim]
            return _x[:, n * dim : (n + 1) * dim]

        X_4ifoc = T.dot(X, self.W_x_ifoc) + self.b_ifoc
        def _active(m, x_4ifoc, pre_h, pre_c, W_h_ifoc):
            ifoc_preact = x_4ifoc + T.dot(pre_h, W_h_ifoc)
            
            i = T.nnet.sigmoid(_slice(ifoc_preact, 0, self.out_size))
            f = T.nnet.sigmoid(_slice(ifoc_preact, 1, self.out_size))
            o = T.nnet.sigmoid(_slice(ifoc_preact, 2, self.out_size))
            gc = T.tanh(_slice(ifoc_preact, 3, self.out_size))
            c = f * pre_c + i * gc
            h = o * T.tanh(c)

            c = c * m[:, None]
            h = h * m[:, None]
            return h, c
        [h, c], updates = theano.scan(_active,
                                      sequences = [mask, X_4ifoc],
                                      outputs_info = [T.alloc(floatX(0.), batch_size, self.out_size),
                                                      T.alloc(floatX(0.), batch_size, self.out_size)],
                                      non_sequences = [self.W_h_ifoc],
                                      strict = True)
        # dropout
        if p > 0:
            srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
            drop_mask = srng.binomial(n = 1, p = 1-p, size = h.shape, dtype = theano.config.floatX)
            self.activation = T.switch(T.eq(is_train, 1), h * drop_mask, h * (1 - p))
        else:
            self.activation = T.switch(T.eq(is_train, 1), h, h)
 