from __future__ import absolute_import

import numpy as np

from . import backend as K
from . import initializations


class constrained_loss(object):
    def __init__(self, loss_fn, constraint_fn, constraint_fn_args, init='zero', constraint_weight=1.0, **kwargs):
        """
        constrained loss function that accepts both a loss function and a contrainst function.  For exmaple, you
        could set the loss function to binary cross entropy and the constraint function to any custom theano
        function e.g. constraint_fn = lambda param,val,epsilon: K.T.nlinalg.trace(K.dot(param-val,param-val)) - epsilon
        which is equivelent to Frobenius norm computing the distance between param and val. Param could be a layer's
        weight matrix and val could be some specific value of that weight matrix.
        :param loss_fn: any theano/keras function computing loss between y_true and y_pred
        :param constraint_fn: any theano/keras fuction computing some constraint
        :param constraint_fn_args: list of args to constaint function
        :param init: initializer for the trainable lagrange multiplier
        :param constraint_weight: akin to a regularization parameter
        :param kwargs:
        """
        from functools import partial

        # lagrange multiplier is scalar so init function must be appropriate for scalar value
        parameterized_inits = {'uniform', 'normal'}
        valid_inits = {'zero', 'one', }.union(parameterized_inits)
        assert init in valid_inits, "init must be one of {valid_inits}, but you passed {init}".format(**locals())

        name = kwargs.get('name')
        if not name:
            prefix = self.__class__.__name__.lower()
            name = prefix + '_' + str(K.get_uid(prefix))
        self.__name__ = name

        self.init = initializations.get(init)
        if init in parameterized_inits:
            # uniform and normal are parameterized by scale argument. Keras defaults scale=0.05 so I did same
            scale = kwargs.get("init_scale", 0.05)
            self.init = partial(self.init, scale=scale)

        self.loss_fn = get(loss_fn)
        self.constraint_fn = constraint_fn
        self.constraint_fn_args = constraint_fn_args

        self.constraint_weight = K.variable(constraint_weight,
                                            name='{}__constraint_weight'.format(self.__class__.__name__),
                                            dtype=K.floatx())

        # fixme initializers unable to output scalar value
        self.lagrange_multiplier = K.variable(0., dtype='float32', name='lagrange_mult')
        # self.init((1,), name='{}__lagrange_multiplier'.format(self.__class__.__name__))

        self.trainable_weights = [self.lagrange_multiplier]

    def __call__(self, y_true, y_pred):
        constrained_loss = (self.loss_fn(y_true, y_pred) +
                            self.lagrange_multiplier *
                            (self.constraint_weight * self.constraint_fn(*self.constraint_fn_args)))
        return constrained_loss


#
# class constrained_norm_loss(object):
#     def __init__(self, loss_func, param, value, beta, epsilon):
#
#
#         self.__name__ = 'constrained_norm_loss'
#         self.loss_fn =  get(loss_func)
#         self.param_val = K.variable(value, dtype=K._floatx, name='constrained_loss:param_value')
#         self.param = param
#         self.beta = K.variable(beta, name='constrained_norm:beta', dtype=K._floatx)
#         self.epsilon = K.variable(epsilon, name='constrained_norm:epsilon', dtype=K._floatx)
#         self.lagrange_multiplier = K.variable(0., name='constrained_norm:lagrange_mult', dtype=K._floatx)
#         self.trainable_weights = [self.lagrange_multiplier]
#
#     def __call__(self, y_true, y_pred):
#         constrained_loss = (self.loss_fn(y_true, y_pred) +
#                             self.lagrange_multiplier * self.beta *
#                             (K.T.nlinalg.trace(K.dot(K.transpose(self.param_val - self.param),
#                                                      self.param_val - self.param)) - self.epsilon))
#         return constrained_loss


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)


def mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), np.inf))
    return 100. * K.mean(diff, axis=-1)


def mean_squared_logarithmic_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), np.inf) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), np.inf) + 1.)
    return K.mean(K.square(first_log - second_log), axis=-1)


def squared_hinge(y_true, y_pred):
    return K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.)), axis=-1)


def hinge(y_true, y_pred):
    return K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)


def categorical_crossentropy(y_true, y_pred):
    '''Expects a binary class matrix instead of a vector of scalar classes.
    '''
    return K.categorical_crossentropy(y_pred, y_true)


def sparse_categorical_crossentropy(y_true, y_pred):
    '''expects an array of integer classes.
    Note: labels shape must have the same number of dimensions as output shape.
    If you get a shape error, add a length-1 dimension to labels.
    '''
    return K.sparse_categorical_crossentropy(y_pred, y_true)


def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)


def kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)


def poisson(y_true, y_pred):
    return K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()), axis=-1)


def cosine_proximity(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -K.mean(y_true * y_pred, axis=-1)


# aliases
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
kld = KLD = kullback_leibler_divergence
cosine = cosine_proximity

from .utils.generic_utils import get_from_module


def get(identifier):
    return get_from_module(identifier, globals(), 'objective')
