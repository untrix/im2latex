#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
    Copyright 2017 Sumeet S Singh

    This file is part of im2latex solution by Sumeet S Singh.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the Affero GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    Affero GNU General Public License for more details.

    You should have received a copy of the Affero GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

Created on Sun Jul  9 11:44:46 2017
Tested on python 2.7

@author: Sumeet S Singh
"""
import numpy as np
from data_commons import logger
import dl_commons as dlc
import tensorflow as tf
from tensorflow.contrib.keras import backend as K
from dl_commons import (mandatory, equalto, boolean, HyperParams, PD, PDL, iscallable, iscallableOrNone,
                        issequenceof, issequenceofOrNone, integer, integerOrNone, decimal, decimalOrNone,
                        instanceofOrNone, instanceof)

class tensor(instanceof):
    """Tensor shape validator to go with ParamDesc"""
    def __init__(self, shape):
        self._shape = shape
        dlc._ParamValidator.__init__(self, tf.Tensor)
    def __contains__(self, obj):
        return instanceof.__contains__(self, obj) and  keras.backend.int_shape(obj) == self._shape


def summarize_layer(weights, biases, activations, coll_name):
    def summarize_vars(var, section_name):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        # mean = tf.reduce_mean(var)
        # tf.summary.scalar(section_name+'/mean', mean)
        # stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # tf.summary.scalar(section_name+'/stddev', stddev)
        # tf.summary.scalar(section_name+'/max', tf.reduce_max(var))
        # tf.summary.scalar(section_name+'/min', tf.reduce_min(var))
        tf.summary.histogram(section_name+'/histogram', var, collections=[coll_name] if (coll_name is not None) else None)

    with tf.variable_scope('Summaries'):
        if weights is not None:
            summarize_vars(weights, 'Weights')
        if biases is not None:
            summarize_vars(biases, 'Biases')
        if activations is not None:
            summarize_vars(activations, 'Activations')
        if state is not None:
            summarize_vars(state, 'State')

class TensorboardParams(HyperParams):
    proto = (
        PD('tb_logdir',
            'Top-level/Root logdir under which run-specific dirs are created. e.g. ./tb_metrics',
            instanceof(str),
            ),
        PD('logdir_tag',
            'Extra tag-name to attach to the run-specific  logdir name (after the date portion).',
            instanceofOrNone(str),
            ),
        PD('tb_weights',
            'Section name under which weight summaries show up on tensorboard',
            None,
            default='Weights'
            ),
        PD('tb_biases',
            'Section name under which bias summaries show up on tensorboard',
            None,
            default='Biases'
            ),
        PD('tb_activations',
            'Section name under which activation summaries show up on tensorboard',
            None,
            default='Activations'
            )
        )
    def __init__(self, initVals=None):
        HyperParams.__init__(self, self.proto, initVals)
    def __copy__(self):
        ## Shallow copy
        return self.__class__(self)

    def copy(self, override_vals={}):
        ## Shallow copy
        return self.__class__(self).updated(override_vals)

class DropoutParams(HyperParams):
    """ Dropout Layer descriptor """
    proto = (
        PD('keep_prob',
              'Probability of keeping an output (i.e. not dropping it).',
              decimal(),
              0.5),
        PD('seed',
              'Integer seed for the random number generator',
              integerOrNone(),
              None
              )
        )
    def __init__(self, initVals=None):
        HyperParams.__init__(self, self.proto, initVals)
    def __copy__(self):
        ## Shallow copy
        return self.__class__(self)

    def copy(self, override_vals={}):
        ## Shallow copy
        return self.__class__(self).updated(override_vals)

class CommonParams(HyperParams):
    proto = (
        PD('activation_fn',
              'The activation function to use. None signifies no activation function.',
              ## iscallable((tf.nn.relu, tf.nn.tanh, tf.nn.sigmoid, None)),
              iscallableOrNone(),
              ),
        PD('normalizer_fn',
              'Normalizer function to use instead of biases. If set, biases are not used.',
              iscallableOrNone(),
              ),
        PD('weights_initializer',
              'Tensorflow weights initializer function',
              iscallable(),
              # tf.contrib.layers.xavier_initializer()
              # tf.contrib.layers.variance_scaling_initializer()
              ),
        PD('biases_initializer',
              'Tensorflow biases initializer function, e.g. tf.zeros_initializer(). ',
              iscallable(),
              ## tf.zeros_initializer()
              ),
        PD('weights_regularizer',
              'L1 / L2 norm regularization',
              iscallableOrNone(),
              # tf.contrib.layers.l2_regularizer(scale, scope=None)
              # tf.contrib.layers.l1_regularizer(scale, scope=None)
              ),
        PD('biases_regularizer',
              'L1 / L2 norm regularization',
              iscallableOrNone()
              ),
#        PD('make_tb_metric',"(boolean): Create tensorboard metrics.",
#              boolean,
#              True),
        PD('dropout', 'Dropout parameters if any.',
              instanceofOrNone(DropoutParams)),
        # PD('weights_coll_name',
        #       'Name of trainable weights collection. There is no need to change the default = WEIGHTS',
        #       ('WEIGHTS',),
        #       'WEIGHTS'
        #       ),
        PD('tb', "Tensorboard Params.",
           instanceofOrNone(TensorboardParams))
        )

    def __init__(self, initVals=None):
        HyperParams.__init__(self, self.proto, initVals)
    def __copy__(self):
        ## Shallow copy
        return self.__class__(self)

    def copy(self, override_vals={}):
        ## Shallow copy
        return self.__class__(self).updated(override_vals)

CommonParamsProto = CommonParams().protoD

class FCLayerParams(HyperParams):
    proto = (
        PD('activation_fn',
              'The activation function to use. None signifies no activation function.',
              ## iscallable((tf.nn.relu, tf.nn.tanh, tf.nn.sigmoid, None)),
              iscallableOrNone(),
              ),
        PD('normalizer_fn',
              'Normalizer function to use instead of biases. If set, biases are not used.',
              iscallableOrNone(),
              ),
        PD('weights_initializer',
              'Tensorflow weights initializer function',
              iscallable(),
              # tf.contrib.layers.xavier_initializer()
              # tf.contrib.layers.variance_scaling_initializer()
              ),
        PD('biases_initializer',
              'Tensorflow biases initializer function, e.g. tf.zeros_initializer(). ',
              iscallable(),
              ## tf.zeros_initializer()
              ),
        PD('weights_regularizer',
              'L1 / L2 norm regularization',
              iscallable(),
              # tf.contrib.layers.l2_regularizer(scale, scope=None)
              # tf.contrib.layers.l1_regularizer(scale, scope=None)
              ),
        PD('biases_regularizer',
              'L1 / L2 norm regularization',
              iscallableOrNone()
              ),
        PD('dropout', 'Dropout parameters if any.',
              instanceofOrNone(DropoutParams)),
        PD('tb', "Tensorboard Params.",
           instanceofOrNone(TensorboardParams)),
        PD('num_units',
          "(integer): number of output units in the layer." ,
          integer(1),
          ),
        )

    def __init__(self, initVals=None):
        HyperParams.__init__(self, self.proto, initVals)

    def __copy__(self):
        ## Shallow copy
        return self.__class__(self)

    def copy(self, override_vals={}):
        ## Shallow copy
        return self.__class__(self).updated(override_vals)

class FCLayer(object):
    def __init__(self, params, batch_input_shape=None):
        self.my_scope = tf.get_variable_scope()
        self._params = FCLayerParams(params)
        self._batch_input_shape = batch_input_shape

    def __call__(self, inp, layer_idx=None):
        with tf.variable_scope(self.my_scope) as var_scope:
            with tf.name_scope(var_scope.original_name_scope) as name_scope:
                ## Parameter Validation
                assert isinstance(inp, tf.Tensor)
                if self._batch_input_shape is not None:
                    assert K.int_shape(inp) == self._batch_input_shape

                params = self._params
                prefix = 'FC' if params.activation_fn is not None else 'Affine'
                scope_name = prefix + '_%d'%(layer_idx+1) if layer_idx is not None else prefix
                with tf.variable_scope(scope_name) as var_scope:
                    if len(K.int_shape(inp)) != 2:
                        logger.warn('Input to operation %s/%s has rank !=2 (%d)', name_scope, scope_name, len(K.int_shape(inp)))

                    coll_w = scope_name + '/_weights_'
                    coll_b = scope_name + '/_biases_'
                    var_colls = {'biases':[coll_b], 'weights':[coll_w, "REGULARIZED_WEIGHTS"]}
                    # var_colls['weights'] = [coll_w, "REGULARIZED_WEIGHTS"] if (params.weights_regularizer is not None) else [coll_w]
                    assert params.weights_regularizer is not None

                    a = tf.contrib.layers.fully_connected(
                            inputs=inp,
                            num_outputs = params.num_units,
                            activation_fn = params.activation_fn,
                            normalizer_fn = params.normalizer_fn if 'normalizer_fn' in params else None,
                            weights_initializer = params.weights_initializer,
                            # weights_regularizer = params.weights_regularizer,
                            biases_initializer = params.biases_initializer,
                            # biases_regularizer = params.biases_regularizer,
                            variables_collections=var_colls,
                            trainable = True
                            )

                    if self._params.dropout is not None:
                        a = DropoutLayer(self._params.dropout, self._batch_input_shape)(a)

                    # For Tensorboard Summaries
                    self._weights = tf.get_collection(coll_w)
                    self._biases = tf.get_collection(coll_b)
                    self._a = a

                if (self._batch_input_shape):
                    a.set_shape(self._batch_input_shape[:-1] + (params.num_units,))

                return a

    def create_summary_ops(self, coll_name):
        with tf.variable_scope(self.my_scope) as var_scope:
            with tf.name_scope(var_scope.original_name_scope):
                params = self._params
                if params.tb is not None:
                    summarize_layer(self._weights, self._biases, self._a, coll_name)

class MLPParams(HyperParams):
    proto = (
            PD('tb', "Tensorboard Params.",
                instanceofOrNone(TensorboardParams)),
            PD('op_name',
               'Name of the layer; will show up in tensorboard visualization',
               None,
               # 'MLPStack'
              ),
            PD('layers',
               "Sequence of FCLayerParams.",
               issequenceof(FCLayerParams),
               ),
        )

    def __init__(self, initVals=None):
        HyperParams.__init__(self, self.proto, initVals)

    def __copy__(self):
        ## Shallow copy
        return self.__class__(self)

    def copy(self, override_vals={}):
        ## Shallow copy
        return self.__class__(self).updated(override_vals)

class MLPStack(object):
    def __init__(self, params, batch_input_shape=None):
        self.my_scope = tf.get_variable_scope()
        self._params = MLPParams(params)
        self._batch_input_shape = batch_input_shape

    def __call__(self, inp):
        with tf.variable_scope(self.my_scope) as var_scope:
            with tf.name_scope(var_scope.original_name_scope):
                assert isinstance(inp, tf.Tensor)
                if self._batch_input_shape is not None:
                    assert K.int_shape(inp) == self._batch_input_shape

                params = self._params

                a = inp
                self._layers = []
                with tf.variable_scope(params.op_name if 'op_name' in params else 'MLPStack'):
                    # for i in xrange(len(self._params.layers_units)):
                    for i, layerParams in enumerate(params.layers):
                        layer = FCLayer(layerParams)
                        a = layer(a, i)
                        self._layers.append(layer)
                return a

    def create_summary_ops(self, coll_name):
        with tf.variable_scope(self.my_scope) as var_scope:
            with tf.name_scope(var_scope.original_name_scope):
                for layer in self._layers:
                    layer.create_summary_ops(coll_name)

class ConvLayerParams(HyperParams):
    proto = (
        PD('tb', "Tensorboard Params.",
           instanceofOrNone(TensorboardParams)),
        PD('activation_fn',
              'The activation function to use. None signifies no activation function.',
              ## iscallable((tf.nn.relu, tf.nn.tanh, tf.nn.sigmoid, None)),
              iscallableOrNone(),
              ),
        PD('normalizer_fn',
              'Normalizer function to use instead of biases. If set, biases are not used.',
              iscallableOrNone(),
              ),
        PD('weights_initializer',
              'Tensorflow weights initializer function',
              iscallable(),
              # tf.contrib.layers.xavier_initializer()
              # tf.contrib.layers.variance_scaling_initializer()
              ),
        PD('biases_initializer',
              'Tensorflow biases initializer function, e.g. tf.zeros_initializer(). ',
              iscallable(),
              ## tf.zeros_initializer()
              ),
        PD('weights_regularizer',
              'L1 / L2 norm regularization',
              iscallable(),
              # tf.contrib.layers.l2_regularizer(scale, scope=None)
              # tf.contrib.layers.l1_regularizer(scale, scope=None)
              ),
        PD('biases_regularizer',
              'L1 / L2 norm regularization',
              iscallableOrNone()
              ),
        PD('padding',
            'Convnet padding: SAME or VALID',
            ('SAME', 'VALID')
        ),
        PD('output_channels',
            "(integer): Depth of output layer = Number of features/channels in the output." ,
            integer(1),
            ),
        PD('kernel_shape',
            '(sequence): shape of kernel',
            issequenceof(int)
            ),
        PD('stride',
            '(sequence): shape of stride',
            issequenceof(int)
            ),
    )
    def __init__(self, initVals=None):
        HyperParams.__init__(self, self.proto, initVals)

    @classmethod
    def get_kernel_half(cls, self_d):
        # self_d is this class distilled into a dictionary
        kernel_shape = self_d['kernel_shape']
        assert (kernel_shape[0] % 2 == 1) and (kernel_shape[1] % 2 == 1)
        return kernel_shape[0] // 2, kernel_shape[1] // 2


ConvLayerParamsProto = ConvLayerParams().protoD


class ConvStackParams(HyperParams):
    proto = (
        PD('tb', "Tensorboard Params.",
           instanceofOrNone(TensorboardParams)),
        PD('op_name',
           'Name of the stack; will show up in tensorboard visualization',
           None,
           ),
        PD('layers',
           '(sequence of *Params): Sequence of layer params. Each value should be either of type ConvLayerParams '
           'or MaxpoolParams. ',
           instanceof(tuple)
           ),
    )

    def __init__(self, initVals=None):
        HyperParams.__init__(self, self.proto, initVals)

    @staticmethod
    def isConvLayer(layer_d):
        # layer_d is *LayerParams distilled into a dict
        return isinstance(layer_d, ConvLayerParams) or (
                ('output_channels' in layer_d) and ('weights_initializer' in layer_d) and ('kernel_shape' in layer_d) and (
                'stride' in layer_d) and ('padding' in layer_d))

    @staticmethod
    def isPoolLayer(layer_d):
        # layer_d is *LayerParams distilled into a dict
        return isinstance(layer_d, MaxpoolParams) or (
                ('output_channels' not in layer_d) and ('weights_initializer' not in layer_d) and (
                'kernel_shape' in layer_d) and ('stride' in layer_d) and ('padding' in layer_d))

    @classmethod
    def get_numConvLayers(cls, slf_d):
        # slf_d is ConvStackParams distilled into a dict
        # if 'numConvLayers' in slf_d:
        #     return slf_d['numConvLayers']
        # else:
        n = 0
        for layer in slf_d['layers']:
            if cls.isConvLayer(layer):
                n += 1
        return n

    @classmethod
    def get_numPoolLayers(cls, slf_d):
        # slf_d is ConvStackParams distilled into a dict
        # if 'numPoolLayers' in slf_d:
        #     return slf_d['numPoolLayers']
        # else:
        n = 0
        for layer in slf_d['layers']:
            if cls.isPoolLayer(layer):
                n += 1
        return n


class ConvStack(object):
    def __init__(self, params, batch_input_shape=None):
        self.my_scope = tf.get_variable_scope()
        self._params = ConvStackParams(params)
        self._batch_input_shape = batch_input_shape

    def __call__(self, inp):
        with tf.variable_scope(self.my_scope) as var_scope:
            with tf.name_scope(var_scope.original_name_scope):
                assert isinstance(inp, tf.Tensor)
                if self._batch_input_shape is not None:
                    assert K.int_shape(inp) == self._batch_input_shape

                params = self._params

                a = inp
                with tf.variable_scope(params.op_name if 'op_name' in params else 'ConvStack'):
                    for i, layerParams in enumerate(params.layers):
                        if isinstance(layerParams, ConvLayerParams):
                            a = ConvLayer(layerParams)(a, i)
                        elif isinstance(layerParams, MaxpoolParams):
                            a = MaxpoolLayer(layerParams)(a, i)
                        elif isinstance(layerParams, DropoutParams):
                            a = DropoutLayer(layerParams)(a, i)
                        else:
                            raise AttributeError('Unsupported params class (%s) found in ConvStackParams.layers'%layerParams.__class__.__name__)
                return a

class ConvLayer(object):
    def __init__(self, params, batch_input_shape=None):
        self.my_scope = tf.get_variable_scope()
        self._params = ConvLayerParams(params)
        self._batch_input_shape = batch_input_shape

    def __call__(self, inp, layer_idx=None):
        with tf.variable_scope(self.my_scope) as var_scope:
            with tf.name_scope(var_scope.original_name_scope):
                ## Parameter Validation
                assert isinstance(inp, tf.Tensor)
                if self._batch_input_shape is not None:
                    assert K.int_shape(inp) == self._batch_input_shape

                params = self._params
                scope_name = 'Conv' + '_%d'%(layer_idx+1) if layer_idx is not None else 'Conv'
                with tf.variable_scope(scope_name) as var_scope:
                    coll_w = scope_name + '/_weights_'
                    coll_b = scope_name + '/_biases_'
                    var_colls = {'biases':[coll_b], 'weights':[coll_w, "REGULARIZED_WEIGHTS"]}
                    # var_colls['weights'] = [coll_w, "REGULARIZED_WEIGHTS"] if (params.weights_regularizer is not None) else [coll_w]
                    assert params.weights_regularizer is not None

                    a = tf.contrib.layers.conv2d(inputs=inp,
                                                num_outputs=params.output_channels,
                                                kernel_size=params.kernel_shape,
                                                stride=params.stride,
                                                padding=params.padding,
                                                activation_fn=params.activation_fn,
                                                normalizer_fn=params.normalizer_fn if 'normalizer_fn' in params else None,
                                                trainable=True,
                                                weights_initializer=params.weights_initializer,
                                                biases_initializer=params.biases_initializer,
                                                # weights_regularizer=params.weights_regularizer,
                                                # biases_regularizer=params.biases_regularizer,
                                                data_format='NHWC',
                                                variables_collections=var_colls
                                                )

                    # Tensorboard Summaries
                    self._weights = tf.get_collection(coll_w)
                    self._biases = tf.get_collection(coll_b)
                    self._a = a

                # if (self._batch_input_shape):
                #     a.set_shape(self._batch_input_shape[:-1] + (params.num_units,))

                return a

    def create_summary_ops(self, coll_name):
        with tf.variable_scope(self.my_scope) as var_scope:
            with tf.name_scope(var_scope.original_name_scope):
                params = self._params
                if params.tb is not None:
                    summarize_layer(self._weights, self._biases, self._a, coll_name)

class MaxpoolParams(HyperParams):
    proto = (
        ConvLayerParamsProto.kernel_shape,
        ConvLayerParamsProto.stride,
        ConvLayerParamsProto.padding,
        ConvLayerParamsProto.tb
        )

    def __init__(self, initVals=None):
        HyperParams.__init__(self, self.proto, initVals)

class MaxpoolLayer(object):
    def __init__(self, params, batch_input_shape=None):
        self.my_scope = tf.get_variable_scope()
        self._params = MaxpoolParams(params)
        self._batch_input_shape = batch_input_shape

    def __call__(self, inp, layer_idx=None):
        with tf.variable_scope(self.my_scope) as var_scope:
            with tf.name_scope(var_scope.original_name_scope):
                ## Parameter Validation
                assert isinstance(inp, tf.Tensor)
                if self._batch_input_shape is not None:
                    assert K.int_shape(inp) == self._batch_input_shape

                params = self._params
                scope_name = 'MaxPool' + '_%d'%(layer_idx+1) if layer_idx is not None else 'MaxPool'
                with tf.variable_scope(scope_name) as var_scope:
                    layer_name = var_scope.name
        #            coll_w = layer_name + '/' + params.tb.tb_weights
        #            coll_b = layer_name + '/' + params.tb.tb_biases

                    a = tf.contrib.layers.max_pool2d(inputs=inp,
                                             kernel_size=params.kernel_shape,
                                             stride=params.stride,
                                             padding=params.padding,
                                             data_format='NHWC')

                    self._a = a

                # if (self._batch_input_shape):
                #     a.set_shape(self._batch_input_shape[:-1] + (params.num_units,))

                return a

    def create_summary_ops(self, coll_name):
        with tf.variable_scope(self.my_scope) as var_scope:
            with tf.name_scope(var_scope.original_name_scope):
                params = self._params
                if params.tb is not None:
                    summarize_layer(None, None, self._a, coll_name)

class DropoutLayer(object):
    def __init__(self, params, batch_input_shape=None):
        self.my_scope = tf.get_variable_scope()
        self._params = DropoutParams(params)
        self._batch_input_shape = batch_input_shape

    def __call__(self, inp, layer_idx=None):
        with tf.variable_scope(self.my_scope) as var_scope:
            with tf.name_scope(var_scope.original_name_scope) as name_scope:
                ## Parameter Validation
                assert isinstance(inp, tf.Tensor)
                if self._batch_input_shape is not None:
                    assert K.int_shape(inp) == self._batch_input_shape

                params = self._params
                scope_name = 'Dropout_%d'%(layer_idx+1) if layer_idx is not None else 'Dropout'
                with tf.variable_scope(scope_name):
                    if len(K.int_shape(inp)) != 2:
                        logger.warn('Input to operation %s/%s has rank !=2 (%d)', name_scope, scope_name, len(K.int_shape(inp)))
                    return tf.nn.dropout(inp, params.keep_prob, seed=params.seed)

class ActivationParams(HyperParams):
    proto = (
        PD('tb', "Tensorboard Params.",
           instanceofOrNone(TensorboardParams)),
        PD('activation_fn',
            'The activation function to use.',
            ## iscallable((tf.nn.relu, tf.nn.tanh, tf.nn.sigmoid, None)),
            iscallable(),
            ),
        PD('dropout', 'Dropout parameters if any.',
           instanceof(DropoutParams)),
        )
    def __init__(self, initVals=None):
        HyperParams.__init__(self, self.proto, initVals)
    def __copy__(self):
        ## Shallow copy
        return self.__class__(self)

    def copy(self, override_vals={}):
        ## Shallow copy
        return self.__class__(self).updated(override_vals)

class Activation(object):
    def __init__(self, params, batch_input_shape=None):
        self.my_scope = tf.get_variable_scope()
        self._params = ActivationParams(params)
        self._batch_input_shape = batch_input_shape

    def __call__(self, inp, layer_idx=None):
        with tf.variable_scope(self.my_scope) as var_scope:
            with tf.name_scope(var_scope.original_name_scope) as name_scope:
                ## Parameter Validation
                assert isinstance(inp, tf.Tensor)
                if self._batch_input_shape is not None:
                    assert K.int_shape(inp) == self._batch_input_shape

                params = self._params

                scope_name = 'Activation_%d'%(layer_idx+1) if layer_idx is not None else 'Activation'
                with tf.variable_scope(scope_name):
                    if len(K.int_shape(inp)) != 2:
                        logger.warn('Input to operation %s/%s has rank !=2 (%d)', name_scope, scope_name, len(K.int_shape(inp)))
                    a = params.activation_fn(inp)

                if ('dropout' in params) and (params.dropout is not None):
                    a = DropoutLayer(params.dropout, self._batch_input_shape)(a)

                self._a = a
                return a

    def create_summary_ops(self, coll_name):
        with tf.variable_scope(self.my_scope) as var_scope:
            with tf.name_scope(var_scope.original_name_scope) as scope_name:
                if self._params.tb is not None:
                    summarize_layer(None, None, self._a, coll_name)

class RNNParams(HyperParams):
    proto = (
            PD('B',
               '(integer or None): Size of mini-batch for training, validation and testing.',
               integerOrNone(1)
               ),
            PD('i',
               '(integer): dimensionality of the input vector (Ey / Ex)',
               integer(1)
               ),
            PD('type',
               'Type of RNN cell to create. Only LSTM is supported at this time.',
               ('lstm',),
               'lstm'),
            PD('dtype',
               "dtype of data",
               (tf.float32, tf.float64),
               tf.float32),
            PD('op_name',
               'Name of the layer; will show up in tensorboard visualization',
               None,
               'LSTMWrapper'
              ),
            PD('layers_units',
              "(sequence of integers): Either layers_units or num_units must be specified. Not both."
              "Sequence of numbers denoting number of units for each layer."
              "As many layers will be created as the length of the sequence",
              issequenceofOrNone(int)
              ),
            CommonParamsProto.weights_initializer,
            CommonParamsProto.weights_regularizer,
            PD('use_peephole',
               '(boolean): whether to employ peephole connections in the decoder LSTM',
               (True, False),
               True),
            PD('forget_bias', '',
               decimal(),
               1.0),
            PD('dropout', 'Dropout parameters if any.',
               instanceofOrNone(DropoutParams)),
            PD('tb', '',
               instanceofOrNone(TensorboardParams))
            )
    def __init__(self, initVals=None):
        HyperParams.__init__(self, self.proto, initVals)

    def __copy__(self):
        ## Shallow copy
        return self.__class__(self)

    def copy(self, override_vals={}):
        ## Shallow copy
        return self.__class__(self).updated(override_vals)


def expand_nested_shape(shape, B):
    """ Convert every scalar in the nested sequence into a (B, s) sequence """
    if not dlc.issequence(shape):
        return (B, shape)
    else:
        return tuple(expand_nested_shape(s, B) for s in shape)

def get_nested_shape(obj):
    """ Get shape of possibly nested sequence of tensors """
    if not dlc.issequence(obj):
        return K.int_shape(obj)
    else:
        return tuple(get_nested_shape(o) for o in obj)

def nested_tf_shape(s):
    """
    Get tf.TensorShape for each tensor in a nested structure of tuples/named tuples.
    TensorShape objects are returned nested within the same tuple/named-tuple structure.
    """
    if dlc.issequence(s):
        lst = [nested_tf_shape(e) for e in s]
        if hasattr(s, '_make'):
            return s._make(lst)
        else:
            return tuple(lst)
    else:
        return s.shape.as_list()


class RNNWrapper(tf.nn.rnn_cell.RNNCell):
    def __init__(self, params, reuse=None, _scope=None, beamsearch_width=1):
        self._params = params = RNNParams(params)
        with tf.variable_scope(_scope or self._params.op_name,
                               initializer=self._params.weights_initializer, #regularizer=self._params.weights_regularizer
                               ) as scope:
            super(RNNWrapper, self).__init__(_reuse=reuse, _scope=scope, name=scope.name)
            self.my_scope = scope
            if len(self._params.layers_units) == 1:
                self._cell = self._make_one_cell(self._params.layers_units[0])
                self._num_layers = 1
            else: # len(params.layers_units) > 1
                ## Note: though we're creating multiple LSTM cells in the same scope, no variables
                ## are being created here. They will be created when the individual cells are called
                ## as part of MultiRNNCell.call(), wherein each is given a unique scope.
                self._cell = tf.nn.rnn_cell.MultiRNNCell(
                        [self._make_one_cell(n) for n in self._params.layers_units])
                self._num_layers = len(self._params.layers_units)

            self._batch_state_shape = expand_nested_shape(self._cell.state_size,
                                                          self._params.B*beamsearch_width)
            self._beamsearch_width = beamsearch_width
            # print('RNNWrapper.__init__: batch input shape = %s'%(self.batch_input_shape,))

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def zero_state(self, batch_size, dtype):
        with tf.variable_scope(self.my_scope) as var_scope:
            with tf.name_scope(var_scope.original_name_scope):
                return self._cell.zero_state(batch_size, dtype)

    @property
    def batch_input_shape(self):
        return (self._params.B*self._beamsearch_width, self._params.i)

    @property
    def batch_output_shape(self):
        return (self._params.B*self._beamsearch_width, self._cell.output_size)

    @property
    def batch_state_shape(self):
        return self._batch_state_shape

#    def _set_beamwidth(self, beamwidth):
#        self._beamsearch_width = beamwidth
#        self._batch_state_shape = expand_nested_shape(self.state_size,
#                                                      self._params.B*beamwidth)

    def assertStateShape(self, state):
        """
        Asserts that the shape of the tensor is consistent with the RNN's state shape.
        For e.g. the state-shape of a MultiRNNCell with L layers would be ((n1, n1), (n2, n2) ... (nL, nL))
        and the corresponding state-tensor should be of shape :
        (((B,n1),(B,n1)), ((B,n2),(B,n2)) ... ((B,nL),(B,nL)))
        """
        assert self.batch_state_shape == get_nested_shape(state), 'state-shape assertion Failed for state type = %s and value = %s'%(type(state), state)

    def assertOutputShape(self, output):
        """
        Asserts that the shape of the tensor is consistent with the RNN's output shape.
        For e.g. the output shape is o then the input-tensor should be of shape (B,o)
        """
        assert self.batch_output_shape == K.int_shape(output)

    def assertInputShape(self, inp):
        """
        Asserts that the shape of the tensor is consistent with the RNN's input shape.
        For e.g. if the input shape is m then the input-tensor should be of shape (B,m)
        """
        assert K.int_shape(inp) == self.batch_input_shape

    def _assertBatchShape(self, shape, tnsr):
        """
        Asserts that the shape of the tensor is consistent with the RNN's shape.
        For e.g. the state-shape of a MultiRNNCell with L layers would be ((n1, n1), (n2, n2) ... (nL, nL))
        and the corresponding state-tensor should be (((B,n1),(B,n1)), ((B,n2),(B,n2)) ... ((B,nL),(B,nL)))
        Similarly, if the input shape is m then the input-tensor should be of shape (B,m)
        """
        return expand_nested_shape(shape, self._params.B*self._beamsearch_width) == K.int_shape(tnsr)

    @property
    def num_layers(self):
        return self._num_layers

    def call(self, inp, state):
        with tf.variable_scope(self.my_scope) as var_scope:
            with tf.name_scope(var_scope.original_name_scope):
                ## Parameter Validation
                assert isinstance(inp, tf.Tensor)
                self.assertInputShape(inp)
                self.assertStateShape(state)

                params = self._params
                output, new_state = self._cell(inp, state)
                # Tensorboard Summaries
                # if params.tb is not None:
                #     summarize_layer(self._params.op_name, None, None, output, new_state)
                    # summarize_layer(self._params.op_name, tf.get_collection('weights'),
                    #                 tf.get_collection('biases'), output, new_state)

                self.assertOutputShape(output)
                self.assertStateShape(new_state)
                return output, new_state

    def _make_one_cell(self, num_units):
        with tf.variable_scope(self.my_scope) as var_scope:
            with tf.name_scope(var_scope.original_name_scope):
                params = self._params
                #The implementation is based on: http://arxiv.org/abs/1409.2329.
                ## LSTMBlockFusedCell is replacement for LSTMBlockCell
                cell = tf.contrib.rnn.LSTMBlockCell(num_units,
                                                   forget_bias=params.forget_bias,
                                                   use_peephole=params.use_peephole
                                                   )
                if params.dropout is not None:
                    with tf.variable_scope('DropoutWrapper'):
                        cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                      input_keep_prob=1.,
                                                      state_keep_prob=params.dropout.keep_prob,
                                                      output_keep_prob=params.dropout.keep_prob,
                                                      variational_recurrent=True,
                                                      dtype=params.dtype
                                                      )
                return cell


def sizeofVar(var):
    shape = K.int_shape(var)
    return np.prod(shape)

def printVars(name, coll):
    logger.critical(name)
    total_n = 0
    for var in tf.trainable_variables():
        n = sizeofVar(var)
        total_n += n
        logger.critical("%s %s num_params = %d", var.name, K.int_shape(var), n)
    logger.critical('Total number of variables = %d', total_n)
    return total_n

def edit_distance3D(B, k, predicted_ids, predicted_lens, target_ids, target_lens, blank_token=None,  space_token=None, eos_token=None):
    """Compute edit distance for matrix of shape (B,k,T) """
    with tf.name_scope('edit_distance3D'):
        p_shape = K.int_shape(predicted_ids)
        t_shape = K.int_shape(target_ids)
        assert len(p_shape) == 3
        assert p_shape[:2] == (B, k)
        assert K.int_shape(predicted_lens) == (B, k)
        assert len(t_shape) == 3
        assert t_shape[:2] == (B, k)
        assert K.int_shape(target_lens) == (B, k)
        predicted_sparse = dense_to_sparse3D(B, predicted_ids, predicted_lens, blank_token, space_token, eos_token=eos_token)
        ## blank tokens should not be present in target_ids
        target_sparse = dense_to_sparse3D(B, target_ids, target_lens, space_token=space_token, eos_token=eos_token)

        d = tf.edit_distance(predicted_sparse, target_sparse)
        # assert K.int_shape(d) == K.int_shape(predicted_lens)
        d.set_shape(predicted_lens.shape) ## Reassert shape in case it got lost.
        return d

def edit_distance2D(B, predicted_ids, predicted_lens, target_ids, target_lens, blank_token=None, space_token=None, eos_token=None):
    """
    Compute edit distance of predicted_ids (optionally ignoring blank_tokens, space_tokens and eos_tokens) with target_ids
    which are assumed to have no blank tokens but may have space tokens and eos_tokens.
    The result is equivalent to computing edit_distance after squashing predicted_lens - but is more efficient since it doesn't
    actually do the squashing.
    Args:
        B: batch-size
        predicted_ids: shape (B, T)
        target_ids: shape (B, T)
        blank_token: blank token as defined by CTC.
    """
    with tf.name_scope('edit_distance2D'):
        p_shape = K.int_shape(predicted_ids)
        t_shape = K.int_shape(target_ids)
        assert len(p_shape) == 2
        assert p_shape[0] == B
        assert K.int_shape(predicted_lens) == (B,)
        assert len(t_shape) == 2
        assert t_shape[0] == B
        assert K.int_shape(target_lens) == (B,)

        predicted_sparse = dense_to_sparse2D(predicted_ids, predicted_lens, blank_token, space_token, eos_token)

        ## blank tokens should not be present in target_ids
        target_sparse = dense_to_sparse2D(target_ids, target_lens, space_token=space_token, eos_token=eos_token)

        d = tf.edit_distance(predicted_sparse, target_sparse)
        # assert K.int_shape(d) == K.int_shape(predicted_lens)
        d.set_shape(predicted_lens.shape) ## Reassert shape in case it got lost.
        return d


def edit_distance2D_sparse(B, predicted_sparse, target_ids, target_lens, space_token=None, eos_token=None):
    """
    Compute edit distance of predicted_ids (ignoring blank and space tokens) with target_ids (which are assumed to have no blank tokens).
    The result is equivalent to computing edit_distance after squashing predicted_lens - but is more efficient since it doesn't
    actually do that.
    Args:
        B: batch-size
        predicted_sparse: sparse tensor of dense_shape (B, T)
        target_ids: dense tensor of shape shape (B, T)
        target_lens: lengths of id-sequences (rows of target_ids) - (B,)
    """
    with tf.name_scope('edit_distance2D'):
        p_shape = predicted_sparse.get_shape().as_list()
        t_shape = K.int_shape(target_ids)
        assert len(p_shape) == 2
        assert p_shape[0] == B
        assert K.int_shape(predicted_lens) == (B,)
        assert len(t_shape) == 2
        assert t_shape[0] == B
        assert K.int_shape(target_lens) == (B,)

        target_sparse = dense_to_sparse2D(target_ids, target_lens, space_token=space_token, eos_token=eos_token)

        d = tf.edit_distance(predicted_sparse, target_sparse)
        d.set_shape(target_lens.shape) ## Reassert shape in case it got lost.
        return d


def seqlens(m, eos_token=0, include_eos_token=True):
    """
    Takes in a tensor m, shaped (..., T) having '...' sequences of length T each optionally containing one or more eos_tokens.
    Returns the length of each sequence upto the first eos_token. If a sequence didn't have a any eos_token
    then it returns T for that sequence. If include_eos_token is True, one eos_token is counted towards the length otherwise
    not.

    Args:
        m: input tensor containing the sequences whose lengths need to be computed. The tensor must have rank at least 2 - i.e.
        rank(m) >= 2. The last dimension must be the time dimension. e.g. (B,T) where B is the batch dimension and T is the time
        dimension. (B,W,T) where W is the beam-width dimension and so on. All dimensions but the time dimension (T) must be
        fully specified.
        eos_token: (integer or integer tensor) (optional) the eos_token. defaults to zero.
        include_eos_token: (boolean) (optional) if True (default) one eos_token is counted towards the length otherwise not.
            However, if no eos_token was found in the sequence then the max_length (T) is returned as its length.

    Returns:
        A tensor of shape m.shape[:-1] holding sequence-lengths - i.e. shape of input minus the last dimension. If the input
        was shape (B,T), output will have shape (B,). If input is shape (B,W,T) output will have shape (B,W) and so on.
    """
    with tf.name_scope('seqlens'):
        orig_shape = K.int_shape(m)
        assert len(orig_shape) >= 2
        T = tf.shape(m)[-1]
        B = np.prod(orig_shape[:-1])
        if len(orig_shape) > 2:
            m = tf.reshape(m, (B, T))
        assert B > 0
        lens = []
        for i in range(B):
            m_i = m[i]
            eos_bool = tf.equal(m_i, eos_token)
            eos_indices = tf.where(eos_bool) ## indices sorted in row-major order, therefore the first occurence is at the beginning
            has_eos = tf.reduce_any(eos_bool)
            len_i = tf.cond(has_eos,
                            true_fn=lambda: tf.cast(((eos_indices[0][0] + 1) if include_eos_token else eos_indices[0][0]), dtype=tf.int64),
                            false_fn=lambda: tf.cast(T, dtype=tf.int64))
#            len_i.set_shape( () )
            lens.append(len_i)
        tf_lens = tf.stack(lens)

        return tf_lens if (len(orig_shape) == 2) else tf.reshape(tf_lens, orig_shape[:-1])


#def seqlens_2d(B, m, eos_token=0):
#    """
#    Takes in a tensor m, shaped (B, T) having 'B' sequences of length T and optionally containing one or more eos_tokens.
#    Returns the length of each sequences upto but not including the first eos_token. If a sequence didn't have a any eos_token
#    then it returns T for that sequence.
#
#    Returns:
#        A tensor of shape (B,) holding sequence-lengths.
#    """
#    with tf.name_scope('seqlens_2d'):
#        assert len(K.int_shape(m)) == 2
#        assert K.int_shape(m)[0] == B
#        T = tf.shape(m)[1]
#        lens = []
#        for i in range(B):
#            m_i = m[i]
#            eos_bool = tf.equal(m_i, eos_token)
#            eos_indices = tf.where(eos_bool, name='eos_indices')
#            has_eos = tf.reduce_any(eos_bool)
#            len_i = tf.cond(has_eos, true_fn=lambda: tf.cast(eos_indices[0][0], dtype=tf.int64), false_fn=lambda: tf.cast(T, dtype=tf.int64))
##            len_i.set_shape( () )
#            lens.append(len_i)
#
#        return tf.stack(lens)


def squash_2d(B, m, lens, blank_token, padding_token=0):
    """
    Takes in a tensor m, shaped (B, T), from each row removes blank_tokens and appends
    an equal number of padding_tokens at the end. Returns the resulting (B, T) shaped tensor and
    a tensor carrying sequence_lengths, shaped (B,).
    """
    with tf.name_scope('squash_2d'):
        assert len(K.int_shape(m)) == 2
        assert K.int_shape(lens) == (B,)
        assert K.int_shape(m)[0] == B
        T = tf.shape(m)[1]
        squashed = []
        squashed_lens = []
        seq_mask = tf.sequence_mask(lens, maxlen=T)
        for i in range(B):
            m_i = m[i]
            mask_i = tf.logical_and(seq_mask[i], tf.not_equal(m_i, blank_token)) #(T,)
            idx_i = tf.where(mask_i) # (N, 1)
            vals_i = tf.gather_nd(m_i, idx_i) # (N, )
            len_i = tf.shape(vals_i)[0]
            paddings = [(0, T - len_i)] # (1, 2)
            squashed.append(tf.pad(vals_i, paddings, mode="CONSTANT", name=None, constant_values=padding_token))
            squashed_lens.append(len_i)
        return tf.stack(squashed), tf.stack(squashed_lens)


def squash_3d(B, k, m, lens, blank_token, padding_token=0):
    """
    Takes in a tensor m, shaped (B, k, T), from each sequence of length T removes blank_tokens and appends
    an equal number of padding_tokens at the end. Returns the resulting (B, k, T) shaped tensor and
    a tensor carrying sequence_lengths, shaped (B, k).
    """
    with tf.name_scope('squash_3d'):
        assert len(K.int_shape(m)) == 3
        assert K.int_shape(m)[:2] == (B,k)
        assert K.int_shape(lens) == (B,k)
        squashed = []
        squashed_lens = []
        for i in range(B):
            s_m, s_l = squash_2d(k, m[i], lens[i], blank_token, padding_token)
            squashed.append(s_m)
            squashed_lens.append(s_l)
        return tf.stack(squashed), tf.stack(squashed_lens)


def _dense_to_sparse(t, mask, blank_token=None, space_token=None, eos_token=None):
        if blank_token is not None:
            mask = tf.logical_and(mask, tf.not_equal(t, blank_token))
        if space_token is not None:
            mask = tf.logical_and(mask, tf.not_equal(t, space_token))
        if eos_token is not None:
            mask = tf.logical_and(mask, tf.not_equal(t, eos_token))

        idx = tf.where(mask)
        vals = tf.gather_nd(t, idx)
        return tf.SparseTensor(idx, vals, tf.shape(t, out_type=tf.int64))


def dense_to_sparse2D(t, lens, blank_token=None, space_token=None, eos_token=None):
    with tf.name_scope('dense_to_sparse2D'):
        assert len(K.int_shape(t)) == 2
        assert len(K.int_shape(lens)) == 1
        mask = tf.sequence_mask(lens, maxlen=tf.shape(t)[1])
        return _dense_to_sparse(t, mask, blank_token, space_token, eos_token)


def dense_to_sparse3D(B, t, lens, blank_token=None, space_token=None, eos_token=None):
    with tf.name_scope('dense_to_sparse3D'):
        assert K.int_shape(t)[0] == B
        assert len(K.int_shape(t)) == 3
        assert len(K.int_shape(lens)) == 2
        mask = tf.stack([tf.sequence_mask(lens[i], maxlen=tf.shape(t)[2]) for i in range(B)])
        return _dense_to_sparse(t, mask, blank_token, space_token, eos_token)


def ctc_loss(yLogits, logits_lens, y_ctc, ctc_len, B, Kv):
    # B = self.C.B
    # ctc_len = self._ctc_len
    # y_ctc = self._y_ctc

    with tf.name_scope('_ctc_loss'):
        assert K.int_shape(yLogits) == (B, None, Kv)
        assert K.int_shape(logits_lens) == (B,)
        ctc_mask = tf.sequence_mask(ctc_len, maxlen=tf.shape(y_ctc)[1], dtype=tf.bool)
        assert K.int_shape(ctc_mask) == (B,None) # (B,T)
        y_idx =    tf.where(ctc_mask)
        y_vals =   tf.gather_nd(y_ctc, y_idx)
        y_sparse = tf.SparseTensor(y_idx, y_vals, tf.shape(y_ctc, out_type=tf.int64))
        ctc_losses = tf.nn.ctc_loss(y_sparse,
                                    yLogits,
                                    logits_lens,
                                    ctc_merge_repeated=False,
                                    time_major=False)
        assert K.int_shape(ctc_losses) == (B, )
        return ctc_losses


def batch_bottom_k_2D(t, k):
    t2, indices = batch_top_k_2D(t * -1, k)
    return t2 * -1, indices


def batch_top_k_2D(t, k):
    """
    Slice a tensor of size (B, W) to size (B, k) by taking the top k of W values of
    each row. The rows in the returned tensor (B,k) are sorted in descending order
    of values.
    """
    shape_t = K.int_shape(t)
    assert len(shape_t) == 2 # (B, W)
    B = shape_t[0]

    with tf.name_scope('_batch_top_k'):
        t2, top_k_ids = tf.nn.top_k(t, k=k, sorted=True) # (B, k)
        # Convert ids returned by tf.nn.top_k to indices appropriate for tf.gather_nd.
        # Expand the id dimension from 1 to 2 while prefixing each id with the batch-index.
        # Thus the indices will go from size (B,k,1) to (B, k, 2). These can then be used to
        # create a new tensor from the original tensor on which top_k was invoked originally.
        # If that tensor was - for e.g. - of shape (B, W, T) then we'll now be able to slice
        # it into a tensor of shape (B,k,T) using the top_k_ids.
        reshaped_ids = tf.reshape(top_k_ids, shape=(B, k, 1)) # (B, k, 1)
        b = tf.reshape(tf.range(B), shape=(B,1)) # (B, 1)
        b = tf.tile(b, [1,k]) # (B, k)
        b = tf.reshape(b, shape=(B, k, 1)) #(B,k,1)
        indices = tf.concat((b, reshaped_ids), axis=2) # (B, k, 2)
        assert K.int_shape(indices) == (B, k, 2)
        return t2, indices


def batch_slice(t, indices):
    """
    Apply slice-indices returned by batch_top_k_2D (B, k, 2) to a tensor
    t (B, W, ...) and return the resulting tensor of shape (B, k, ...).
    Used for cases where the top_k values are sliced off from a tensor
    containing one metric (say score) and then applied to a tensor having
    another metric (say accuracy).
    """
    with tf.name_scope('_batch_slice'):
        shape_t = K.int_shape(t) # (B, W ,...)
        shape_i = K.int_shape(indices) # (B, k, 2)
        assert len(shape_t) >= 2
        assert len(shape_i) == 3
        assert shape_i[2] == 2
        B = shape_t[0]
        W = shape_t[1]
        k = shape_i[1]
        assert W >= k

        t_slice = tf.gather_nd(t, indices) # (B, k ,...)
        return t_slice


def group2D(a, stride):
    """
    Flattens and concatenates rectangular blocks of a tensor (B, H, W, C) along dimensions H and W. The first dimension
    B (batch dimension) is left untouched. Given a stride of (h,w), the operation scans the 2D shape (H,W)
    in strides lengths of (h,w) and reshapes each (h,w,C) block to a shape of (1,1,C*h*w). The resulting output shape
    thus is (B, H/h, W/w, C*h*w). H must be divisible by h and W by w.
    The operation is a 2D pooling operation similar to max-pooling except that
    all channels are concatenated instead of taking their max. Hence, the channel dimension gets expanded by the
    block-size.
    :param a: The input tensor. Must be of rank 3 or more and shape (B, H, W, ...).
    :param stride: Shape - (h,w) - of blocks to be fused along dimensions H and W of the input.
            Similar to specifying the shape of a convolution kernel.
    :return: The reshaped tensor of shape (B, H/h, W/w, C*h*w)
    """
    shape = K.int_shape(a)
    B = shape[0]
    H = shape[1]
    W = shape[2]
    assert B is not None
    assert H is not None
    assert W is not None
    h,w = stride
    assert H%h == 0
    assert W%w == 0

    rows = []
    for j in range(H/h):
        row = []
        for i in range(W/w):
            block = a[:, j*h:(j+1)*h, i*w:(i+1)*w]  # (B, h, w, C)
            block = tf.reshape(block, [B, -1])  # (B, h*w*C)
            row.append(block)  # [(B, h*w*C), ...]
        row = tf.stack(row, axis=1)  # (B, W/w, h*w*C)
        rows.append(row)  # [(B, W/w, h*w*C), ...]

    return tf.stack(rows, axis=1)  # [(B, H/h, W/w, h*w*C), ...]


def add_to_collection(name, value):
    if value in tf.get_collection(name):
        logger.warn('tfc.add_to_collection: collection %s already has value %s'%(name, value))
    else:
        tf.add_to_collection(name, value)