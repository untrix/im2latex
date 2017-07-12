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
import os
import time
import dl_commons as dlc
import tensorflow as tf
import keras
from keras import backend as K
from dl_commons import mandatory, instanceof, equalto, boolean, HyperParams, PD, PDL, iscallable
from dl_commons import issequenceof, integer, decimal


class tensor(instanceof):
    """Tensor shape validator to go with ParamDesc"""
    def __init__(self, shape):
        self._shape = shape
        dlc._ParamValidator.__init__(self, tf.Tensor)
    def __contains__(self, obj):
        return instanceof.__contains__(self, obj) and  keras.backend.int_shape(obj) == self._shape
        

def summarize_layer(layer_name, weights, biases, activations):
    def summarize_vars(var, section_name):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        mean = tf.reduce_mean(var)
        tf.summary.scalar(section_name+'/mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar(section_name+'/stddev', stddev)
        tf.summary.scalar(section_name+'/max', tf.reduce_max(var))
        tf.summary.scalar(section_name+'/min', tf.reduce_min(var))
        tf.summary.histogram(section_name+'/histogram', var)

    with tf.variable_scope('SummaryLogs', reuse=False):
        if weights is not None:
            summarize_vars(weights, 'Weights')
        if biases is not None:
            summarize_vars(biases, 'Biases')
        if activations is not None:
            summarize_vars(activations, 'Activations')

class TensorboardParams(HyperParams):
    proto = (
        PD('tb_logdir', '', None, "tb_metrics/"),
        PD('tb_weights',
              'Section name under which weight summaries show up on tensorboard',
              None, 'Weights'
              ),
        PD('tb_biases',
              'Section name under which bias summaries show up on tensorboard',
              None,
              'Biases'
              ),
        PD('tb_activations',
              'Section name under which activation summaries show up on tensorboard',
              None, 'Activations'
              )
        )
    def __init__(self, initVals=None):
        HyperParams.__init__(self, self.proto, initVals)

class DropoutParams(HyperParams):
    """ Dropout Layer descriptor """
    proto = (
        PD('keep_prob',     
              'Probability of keeping an output (i.e. not dropping it).',
              decimal(0.1,1.),
              0.9
              ),
        PD('seed',
              'Integer seed for the random number generator',
              integerOrNone(),
              None
              )
        )
    def __init__(self, initVals=None):
        HyperParams.__init__(self, self.proto, initVals)

class CommonParams(HyperParams):
    proto = (
        PD('activation_fn',
              'The activation function to use. None signifies no activation function.', 
              ## iscallable((tf.nn.relu, tf.nn.tanh, tf.nn.sigmoid, None)),
              iscallableOrNone(),
              tf.nn.tanh),
        PD('normalizer_fn',
              'Normalizer function to use instead of biases. If set, biases are not used.', 
              (None,) 
              ),
        PD('weights_initializer', 
              'Tensorflow weights initializer function', 
              iscallableOrNone(),
              tf.contrib.layers.xavier_initializer()
              # tf.contrib.layers.xavier_initializer_conv2d()
              # tf.contrib.layers.variance_scaling_initializer()
              ),
        PD('biases_initializer', 
              'Tensorflow biases initializer function, e.g. tf.zeros_initializer(). ',
              iscallable(),
              tf.zeros_initializer()
              ),
        PD('weights_regularizer',
              'Defined in tf.contrib.layers. Not sure what this is, but probably a normalizer?',
              iscallable(noneokay=True), None),
        PD('biases_regularizer',
              'Defined in tf.contrib.layers. Not sure what this is, but probably a normalizer?',
              iscallable(noneokay=True), None),
        PD('make_tb_metric',"(boolean): Create tensorboard metrics.",
              boolean,
              True),
        PD('dropout', 'Dropout parameters if any.',
              instanceof(DropoutParams, True)),
        PD('weights_coll_name',
              'Name of trainable weights collection. There is no need to change the default = WEIGHTS',
              ('WEIGHTS',),
              'WEIGHTS'
              ),
        PD('tb', "Tensorboard Params.",
           instanceof(TensorboardParams))
        )
        
    def __init__(self, initVals=None):
        HyperParams.__init__(self, self.proto, initVals)

CommonParamsProto = CommonParams().protoD

class MLPParams(HyperParams):
    proto = CommonParams.proto + (
            PD('op_name',
               'Name of the layer; will show up in tensorboard visualization',
               None,
               'MLP'
              ),
            PD('layers_units',
              "(sequence of integers): sequence of numbers denoting number of units for each layer." 
              "As many layers will be created as the length of the sequence",
              issequenceof(int)
              ),
        )
    def __init__(self, initVals=None):
        HyperParams.__init__(self, self.proto, initVals)
        
    def get_layer_params(self, i):
        return FCLayerParams(self).updated({'num_units': self.layers_units[i]})

class MLPStack(object):
    def __init__(self, params, batch_input_shape=None):
        self._params = MLPParams(params)
        self._batch_input_shape = batch_input_shape
        
    def __call__(self, inp):
        assert isinstance(inp, tf.Tensor)
        if self._batch_input_shape is not None:
            assert K.int_shape(inp) == self._batch_input_shape

        params = self._params
        dropout = params.dropout is not None and (params.dropout.keep_prob < 1.)
        
        a = inp
        with tf.variable_scope(params.op_name):
            for i in xrange(len(self._params.layers_units)):
                a = FCLayer(self._params.get_layer_params(i))(a, i)
                if dropout:
                    a = DropoutLayer(self._params.dropout)(a, i)
        return a
            
class FCLayerParams(HyperParams):
    proto = CommonParams.proto + (
        PD('num_units', 
          "(integer): number of output units in the layer." ,
          integer(1),
          ),
        )
        
    def __init__(self, initVals=None):
        HyperParams.__init__(self, self.proto, initVals)

class FCLayer(object):
    def __init__(self, params, batch_input_shape=None):
        self._params = FCLayerParams(params)
        self._batch_input_shape = batch_input_shape

    def __call__(self, inp, layer_idx=None):
        ## Parameter Validation
        assert isinstance(inp, tf.Tensor)
        if self._batch_input_shape is not None:
            assert K.int_shape(inp) == self._batch_input_shape

        params = self._params    
        prefix = 'FC' if params.activation_fn is not None else 'Affine'
        scope_name = prefix + '_%d'%(layer_idx+1) if layer_idx is not None else prefix
        with tf.variable_scope(scope_name) as var_scope:
            layer_name = var_scope.name
            coll_w = layer_name + '/' + params.tb.tb_weights
            coll_b = layer_name + '/' + params.tb.tb_biases
            
            h = tf.contrib.layers.fully_connected(
                    inputs=inp,
                    num_outputs = params.num_units,
                    activation_fn = params.activation_fn,
                    normalizer_fn = params.normalizer_fn,
                    weights_initializer = params.weights_initializer,
                    weights_regularizer = params.weights_regularizer,
                    biases_initializer = params.biases_initializer,
                    biases_regularizer = params.biases_regularizer,
                    variables_collections = {"weights":[coll_w, params.weights_coll_name],
                                             "biases":[coll_b]},
                    trainable = True
                    )
    
            # Tensorboard Summaries
            if params.make_tb_metric:
                summarize_layer(layer_name, tf.get_collection(coll_w), tf.get_collection(coll_b), h)
    
        return h

class DropoutLayer(object):
    def __init__(self, params, batch_input_shape=None):
        self._params = DropoutParams(params)
        self._batch_input_shape = batch_input_shape

    def __call__(self, inp, layer_idx=None):
        ## Parameter Validation
        assert isinstance(inp, tf.Tensor)
        if self._batch_input_shape is not None:
            assert K.int_shape(inp) == self._batch_input_shape
        
        params = self._params    
        scope_name = 'Dropout_%d'%(layer_idx+1) if layer_idx is not None else 'Dropout'
        with tf.variable_scope(scope_name):
            return tf.nn.dropout(inp, params.keep_prob, seed=params.seed)

class ActivationParams(HyperParams):
    proto = (CommonParamsProto.activation_fn, 
             CommonParamsProto.tb,
             CommonParamsProto.make_tb_metric)
    def __init__(self, initVals=None):
        HyperParams.__init__(self, self.proto, initVals)

class Activation(object):
    def __init__(self, params, batch_input_shape=None):
        self._params = ActivationParams(params)
        self._batch_input_shape = batch_input_shape

    def __call__(self, inp, layer_idx=None):
        ## Parameter Validation
        assert isinstance(inp, tf.Tensor)
        if self._batch_input_shape is not None:
            assert K.int_shape(inp) == self._batch_input_shape

        params = self._params        
        scope_name = 'Activation_%d'%(layer_idx+1) if layer_idx is not None else 'Activation'
        with tf.variable_scope(scope_name):
            h = params.activation_fn(inp)
            # Tensorboard Summaries
            if params.make_tb_metric:
                summarize_layer(scope_name, None, None, h)

        return h

class RNNParams(HyperParams):
    proto = (
            PD('type', 
               'Type of RNN cell to create. Only LSTM is supported at this time.',
               ('lstm',),
               'lstm'),
            PD('layers_units',
              "(sequence of integers): sequence of numbers denoting number of units for each layer." 
              "As many layers will be created as the length of the sequence",
              issequenceof(int)
              ),
            PD('activation_fn',
                  'Output activation function to use.', 
                  iscallable((tf.nn.tanh, tf.nn.sigmoid)),
                  tf.nn.tanh),
            PD('weights_initializer', 
                  'Tensorflow weights initializer function', 
                  iscallableOrNone(),
                  None
                  #tf.contrib.layers.xavier_initializer()
                  # tf.contrib.layers.xavier_initializer_conv2d()
                  # tf.contrib.layers.variance_scaling_initializer()
                  ),
            PD('use_peephole',
               '(boolean): whether to employ peephole connections in the decoder LSTM',
               (True, False),
               False),
            PD('forget_bias', '',
               decimal(),
               1.0),
            PD('dropout', 'Dropout parameters if any.',
                  instanceof(DropoutParams, True)),
            PD('tb', '',
               instanceof(TensorboardParams))
            )
    def __init__(self, initVals=None):
        HyperParams.__init__(self, self.proto, initVals)

class RNN(object):
    def __init__(self, params, batch_input_shape=None):
        self._params = RNNParams(params)
        self._batch_input_shape = batch_input_shape
        
        if len(params.layers_units) == 1:
            #The implementation is based on: http://arxiv.org/abs/1409.2329.
            self._LSTM_cell = tf.contrib.rnn.LSTMBlockCell(params.layers_units[0], 
                                                           forget_bias=params.forget_bias, 
                                                           use_peephole=params.use_peephole,
                                                           initializer=params.weights_initializer,
                                                           activation=params.activation_fn)
        if params.dropout is not None:
            tf.nn.rnn_cell.DropoutWrapper(self._LSTM_cell,
                                          input_keep_prob=params.keep_prob,
                                          state_keep_prob=params.keep_prob,
                                          output_keep_prob==params.keep_prob,
                                          variational_recurrent=True,
                                          input_size=batch_input_size[1:]
                                          )
        
        
    def __call__(self, inp, state):
        ## Parameter Validation
        assert isinstance(inp, tf.Tensor)
        if self._batch_input_shape is not None:
            self.assertInputSize(inp)
            self.assertStateSize(state)

        params = self._params
        with tf.variable_scope('LSTM'):
            

def makeTBDir(params):
    dir = params.tb_logdir + '/' + time.strftime('%Y-%m-%d %H-%M-%S %Z')
    os.makedirs(dir)
    return dir