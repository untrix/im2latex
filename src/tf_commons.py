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
#        PD('make_tb_metric',"(boolean): Create tensorboard metrics.",
#              boolean,
#              True),
        PD('dropout', 'Dropout parameters if any.',
              instanceof(DropoutParams, True)),
        PD('weights_coll_name',
              'Name of trainable weights collection. There is no need to change the default = WEIGHTS',
              ('WEIGHTS',),
              'WEIGHTS'
              ),
        PD('tb', "Tensorboard Params.",
           instanceofOrNone(TensorboardParams))
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
            if params.tb is not None:
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
             CommonParamsProto.tb)
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
            if params.tb is not None:
                summarize_layer(scope_name, None, None, h)

        return h

def make_lstm_optname(_, d):
    opt_name = 'LSTMWrapper'
    try:
        if ("dropout" in d and d['dropout'] is not None and d['dropout']['keep_prob'] < 1.):
            opt_name = 'LSTMWrapper_w_Dropout'
    except:
        pass
    return opt_name

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
            PD('op_name',
               'Name of the layer; will show up in tensorboard visualization',
               None,
               make_lstm_optname
              ),
            PD('num_units', 
               "(integer): Either layers_units or num_units must be specified. Not both."
               "Number of units in the single-layer RNN. If you want to instantiate a single-layer RNN "
               "you may either specify num_units, or layers_units with a tuple of length one.",
               integerOrNone(1)),
            PD('layers_units',
              "(sequence of integers): Either layers_units or num_units must be specified. Not both."
              "Sequence of numbers denoting number of units for each layer." 
              "As many layers will be created as the length of the sequence",
              issequenceofOrNone(int)
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
               instanceofOrNone(DropoutParams)),
            PD('tb', '',
               instanceofOrNone(TensorboardParams))
            )
    def __init__(self, initVals=None):
        HyperParams.__init__(self, self.proto, initVals)
        ## Additional parameter validation
        b_num_units = self.num_units is not None
        b_layers_units = self.layers_units is not None
        if (b_num_units and b_layers_units):
            assert len(self.layers_units) == 1
            assert self.num_units == self.layers_units[0]
        else:
            assert (b_num_units or b_layers_units)
        
        ## If only one unit is specified, then make sure num_units and layers_units hold equivalent values
        ## so that client-code may refer to either.
        if b_num_units:
            self.layers_units = (self.num_units,)
        elif len(self.layers_units) == 1:
            self.num_units = self.layers_units[0]



class RNNWrapper(tf.nn.rnn_cell.RNNCell):
    def __init__(self, params, reuse=None, _scope=None):
        self._params = params = RNNParams(params)
        with tf.variable_scope(_scope or self._params.op_name) as scope:
            super(RNNWrapper, self).__init__(_reuse=reuse, _scope=scope, name=scope.name)
            
            if len(self._params.layers_units) == 1:
                self._cell = self._make_one_cell(self._params.layers_units[0])
            else: # len(params.layers_units) > 1
                self._cell = tf.nn.rnn_cell.MultiRNNCell(
                        [self._make_one_cell(n) for n in self._params.layers_units])
    
            self._batch_state_shape = self.recursive_expand_shape(self._cell.state_size, self._params.B)

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)

    @property
    def batch_input_shape(self):
        return (self._params.B, self._params.i)
        
    @property
    def batch_output_shape(self):
        return (self._params.B, self._cell.output_size)
        
    @property
    def batch_state_shape(self):
        return self._batch_state_shape
        
    @staticmethod
    def recursive_expand_shape(shape, B):
        """ Convert every scalar in the nested sequence into a (B, s) sequence """
        if not dlc.issequence(shape):
            return (B, shape)
        else:
            return tuple(RNNWrapper.recursive_expand_shape(s, B) for s in shape)
        
    @staticmethod
    def recursive_get_shape(obj):
        """ Get shape of possibly nested sequence of tensors """
        if not dlc.issequence(obj):
            return K.int_shape(obj)
        else:
            return tuple(RNNWrapper.recursive_get_shape(o) for o in obj)
        
    def assertStateShape(self, state):
        """ 
        Asserts that the shape of the tensor is consistent with the RNN's state shape.
        For e.g. the state-shape of a MultiRNNCell with L layers would be ((n1, n1), (n2, n2) ... (nL, nL))
        and the corresponding state-tensor should be of shape :
        (((B,n1),(B,n1)), ((B,n2),(B,n2)) ... ((B,nL),(B,nL)))
        """
        try:
            assert self.batch_state_shape == self.recursive_get_shape(state)
        except:
            print 'state-shape assertion Failed for state type = ', type(state), ' and value = ', state
            raise
        
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
        return self.recursive_expand_shape(shape, self._params.B) == K.int_shape(tnsr)
    
    def call(self, inp, state):
        ## Parameter Validation
        assert isinstance(inp, tf.Tensor)
        self.assertInputShape(inp)
        self.assertStateShape(state)

        params = self._params
#        with tf.variable_scope(self._var_scope) as scope:
        output, new_state = self._cell(inp, state)
        # Tensorboard Summaries
        if params.tb is not None:
            summarize_layer(self._params.op_name, tf.get_collection('weights'), 
                            tf.get_collection('biases'), output)
            summarize_layer(self._params.op_name, None, None, new_state)

        self.assertOutputShape(output)
        self.assertStateShape(new_state)
        return (output, new_state)
            
    def _make_one_cell(self, num_units):
        params = self._params
#        with tf.variable_scope(self._var_scope):
        #The implementation is based on: http://arxiv.org/abs/1409.2329.
        cell = tf.contrib.rnn.LSTMBlockCell(num_units, 
                                           forget_bias=params.forget_bias, 
                                           use_peephole=params.use_peephole
                                           )
        if params.dropout is not None and params.dropout.keep_prob < 1.:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                          input_keep_prob=params.keep_prob,
                                          state_keep_prob=params.keep_prob,
                                          output_keep_prob=params.keep_prob,
                                          variational_recurrent=True,
                                          input_size=params.input_size
                                          )
        return cell

def makeTBDir(params):
    dir = params.tb_logdir + '/' + time.strftime('%Y-%m-%d %H-%M-%S %Z')
    os.makedirs(dir)
    return dir