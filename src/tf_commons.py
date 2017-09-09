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
import numpy as np
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


def summarize_layer(layer_name, weights, biases, activations, state=None):
    def summarize_vars(var, section_name):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        # mean = tf.reduce_mean(var)
        # tf.summary.scalar(section_name+'/mean', mean)
        # stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # tf.summary.scalar(section_name+'/stddev', stddev)
        # tf.summary.scalar(section_name+'/max', tf.reduce_max(var))
        # tf.summary.scalar(section_name+'/min', tf.reduce_min(var))
        tf.summary.histogram(section_name+'/histogram', var)

    with tf.variable_scope('Instrumentation'):
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
    def __copy__(self):
        ## Shallow copy
        return self.__class__(self)

    def copy(self, override_vals=None):
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
              integerOrNone()
              )
        )
    def __init__(self, initVals=None):
        HyperParams.__init__(self, self.proto, initVals)
    def __copy__(self):
        ## Shallow copy
        return self.__class__(self)

    def copy(self, override_vals=None):
        ## Shallow copy
        return self.__class__(self).updated(override_vals)

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
              'L1 / L2 norm regularization',
              iscallable(noneokay=True), 
              None ## Trickle down value from above
              # tf.contrib.layers.l2_regularizer(scale, scope=None)
              # tf.contrib.layers.l1_regularizer(scale, scope=None)
              ),
        PD('biases_regularizer',
              'L1 / L2 norm regularization',
              iscallable(noneokay=True), None),
#        PD('make_tb_metric',"(boolean): Create tensorboard metrics.",
#              boolean,
#              True),
        PD('dropout', 'Dropout parameters if any.',
              instanceofOrNone(DropoutParams)),
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
    def __copy__(self):
        ## Shallow copy
        return self.__class__(self)

    def copy(self, override_vals=None):
        ## Shallow copy
        return self.__class__(self).updated(override_vals)

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
    def __copy__(self):
        ## Shallow copy
        return self.__class__(self)

    def copy(self, override_vals=None):
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
                with tf.variable_scope(params.op_name):
                    for i in xrange(len(self._params.layers_units)):
                        a = FCLayer(self._params.get_layer_params(i))(a, i)
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
    def __copy__(self):
        ## Shallow copy
        return self.__class__(self)

    def copy(self, override_vals=None):
        ## Shallow copy
        return self.__class__(self).updated(override_vals)

class FCLayer(object):
    def __init__(self, params, batch_input_shape=None):
        self.my_scope = tf.get_variable_scope()
        self._params = FCLayerParams(params)
        self._batch_input_shape = batch_input_shape

    def __call__(self, inp, layer_idx=None):
        with tf.variable_scope(self.my_scope) as var_scope:
            with tf.name_scope(var_scope.original_name_scope):
                ## Parameter Validation
                assert isinstance(inp, tf.Tensor)
                if self._batch_input_shape is not None:
                    assert K.int_shape(inp) == self._batch_input_shape

                params = self._params
                prefix = 'FC' if params.activation_fn is not None else 'Affine'
                scope_name = prefix + '_%d'%(layer_idx+1) if layer_idx is not None else prefix
                with tf.variable_scope(scope_name) as var_scope:
                    layer_name = var_scope.name
        #            coll_w = layer_name + '/' + params.tb.tb_weights
        #            coll_b = layer_name + '/' + params.tb.tb_biases

                    a = tf.contrib.layers.fully_connected(
                            inputs=inp,
                            num_outputs = params.num_units,
                            activation_fn = params.activation_fn,
                            normalizer_fn = params.normalizer_fn,
                            weights_initializer = params.weights_initializer,
                            weights_regularizer = params.weights_regularizer,
                            biases_initializer = params.biases_initializer,
                            biases_regularizer = params.biases_regularizer,
        #                    variables_collections = {"weights":[coll_w, params.weights_coll_name],
        #                                             "biases":[coll_b]},
                            trainable = True
                            )

                    if self._params.dropout is not None:
                        a = DropoutLayer(self._params.dropout, self._batch_input_shape)(a)

                    # Tensorboard Summaries
                    # if params.tb is not None:
                    #     summarize_layer(layer_name, None, None, a)
                    # if params.tb is not None:
                    #     summarize_layer(layer_name, tf.get_collection('weights'), tf.get_collection('biases'), a)

                if (self._batch_input_shape):
                    a.set_shape(self._batch_input_shape[:-1] + (params.num_units,))

                return a

class Conv2dLayerParams(HyperParams):
    proto = (
        PD('activation_fn',
              'The activation function to use. None signifies no activation function.',
              iscallable((tf.nn.relu, tf.nn.tanh, None)),
              tf.nn.tanh),
        PD('normalizer_fn',
              'Normalizer function to use instead of biases. If set, biases are not used.',
              (None,)
              ),
        PD('weights_initializer',
              'Tensorflow weights initializer function',
              iscallableOrNone(),
              tf.contrib.layers.xavier_initializer_conv2d()
              # tf.contrib.layers.variance_scaling_initializer()
              ),
        PD('biases_initializer',
              'Tensorflow biases initializer function, e.g. tf.zeros_initializer(). ',
              iscallable(),
              tf.zeros_initializer()
              ),
        PD('output_channels',
          "(integer): Depth of output layer = Number of features/channels in the output." ,
          integer(1),
          ),
        PD('kernel_shape',
           '(sequence): shape of kernel',
           isSequenceOf(integer)
           ),
        PD('stride',
           '(sequence): shape of stride',
           isSequenceOf(integer)
           ),
        PD('padding',
           'Convnet padding: SAME or VALID',
           ('SAME', 'VALID', None)
           ),
        PD('tb', "Tensorboard Params.",
           instanceofOrNone(TensorboardParams)
           )
        )

    def __init__(self, initVals=None):
        HyperParams.__init__(self, self.proto, initVals)
    def __copy__(self):
        ## Shallow copy
        return self.__class__(self)

    def copy(self, override_vals=None):
        ## Shallow copy
        return self.__class__(self).updated(override_vals)

class Conv2dLayer(object):
    def __init__(self, params, batch_input_shape=None):
        self.my_scope = tf.get_variable_scope()
        self._params = FCLayerParams(params)
        self._batch_input_shape = batch_input_shape

    def __call__(self, inp, layer_idx=None):
        with tf.variable_scope(self.my_scope) as var_scope:
            with tf.name_scope(var_scope.original_name_scope):
                ## Parameter Validation
                assert isinstance(inp, tf.Tensor)
                if self._batch_input_shape is not None:
                    assert K.int_shape(inp) == self._batch_input_shape

                params = self._params
                prefix = 'Conv'
                scope_name = prefix + '_%d'%(layer_idx+1) if layer_idx is not None else prefix
                with tf.variable_scope(scope_name) as var_scope:
                    layer_name = var_scope.name
        #            coll_w = layer_name + '/' + params.tb.tb_weights
        #            coll_b = layer_name + '/' + params.tb.tb_biases

                    a = tf.contrib.layers.conv2d(inputs=a,
                                                num_outputs=params.output_channels, 
                                                kernel_size=params.kernel_shape, 
                                                stride=params.stride,
                                                padding=params.padding,
                                                activation_fn=params.activation_fn,
                                                normalizer_fn=params.normalizer_fn,
                                                scope=scope,
                                                trainable=True,
                                                weights_initializer=params.weights_initializer_conv2d(),
                                                biases_initializer=params.biases_initializer
                                                )

                    # Tensorboard Summaries
                    # if params.tb is not None:
                    #     summarize_layer(layer_name, None, None, a)
                    # if params.tb is not None:
                    #     summarize_layer(layer_name, tf.get_collection('weights'), tf.get_collection('biases'), a)

                # if (self._batch_input_shape):
                #     a.set_shape(self._batch_input_shape[:-1] + (params.num_units,))

                return a

class DropoutLayer(object):
    def __init__(self, params, batch_input_shape=None):
        self.my_scope = tf.get_variable_scope()
        self._params = DropoutParams(params)
        self._batch_input_shape = batch_input_shape

    def __call__(self, inp, layer_idx=None):
        with tf.variable_scope(self.my_scope) as var_scope:
            with tf.name_scope(var_scope.original_name_scope):
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
    def __copy__(self):
        ## Shallow copy
        return self.__class__(self)

    def copy(self, override_vals=None):
        ## Shallow copy
        return self.__class__(self).updated(override_vals)

class Activation(object):
    def __init__(self, params, batch_input_shape=None):
        self.my_scope = tf.get_variable_scope()
        self._params = ActivationParams(params)
        self._batch_input_shape = batch_input_shape

    def __call__(self, inp, layer_idx=None):
        with tf.variable_scope(self.my_scope) as var_scope:
            with tf.name_scope(var_scope.original_name_scope):
                ## Parameter Validation
                assert isinstance(inp, tf.Tensor)
                if self._batch_input_shape is not None:
                    assert K.int_shape(inp) == self._batch_input_shape

                params = self._params
                scope_name = 'Activation_%d'%(layer_idx+1) if layer_idx is not None else 'Activation'
                with tf.variable_scope(scope_name):
                    h = params.activation_fn(inp)
                    # Tensorboard Summaries
                    # if params.tb is not None:
                    #     summarize_layer(scope_name, None, None, h)

                return h

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
            PD('activation_fn',
              'Output activation function to use.',
              iscallable((tf.nn.tanh, tf.nn.sigmoid)),
              tf.nn.tanh),
            PD('weights_initializer',
              'Tensorflow weights initializer function',
              iscallableOrNone(),
              tf.contrib.layers.xavier_initializer()
              # tf.contrib.layers.xavier_initializer_conv2d()
              # tf.contrib.layers.variance_scaling_initializer()
              ),
            PD('weights_regularizer',
                'L1 / L2 norm regularization',
                iscallable(noneokay=True), 
                None ## Trickle down value from above
                # tf.contrib.layers.l2_regularizer(scale, scope=None)
                # tf.contrib.layers.l1_regularizer(scale, scope=None)
                ),
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

    def copy(self, override_vals=None):
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



class RNNWrapper(tf.nn.rnn_cell.RNNCell):
    def __init__(self, params, reuse=None, _scope=None, beamsearch_width=1):
        self._params = params = RNNParams(params)
        with tf.variable_scope(_scope or self._params.op_name,
                               initializer=self._params.weights_initializer,
                               regularizer=self._params.weights_regularizer) as scope:
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
        try:
            assert self.batch_state_shape == get_nested_shape(state)
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


def makeTBDir(params):
    dir = params.tb_logdir + '/' + time.strftime('%Y-%m-%d %H-%M-%S %Z')
    os.makedirs(dir)
    return dir

def sizeofVar(var):
    shape = K.int_shape(var)
    return np.prod(shape)

def printVars(name, coll):
    print name
    total_n = 0
    for var in tf.trainable_variables():
        n = sizeofVar(var)
        total_n += n
        print var.name, K.int_shape(var), 'num_params = ', n
    print '\nTotal number of variables = ', total_n
    return total_n

def edit_distance3D(B, k, predicted_ids, predicted_lens, target_ids, target_lens, blank_token=None):
    with tf.name_scope('edit_distance3D'):
        p_shape = K.int_shape(predicted_ids)
        t_shape = K.int_shape(target_ids)
        assert len(p_shape) == 3
        assert p_shape[:2] == (B, k)
        assert K.int_shape(predicted_lens) == (B, k)
        assert len(t_shape) == 3
        assert t_shape[:2] == (B, k)
        assert K.int_shape(target_lens) == (B, k)
        predicted_sparse = dense_to_sparse3D(B, predicted_ids, predicted_lens, blank_token)
        ## blank tokens should not be present in target_ids
        target_sparse = dense_to_sparse3D(B, target_ids, target_lens)

        d = tf.edit_distance(predicted_sparse, target_sparse)
        # assert K.int_shape(d) == K.int_shape(predicted_lens)
        d.set_shape(predicted_lens.shape) ## Reassert shape in case it got lost.
        return d

def edit_distance2D(B, predicted_ids, predicted_lens, target_ids, target_lens, blank_token=None):
    with tf.name_scope('edit_distance2D'):
        p_shape = K.int_shape(predicted_ids)
        t_shape = K.int_shape(target_ids)
        assert len(p_shape) == 2
        assert p_shape[0] == B
        assert K.int_shape(predicted_lens) == (B,)
        assert len(t_shape) == 2
        assert t_shape[0] == B
        assert K.int_shape(target_lens) == (B,)
        predicted_sparse = dense_to_sparse2D(predicted_ids, predicted_lens, blank_token)
        ## blank tokens should not be present in target_ids
        target_sparse = dense_to_sparse2D(target_ids, target_lens)
        d = tf.edit_distance(predicted_sparse, target_sparse)
        # assert K.int_shape(d) == K.int_shape(predicted_lens)
        d.set_shape(predicted_lens.shape) ## Reassert shape in case it got lost.
        return d


def squash_2d(B, m, lens, blank_token, padding_token=0):
    """
    Takes in a matrix shaped (B, T), from each row removes blank_tokens and appends
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
            mask_i = tf.logical_and( seq_mask[i], tf.not_equal(m_i, blank_token)) #(T,)
            idx_i = tf.where(mask_i) # (N, 1)
            vals_i = tf.gather_nd(m_i, idx_i) # (N, )
            len_i = tf.shape(vals_i)[0]
            paddings = [(0, T - len_i)] # (1, 2)
            squashed.append(tf.pad(vals_i, paddings, mode="CONSTANT", name=None, constant_values=padding_token))
            squashed_lens.append(len_i)
        return tf.stack(squashed), tf.stack(squashed_lens)

def squash_3d(B, k, m, lens, blank_token, padding_token=0):
    """
    Takes in a matrix shaped (B, k, T), from each sequence of length T removes blank_tokens and appends
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

def _dense_to_sparse(t, mask, blank_token=None):
        if blank_token is not None:
            mask = tf.logical_and(mask, tf.not_equal(t, blank_token))
        idx = tf.where(mask)
        vals = tf.gather_nd(t, idx)
        return tf.SparseTensor(idx, vals, tf.shape(t, out_type=tf.int64))    

def dense_to_sparse2D(t, lens, blank_token=None):
    with tf.name_scope('dense_to_sparse2D'):
        assert len(K.int_shape(t)) == 2
        assert len(K.int_shape(lens)) == 1
        mask = tf.sequence_mask(lens, maxlen=tf.shape(t)[1])
        return _dense_to_sparse(t, mask, blank_token)

def dense_to_sparse3D(B, t, lens, blank_token=None):
    with tf.name_scope('dense_to_sparse3D'):
        assert K.int_shape(t)[0] == B
        assert len(K.int_shape(t)) == 3
        assert len(K.int_shape(lens)) == 2
        mask = tf.stack([tf.sequence_mask(lens[i], maxlen=tf.shape(t)[2]) for i in range(B)])
        return _dense_to_sparse(t, mask, blank_token)

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

