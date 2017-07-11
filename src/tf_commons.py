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
from dl_commons import ParamDesc as PD
from dl_commons import mandatory, instanceof, equalto, boolean, HyperParams


class tensor(dlc.instanceof):
    """Tensor shape validator to go with ParamDesc"""
    def __init__(self, shape):
        self._shape = shape
        dlc._ParamValidator.__init__(self, tf.Tensor)
    def __contains__(self, obj):
        return dlc.instanceof.__contains__(self, obj) and  keras.backend.int_shape(obj) == self._shape
        

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
        summarize_vars(weights, 'Weights')
        summarize_vars(biases, 'Biases')
        summarize_vars(activations, 'Activations')

class TensorboardParams(dlc.HyperParams):
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
              dlc.decimal(0.1,1.),
              0.9
              ),
        PD('seed',
              'Integer seed for the random number generator',
              dlc.integerOrNone(),
              None
              )
        )
    def __init__(self, initVals=None):
        HyperParams.__init__(self, self.proto, initVals)

class MLPParams(HyperParams):
    proto = (
        PD('op_name',
              'Name of the layer; will show up in tensorboard visualization'
              ),
        PD('num_units', 
              "(sequence of integers): sequence of numbers denoting number of units for each layer." 
              "As many layers will be created as the length of the sequence",
              dlc.issequenceof(int)
              ),
        PD('activation_fn',   
              'The activation function to use. None signifies no activation function.', 
              dlc.iscallable((tf.nn.relu, tf.nn.tanh, tf.nn.sigmoid, None)),
              tf.nn.tanh),
        PD('normalizer_fn',
              'Normalizer function to use instead of biases. If set, biases are not used.', 
              (None,) 
              ),
        PD('weights_initializer', 
              'Tensorflow weights initializer function', 
              dlc.iscallable(noneokay=True),
              tf.contrib.layers.xavier_initializer()
              # tf.contrib.layers.xavier_initializer_conv2d()
              # tf.contrib.layers.variance_scaling_initializer()
              ),
        PD('biases_initializer', 
              'Tensorflow biases initializer function, e.g. tf.zeros_initializer(). ',
              dlc.iscallable(),
              tf.zeros_initializer()
              ),
        PD('weights_regularizer',
              'Defined in tf.contrib.layers. Not sure what this is, but probably a normalizer?',
              dlc.iscallable(noneokay=True), None),
        PD('biases_regularizer',
              'Defined in tf.contrib.layers. Not sure what this is, but probably a normalizer?',
              dlc.iscallable(noneokay=True), None),
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
        dlc.HyperParams.__init__(self, self.proto, initVals)

class MLP(object):
    def __init__(self, params):
        self._params = MLPParams(params)
        
    def __call__(self, inp):
        params = self._params
        dropout = params.dropout is not None
        
        a = inp
        with tf.variable_scope(params.op_name):
            i = 1
            for num_units in self._params.num_units:
                a = self._make_fc_layer(num_units, params, a, i)
                if dropout:
                    a = self._make_dropout_layer(self._params.dropout, a, i)
                i += 1
        return a
            
    def _make_fc_layer(self, num_units, params, inp, i=1):
        assert isinstance(inp, tf.Tensor)

        with tf.variable_scope('FC_%d'%i) as var_scope:
            layer_name = var_scope.name
            coll_w = layer_name + '/' + params.tb.tb_weights
            coll_b = layer_name + '/' + params.tb.tb_biases
            
            h = tf.contrib.layers.fully_connected(
                    inputs=inp,
                    num_outputs = num_units,
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

    def _make_dropout_layer(self, params, inp, i=1):
        assert isinstance(inp, tf.Tensor)
        
        with tf.variable_scope('DO_%d'%i):
            h = tf.nn.dropout(inp, params.keep_prob, seed=params.seed)
    
        return h

def makeTBDir(params):
    dir = params.tb_logdir + '/' + time.strftime('%Y-%m-%d %H-%M-%S %Z')
    os.makedirs(dir)
    return dir