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

Created on Sat Jul  8 19:33:38 2017
Tested on python 2.7

@author: Sumeet S Singh
"""

import collections
import itertools
import dl_commons as dlc
import tf_commons as tfc
import tensorflow as tf
from keras.applications.vgg16 import VGG16
#from keras.layers import Input, Embedding, Dense, Activation, Dropout, Concatenate, Permute
from keras import backend as K
from dl_commons import PD, mandatory, boolean, integer, decimal, equalto, instanceof
from CALSTM import CALSTM, CALSTMParams
from Im2LatexModelParams_2 import Im2LatexModelParams, HYPER

Im2LatexState = collections.namedtuple('Im2LatexState', ('rnn_state', 'yProbs'))
class Im2LatexModel(tf.nn.rnn_cell.RNNCell):
    """
    One timestep of the decoder model. The entire function can be seen as a complex RNN-cell
    that includes a LSTM stack and an attention model.
    """
    def __init__(self, params, context_op, beamsearch_width=1, reuse=None):
        """
        Args:
            params (Im2LatexModelParams)
            context_op (tensor): A op that outputs a batch of image-context vectors - i.e. output of 
                the conv-net. Note that this op should supply a new batch of context with every
                mini-batch. The context-vectors in the mini-batch should be aligned with the input
                word-sequence.
            beamsearch_width (integer): Only used when inferencing with beamsearch. Otherwise set it to 1.
                Will cause the batch_size in internal assert statements to get multiplied by beamwidth.
            reuse: Passed into the _reuse Tells the underlying RNNs whether or not to reuse the scope.
        """
#        self._numSteps = 0
        self._params = self.C = Im2LatexModelParams(params)
        assert K.int_shape(context_op) == (self.C.B, self.C.L, self.C.D)
        self.outer_scope = tf.get_variable_scope()
        with tf.variable_scope('Im2LatexModel') as scope:
            super(Im2LatexModel, self).__init__(_reuse=reuse, _scope=scope, name=scope.name)
            self.rnn_scope = scope
            ## Beam Width to be supplied to BeamsearchDecoder. It essentially broadcasts/tiles a
            ## batch of input from size B to B * BeamWidth. Set this value to 1 in the training
            ## phase.
            self._beamsearch_width = beamsearch_width

            self._a = context_op ## Image features from the Conv-Net
            if len(self.C.D_RNN) == 1:
                self._CALSTM_stack = CALSTM(self.C.D_RNN[0], context_op, beamsearch_width, reuse)
            else:
                self._CALSTM_stack = tf.nn.rnn_cell.MultiRNNCell(
                        [CALSTM(rnn_params, context_op, beamsearch_width, reuse) 
                        for rnn_params in self.C.D_RNN])
            self._define_params()

    @property
    def BeamWidth(self):
        return self._beamsearch_width
    

#    def _define_attention_params(self):
#        """ Define Shared Weights for Attention Model """
#        ## 1) Dense layer, 2) Optional gather layer and 3) softmax layer
#
#        ## Renaming HyperParams for convenience
#        B = HYPER.B
#        n = HYPER.n
#        L = HYPER.L
#        D = HYPER.D
#        
#        ## _att_dense array indices start from 1
#        self._att_dense_layer = []
#
#        if HYPER.att_share_weights:
#        ## Here we'll effectively create L MLP stacks all sharing the same weights. Each
#        ## stack receives a concatenated vector of a(l) and h as input.
#            dim = D+n
#            for i in range(1, HYPER.att_layers+1):
#                n_units = HYPER['att_%d_n'%(i,)]; assert(n_units <= dim)
#                self._att_dense_layer.append(Dense(n_units, activation=HYPER.att_activation,
#                                                   batch_input_shape=(B,L,dim)))
#                dim = n_units
#            ## Optional gather layer (that comes after the Dense Layer)
#            if HYPER.att_weighted_gather:
#                self._att_gather_layer = Dense(1, activation='linear') # output shape = (B, L, 1)
#        else:
#            ## concatenate a and h_prev and pass them through a MLP. This is different than the theano
#            ## implementation of the paper because we flatten a from (B,L,D) to (B,L*D). Hence each element
#            ## of the L*D vector receives its own weight because the effective weight matrix here would be
#            ## shape (L*D, num_dense_units) as compared to (D, num_dense_units) as in the shared_weights case
#            dim = L*D+n        
#            for i in range(1, HYPER.att_layers+1):
#                n_units = HYPER['att_%d_n'%(i,)]; assert(n_units <= dim)
#                self._att_dense_layer.append(Dense(n_units, activation=HYPER.att_actv,
#                                                   batch_input_shape=(B,dim)))
#                dim = n_units
#        
#        assert dim >= L
#        self._att_softmax_layer = Dense(L, activation='softmax', name='alpha')
#        
#    def _build_attention_model(self, a, h_prev):
#        B = HYPER.B
#        n = HYPER.n
#        L = HYPER.L
#        D = HYPER.D
#        h = h_prev
#
#        assert K.int_shape(h_prev) == (B, n)
#        assert K.int_shape(a) == (B, L, D)
#
#        ## For #layers > 1 this will endup being different than the paper's implementation
#        if HYPER.att_share_weights:
#            """
#            Here we'll effectively create L MLP stacks all sharing the same weights. Each
#            stack receives a concatenated vector of a(l) and h as input.
#
#            TODO: We could also
#            use 2D convolution here with a kernel of size (1,D) and stride=1 resulting in
#            an output dimension of (L,1,depth) or (B, L, 1, depth) including the batch dimension.
#            That may be more efficient.
#            """
#            ## h.shape = (B,n). Convert it to (B,1,n) and then broadcast to (B,L,n) in order
#            ## to concatenate with feature vectors of 'a' whose shape=(B,L,D)
#            h = K.tile(K.expand_dims(h, axis=1), (1,L,1))
#            ## Concatenate a and h. Final shape = (B, L, D+n)
#            ah = tf.concat([a,h], -1)
#            for i in range(HYPER.att_layers) :
#                ah = self._att_dense_layer[i](ah)
#
#            ## Below is roughly how it is implemented in the code released by the authors of the paper
##                 for i in range(1, HYPER.att_a_layers+1):
##                     a = Dense(HYPER['att_a_%d_n'%(i,)], activation=HYPER.att_actv)(a)
##                 for i in range(1, HYPER.att_h_layers+1):
##                     h = Dense(HYPER['att_h_%d_n'%(i,)], activation=HYPER.att_actv)(h)    
##                ah = a + K.expand_dims(h, axis=1)
#
#            ## Gather all activations across the features; go from (B, L, dim) to (B,L,1).
#            ## One could've just summed/averaged them all here, but the paper uses yet
#            ## another set of weights to accomplish this. So we'll keeep that as an option.
#            if HYPER.att_weighted_gather:
#                ah = self._att_gather_layer(ah) # output shape = (B, L, 1)
#                ah = K.squeeze(ah, axis=2) # output shape = (B, L)
#            else:
#                ah = K.mean(ah, axis=2) # output shape = (B, L)
#
#        else: # weights not shared across L
#            ## concatenate a and h_prev and pass them through a MLP. This is different than the theano
#            ## implementation of the paper because we flatten a from (B,L,D) to (B,L*D). Hence each element
#            ## of the L*D vector receives its own weight because the effective weight matrix here would be
#            ## shape (L*D, num_dense_units) as compared to (D, num_dense_units) as in the shared_weights case
#
#            ## Concatenate a and h. Final shape will be (B, L*D+n)
#            ah = K.concatenate(K.batch_flatten(a), h)
#            for i in range(HYPER.att_layers):
#                ah = self._att_dense_layer(ah)
#            ## At this point, ah.shape = (B, dim)
#
#        alpha = self._att_softmax_layer(ah) # output shape = (B, L)
#        assert K.int_shape(alpha) == (B, L)
#        return alpha
#            
#    def _define_output_params(self):
#        ## Renaming HyperParams for convenience
#        B = HYPER.B
#        n = HYPER.n
#        D = HYPER.D
#        m = HYPER.m
#        Kv= HYPER.K
#
#        ## First layer of output MLP
#        ## Affine transformation of h_t and z_t from size n/D to m followed by a summation
#        self._output_affine = Dense(m, activation='linear', batch_input_shape=(B,n+D)) # output size = (B, m)
#        ## non-linearity for the first layer - will be chained by the _call function after adding Ex / Ey
#        self._output_activation = Activation(HYPER.output_activation)
#
#        ## Additional layers if any
#        if HYPER.decoder_out_layers > 1:
#            self._output_dense = []
#            for i in range(1, HYPER.decoder_out_layers):
#                self._output_dense.append(Dense(m, activation=HYPER['output_%d_activation'%i], 
#                                           batch_input_shape=(B,m))
#                                         )
#
#        ## Final softmax layer
#        self._output_softmax = Dense(Kv, activation='softmax', batch_input_shape=(B,m))
        
#    def _build_output_layer(self, Ex_t, h_t, z_t):
#        ## Renaming HyperParams for convenience
#        B = HYPER.B
#        n = HYPER.n
#        D = HYPER.D
#        m = HYPER.m
#        Kv =HYPER.K
#        
#        assert K.int_shape(Ex_t) == (B, m)
#        assert K.int_shape(h_t) == (B, n)
#        assert K.int_shape(z_t) == (B, D)
#        
#        ## First layer of output MLP
#        ## Affine transformation of h_t and z_t from size n/D to size m followed by a summation
#        o_t = Dense(m, activation='linear', batch_input_shape=(B,n+D))(tf.concat([h_t, z_t], -1)) # output size = (B, m)
#        o_t = o_t + Ex_t
#        
#        ## non-linearity for the first layer
#        o_t = Activation(HYPER.output_activation)(o_t)
#
#        ## Subsequent MLP layers
#        if HYPER.decoder_out_layers > 1:
#            for i in range(1, HYPER.decoder_out_layers):
#                o_t = Dense(m, 
#                            activation=HYPER['output_%d_activation'%i], 
#                            batch_input_shape=(B,m))(o_t)
#                
#        ## Final logits
#        logits_t = Dense(Kv, activation=HYPER.output_activation, batch_input_shape=(B,m))(o_t) # shape = (B,K)
#        assert K.int_shape(logits_t) == (B, Kv)
#        
#        # softmax
#        return tf.nn.softmax(logits_t), logits_t

    def _output_layer(self, Ex_t, h_t, z_t):
        
        ## Renaming HyperParams for convenience
        CONF = self.C
        B = self.C.B*self.BeamWidth
        D = self.C.D
        m = self.C.m
        Kv =self.C.K
        n = self._CALSTM_stack.output_size
        
        assert K.int_shape(Ex_t) == (B, m)
        self._CALSTM_stack.assertOutputShape(h_t)
        assert K.int_shape(z_t) == (B, D)
        
        ## First layer of output MLP
        if CONF.output_follow_paper: ## Follow the paper.
            ## Affine transformation of h_t and z_t from size n/D to bring it down to m
            o_t = tfc.FCLayer({'num_units':m, 'activation_fn':None, 'tb':CONF.tb}, 
                              batch_input_shape=(B,n+D))(tf.concat([h_t, z_t], -1)) # o_t: (B, m)
            ## h_t and z_t are both dimension m now. So they can now be added to Ex_t.
            o_t = o_t + Ex_t # Paper does not multiply this with weights - weird.
            ## non-linearity for the first layer
            o_t = tfc.Activation(CONF, batch_input_shape=(B,m))(o_t)
            dim = m
        else: ## Use a straight MLP Stack
            o_t = K.concatenate((Ex_t, h_t, z_t)) # (B, m+n+D)
            dim = m+n+D

        ## Regular MLP layers
        assert CONF.output_layers.layers_units[-1] == Kv
        logits_t = tfc.MLPStack(CONF.output_layers, batch_input_shape=(B,dim))(o_t)
            
        assert K.int_shape(logits_t) == (B, Kv)
        
        return tf.nn.softmax(logits_t), logits_t

#    def _define_init_params(self):
#        ## As per the paper, this is a two-headed MLP. It has a stack of common layers at the bottom
#        ## two output layers at the top - one each for h and c LSTM states.
#        self._init_layer = []
#        self._init_dropout = []
#        for i in xrange(1, HYPER.init_layers):
#            key = 'init_%d_'%i
#            self._init_layer.append(Dense(HYPER[key+'n'], activation=HYPER[key+'activation']))
#            if HYPER[key+'dropout_rate'] > 0.0:
#                self._init_dropout.append(Dropout(HYPER[key+'dropout_rate']))
#
#        ## Final layer for h
#        self._init_h = Dense(HYPER['n'], activation=HYPER['init_h_activation'])
#        if HYPER['init_h_dropout_rate'] > 0.0:
#            self._init_h_dropout = Dropout(HYPER['init_h_dropout_rate'])
#
#        ## Final layer for c
#        self._init_c = Dense(HYPER['n'], activation=HYPER['init_c_activation'])
#        if HYPER['init_c_dropout_rate'] > 0.0:
#            self._init_c_dropout = Dropout(HYPER['init_c_dropout_rate'])
#
    def init_state_model(self):
        
        ################ Initializer MLP ################
        with tf.variable_scope(self.outer_scope):
            with tf.variable_scope('Initializer_MLP'):
    
                ## As per the paper, this is a multi-headed MLP. It has a base stack of common layers, plus
                ## one additional output layer for each of the h and c LSTM states. So if you had
                ## configured - say 2 LSTM cells in the attention-LSTM, you would end-up with 20 top-layers 
                ## on top of the base MLP stack. Base MLP stack is specified in param 'init_model'
                a = K.mean(self._a, axis=1) # final shape = (B, D)
                a = tfc.MLPStack(HYPER.init_model)(a)

                counter = itertools.count(1)
                def zero_to_init_state(s, counter):
                    assert dlc.issequence(s)
                    if hasattr(s, 'h'):
                        num_units = K.int_shape(s.h)[1]
                        layer_params = tfc.FCLayerParams(HYPER.init_model).updated({'num_units':num_units})
                        s = s._replace(h = tfc.FCLayer(layer_params)(a, counter.next()))
                    if hasattr(s, 'c'):
                        num_units = K.int_shape(s.c)[1]
                        layer_params = tfc.FCLayerParams(HYPER.init_model).updated({'num_units':num_units})
                        s = s._replace(c=tfc.FCLayer(layer_params)(a, counter.next()))
                        
                    lst = []

                    for i in xrange(len(s)):
                        if dlc.issequence(s[i]):
                            lst.append(zero_to_init_state(s[i], counter))
                        else:
                            lst.append(s[i])
                    
                    if hasattr(s, '_make'):
                        return s._make(lst)
                    else:
                        return tuple(lst)
                            
                with tf.variable_scope('Output_Layers'):
                    lstm_init_state = self._CALSTM_stack.zero_state(self.C.B*self.BeamWidth, dtype=self.C.dtype)
                    lstm_init_state = zero_to_init_state(lstm_init_state, counter)

        return Im2LatexState(lstm_init_state, None)
            
    def _embedding_lookup(self, ids):
        m = HYPER.m
        assert self._embedding_matrix is not None
        #assert K.int_shape(ids) == (B,)
        shape = list(K.int_shape(ids))
        embedded = tf.nn.embedding_lookup(self._embedding_matrix, ids)
        shape.append(m)
        ## Embedding lookup forgets the leading dimensions (e.g. (B,))
        ## Fix that here.
        embedded.set_shape(shape) # (...,m)
        return embedded    
    
    def _define_params(self):
        ## Renaming HyperParams for convenience
        B = HYPER.B
        n = HYPER.n
        m = HYPER.m
        Kv = HYPER.K
        e_init = HYPER.embeddings_initializer

#        ################ Attention Model ################
#        with tf.variable_scope('Attention'):
#            self._define_attention_params()
                
        ################ Embedding Layer ################
        with tf.variable_scope('Ey'):
#            self._embedding = Embedding(Kv, m, 
#                                        embeddings_initializer=e_init, 
#                                        mask_zero=True, 
#                                        input_length=1,
#                                        batch_input_shape=(B,1)
#                                        #input_shape=(1,)
#                                        ) ## (B, 1, m)
#            
            ## Above Embedding layer will get replaced by this one.
            self._embedding_matrix = tf.get_variable('Embedding_Matrix', (Kv, m))
        
#        ################ Decoder LSTM Cell ################
#        with tf.variable_scope('Decoder_LSTM'):
#            self._decoder_lstm = tf.contrib.rnn.LSTMBlockCell(n, forget_bias=1.0, 
#                                                              use_peephole=HYPER.decoder_lstm_peephole)
#            
#        ################ Output Layer ################
#        with tf.variable_scope('Decoder_Output_Layer'):
#            self._define_output_params()

#        ################ Initializer MLP ################
#        with tf.variable_scope('Initializer_MLP'):
#            self._define_init_params()
            
    def _build_rnn_step1(self, out_t_1, x_t, testing=False):
        """
        Builds tf graph for the first iteration of the RNN. Works for both training and testing graphs.
        """
        return self._build_rnn_step(out_t_1, x_t, isStep1=True, testing=testing)
        
    def _build_rnn_training_stepN(self, out_t_1, x_t):
        """
        Builds tf graph for the subsequent iterations of the RNN - training mode.
        """
        return self._build_rnn_step(out_t_1, x_t, isStep1=False, testing=False)
        
    def _build_rnn_testing_stepN(self, out_t_1, x_t):
        """
        Builds tf graph for the subsequent iterations of the RNN - testing mode.
        """
        return self._build_rnn_step(out_t_1, x_t, isStep1=False, testing=True)
    
    def call(self, Ex_t, state):
        """ 
        One step of the RNN API of this class.
        Layers a deep-output layer on top of CALSTM
        """
        ## State
        rnn_state_t_1 = state.rnn_state
        
        htop_t, rnn_state_t = self._CALSTM_stack(Ex_t, rnn_state_t_1)
        yProbs_t, yLogits_t = self._output_layer(Ex_t, htop_t, rnn_state_t.ztop)
        
        return yLogits_t, Im2LatexState(rnn_state_t, yProbs_t)
        
    def _scan_step_training(self, out_t_1, x_t):
        """
        Conforms to loop function fn required by tf.scan. Takes in previous lstm states (h and c), 
        the current input and the image annotations (a) as input. Returns the new states and outputs.
        Note that input(t) = Ex(t) = Ey(t-1). Input(t=0) = Null. When training, the target output is used for Ey
        whereas at prediction time (via. beam-search for e.g.) the actual output is used.
        Args:
            out_t_1 (tuple of tensors): Output returned by this function at previous time-step.
            x_t (tensor): is a input for one time-step. Should be a tensor of shape (batch-size, 1).
        Returns:
            out_t (tuple of tensors): The output y_t shape= (B,K) - the probability of words/tokens. Also returns
                states needed in the next iteration of the RNN - i.e. (h_t, lstm_states_t and a). lstm_states_t = 
                (h_t, c_t) - which means h_t is included twice in the returned tuple.            
        """
        with tf.variable_scope('Ey'):
            Ex_t = self._embedding_lookup(x_t)

        yLogits_t, state_t = self(Ex_t, out_t_1.state)
        
        return yLogits_t, state_t
    
    def train(self, x):
        """ Build the training graph of the model """
        pass
    
    def beamsearch(self, x_0):
        """ Build the prediction graph of the model using beamsearch """
        pass
        
        
    def _build_rnn_testing(self, a, y_s, init_c, init_h):
        return self._build_rnn(a, y_s, init_c, init_h, False)

    def _build_rnn_training(self, a, y_s, init_c, init_h):
        return self._build_rnn(a, y_s, init_c, init_h, True)

    def _build_rnn(self, a, y_s, init_c, init_h, training=True):
        B = HYPER.B
        L = HYPER.L
        D = HYPER.D
        n = HYPER.n
        assert K.int_shape(a) == (B, L, D)
        assert K.int_shape(y_s) == (B, None) # (B, T, 1)
        assert K.int_shape(init_c) == (B, n)
        assert K.int_shape(init_h) == (B, n)
        
        # LSTMStateTuple Stores two elements: (c, h), in that order.
        init_lstm_states = tf.contrib.rnn.LSTMStateTuple(init_c, init_h)

        ## tf.scan requires time-dimension to be the first dimension
        
        y_s = K.permute_dimensions(y_s, (1, 0)) # (T, B)

        ################ Build x_s ################
        ## First step of x_s is zero indicating begin-of-sequence
        x_s = tf.zeros((1, tf.shape(y_s)[1]), dtype=tf.int32)
        if training:
            ## x_s is y_s shifted forward by 1 timestep
            ## last time-step of y_s which is zero indicating <eos> will get removed.
            x_s = K.concatenate((x_s, y_s[0:-2]), axis=0)
            

        ################ Build RNN ################
        with tf.variable_scope('Decoder_RNN'):
            initial_accum = (0, init_h, init_lstm_states, a)
            ## Weights are created in first step and then reused in subsequent steps.
            ## Hence we need to separate them out.
            step1_out = self._build_rnn_step1(initial_accum, x_s[0], testing=(not training))
            ## Subsequent steps in training are different than validation/testing/prediction
            ## Hence we need to separate them
            if training:
                stepN_out = tf.scan(self._build_rnn_training_stepN, x_s[1:], initializer=step1_out)
            else:
                ## Uses y_t_1 as input instead of x_t
                stepN_out = tf.scan(self._build_rnn_testing_stepN, x_s[1:], initializer=step1_out)

            yProbs1, yLogits1, alpha1 = step1_out[4], step1_out[5], step1_out[6]
            yProbsN, yLogitsN, alphaN = stepN_out[4], stepN_out[5], stepN_out[6]

            yProbs = K.concatenate([K.expand_dims(yProbs1, axis=0), yProbsN], axis=0)
            yLogits = K.concatenate([K.expand_dims(yLogits1, axis=0), yLogitsN], axis=0)
            alpha = K.concatenate([K.expand_dims(alpha1, axis=0), alphaN], axis=0)
        
            ## Switch the batch dimension back to first position - (B, T, ...)
            yProbs = K.permute_dimensions(yProbs, [1,0,2])
            yLogits = K.permute_dimensions(yLogits, [1,0,2])
            alpha = K.permute_dimensions(alpha, [1,0,2])
            
        return yProbs, yLogits, alpha
        
    def _build_loss(self, yLogits, y_s, alpha, sequence_lengths):
        B = HYPER.B
        Kv =HYPER.K
        L = HYPER.L
        
        assert K.int_shape(yLogits) == (B, None, Kv) # (B, T, K)
        assert K.int_shape(y_s) == (B, None) # (B, T)
        assert K.int_shape(alpha) == (B, None, L) # (B, T, L)
        assert K.int_shape(sequence_lengths) == (B,) # (B,)
        
        ################ Build Cost Function ################
        with tf.variable_scope('Cost'):
            sequence_mask = tf.sequence_mask(sequence_lengths, maxlen=tf.shape(y_s)[1], dtype=tf.float32) # (B, T)

            ## Masked negative log-likelihood of the sequence.
            ## Note that log(product(p_t)) = sum(log(p_t)) therefore taking taking log of
            ## joint-sequence-probability is same as taking sum of log of probability at each time-step

            ## Compute Sequence Log-Loss / Log-Likelihood = -Log( product(p_t) ) = -sum(Log(p_t))
            if HYPER.sum_logloss:
                ## Here we do not normalize the log-loss across time-steps because the
                ## paper as well as it's source-code do not do that.
                loss_vector = tf.contrib.seq2seq.sequence_loss(logits=yLogits, 
                                                               targets=y_s, 
                                                               weights=sequence_mask, 
                                                               average_across_timesteps=False,
                                                               average_across_batch=True)
                loss = tf.reduce_sum(loss_vector) # scalar
            else: ## Standard log perplexity (average per-word)
                loss = tf.contrib.seq2seq.sequence_loss(logits=yLogits, 
                                                               targets=y_s, 
                                                               weights=sequence_mask, 
                                                               average_across_timesteps=True,
                                                               average_across_batch=True)

            ## Calculate the alpha penalty: lambda * sum_over_i(square(C/L - sum_over_t(alpha_i)))
            ## 
            if HYPER.MeanSumAlphaEquals1:
                mean_sum_alpha_i = 1.0
            else:
                mean_sum_alpha_i = tf.cast(sequence_lengths, dtype=tf.float32) / HYPER.L # (B,)

            sum_alpha_i = tf.reduce_sum(tf.multiply(alpha,sequence_mask), axis=1, keep_dims=False)# (B, L)
            squared_diff = tf.squared_difference(sum_alpha_i, mean_sum_alpha_i)
            penalty = HYPER.pLambda * tf.reduce_sum(squared_diff, keep_dims=False) # scalar
            
            cost = loss + penalty

        ################ Build CTC Cost Function ################
        with tf.variable_scope('CTC_Cost'):
            ## sparse tensor
            y_idx =    tf.where(tf.not_equal(y_s, 0))
            y_vals =   tf.gather_nd(y_s, y_idx)
            y_sparse = tf.SparseTensor(y_idx, y_vals, tf.shape(y_s, out_type=tf.int64))
            ctc_loss = tf.nn.ctc_loss(y_sparse, yLogits, sequence_lengths,
                           ctc_merge_repeated=False, time_major=False)

            ################ Build Scoring Function ################
            ## People have used BLEU score, but that probably is not suitable for markup comparison
            ## Best of course is to compare images produced by the output markup.
            
            ## Compute CTC score with intermediate blanks collapsed (we've collapsed all blanks in our
            ## train/test sequences to a single space so we'll hopefully get a better comparison by
            ## using CTC.)
        
        return cost, ctc_loss
    
    def _build_image_context(self, image_batch):
        ## Conv-net
        assert K.int_shape(image_batch) == (HYPER.B,) + HYPER.image_shape
        ################ Build VGG Net ################
        with tf.variable_scope('VGGNet'):
            # K.set_image_data_format('channels_last')
            convnet = VGG16(include_top=False, weights='imagenet', pooling=None, input_shape=HYPER.image_shape)
            convnet.trainable = False
            print 'convnet output_shape = ', convnet.output_shape
            a = convnet(image_batch)
            assert K.int_shape(a) == (HYPER.B, HYPER.H, HYPER.W, HYPER.D)

            ## Combine HxW into a single dimension L
            a = tf.reshape(a, shape=(HYPER.B or -1, HYPER.L, HYPER.D))
            assert K.int_shape(a) == (HYPER.B, HYPER.L, HYPER.D)
        
        return a
        
    def build(self):
        B = HYPER.B
        Kv = HYPER.K
        L = HYPER.L
        
        ## TODO: Introduce Stochastic Learning
        
        rv = dlc.Properties()
        im = tf.placeholder(dtype=tf.float32, shape=(HYPER.B,) + HYPER.image_shape, name='image_batch')
        y_s = tf.placeholder(tf.int32, shape=(HYPER.B, None))
        #a = tf.placeholder(tf.float32, shape=(HYPER.B, HYPER.L, HYPER.D))
        sequence_lengths = tf.placeholder(tf.int32, shape=(HYPER.B,))

        a = self._build_image_context(im)
        init_c, init_h = self._build_init_layer(a)
        yProbs, yLogits, alpha = self._build_rnn_training(a, y_s, init_c, init_h)
        self._build_rnn_testing(a, y_s, init_c, init_h)
        loss = self._build_loss(yLogits, y_s, alpha, sequence_lengths)
        
        assert K.int_shape(yProbs) == (B, None, Kv)
        assert K.int_shape(yLogits) == (B, None, Kv)
        assert K.int_shape(alpha) == (B, None, L)
        
        rv.im = im
        rv.y_s = y_s
        rv.Ts = sequence_lengths
        rv.yProbs = yProbs
        rv.yLogits = yLogits
        rv.alpha = alpha
        
        return rv.freeze()
        
