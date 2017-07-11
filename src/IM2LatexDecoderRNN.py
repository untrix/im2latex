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
import dl_commons as dlc
import tf_commons as tfc
import tensorflow as tf
from keras.layers import Input, Embedding, Dense, Activation, Dropout, Concatenate, Permute
from keras import backend as K
import collections

Im2LatexRNNStateTuple = collections.namedtuple("Im2LatexRNNStateTuple", ('lstm_state', 'alpha'))

class Im2LatexDecoderRNN(tf.nn.rnn_cell.RNNCell):
    """
    One timestep of the decoder model. The entire function can be seen as a complex RNN-cell
    that includes a LSTM stack and an attention model.
    """

    def __init__(self, config, context, beamsearch_width, reuse=None):
        assert K.int_shape(context) == (config.B, config.L, config.D)
        super(Im2LatexDecoderRNN, self).__init__(_reuse=reuse)
        self.C = config.copy().freeze()

        ## Beam Width to be supplied to BeamsearchDecoder. It essentially broadcasts/tiles a
        ## batch of input from size B to B * BeamWidth. Set this value to 1 in the training
        ## phase.
        self._beamsearch_width = beamsearch_width

        self._a = context ## Image features from the Conv-Net
        ## Broadcast context from size B to B*BeamWidth, because that's what BeamSearchDecoder does
        ## to the input batch.
        self._a = K.tile(self._a, (beamsearch_width,1,1))

        #LSTM by Zaremba et. al 2014: http://arxiv.org/abs/1409.2329
        self._LSTM_cell = tf.contrib.rnn.LSTMBlockCell(self.C.n, 
                                                       forget_bias=1.0, 
                                                       use_peephole=self.C.decoder_lstm_peephole)


    @property
    def BeamWidth(self):
        return self._beamsearch_width
    
    @property
    def state_size(self):
        n = self.C.n
        L = self.C.L
    
        # lstm_states_t, alpha_t
        #return Im2LatexRNNStateTuple(tf.nn.rnn_cell.LSTMStateTuple(n, n), L)
        return ((n,n), L)

    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            return (tuple(self._LSTM_cell.zero_state(batch_size, dtype)), 
                    tf.zeros((batch_size, self.C.L), dtype=dtype))

    @property
    def output_size(self):
        # yLogits
        return self.C.K
       
    def _attention_model(self, a, h_prev):
        CONF = self.C
        B = CONF.B*self.BeamWidth
        n = CONF.n
        L = CONF.L
        D = CONF.D
        h = h_prev

        assert K.int_shape(h_prev) == (B, n)
        assert K.int_shape(a) == (B, L, D)

        ## For #layers > 1 this will endup being different than the paper's implementation
        if CONF.att_share_weights:
            """
            Here we'll effectively create L MLP stacks all sharing the same weights. Each
            stack receives a concatenated vector of a(l) and h as input.

            TODO: We could also
            use 2D convolution here with a kernel of size (1,D) and stride=1 resulting in
            an output dimension of (L,1,depth) or (B, L, 1, depth) including the batch dimension.
            That may be more efficient.
            """
            ## h.shape = (B,n). Convert it to (B,1,n) and then broadcast to (B,L,n) in order
            ## to concatenate with feature vectors of 'a' whose shape=(B,L,D)
            h = K.tile(K.expand_dims(h, axis=1), (1,L,1))
            ## Concatenate a and h. Final shape = (B, L, D+n)
            ah = tf.concat([a,h], -1); # dim = D+n
            ah = tfc.MLP(CONF.att_layers)(ah); dim = CONF.att_layers.num_units[-1]
#            for i in range(1, CONF.att_layers+1):
#                n_units = CONF['att_%d_n'%(i,)]; assert(n_units <= dim)
#                ah = Dense(n_units, activation=CONF.att_activation, batch_input_shape=(B,L,dim))(ah)
#                dim = n_units
            assert K.int_shape(ah) == (B, L, dim)
                
            ## Below is roughly how it is implemented in the code released by the authors of the paper
            ##     for i in range(1, CONF.att_a_layers+1):
            ##         a = Dense(CONF['att_a_%d_n'%(i,)], activation=CONF.att_actv)(a)
            ##     for i in range(1, CONF.att_h_layers+1):
            ##         h = Dense(CONF['att_h_%d_n'%(i,)], activation=CONF.att_actv)(h)    
            ##    ah = a + K.expand_dims(h, axis=1)

            ## Gather all activations across the features; go from (B, L, dim) to (B,L,1).
            ## One could've just summed/averaged them all here, but the paper uses yet
            ## another set of weights to accomplish this. So we'll keeep that as an option.
            if CONF.att_weighted_gather:
                ah = Dense(1, activation='linear')(ah) # output shape = (B, L, 1)
                ah = K.squeeze(ah, axis=2) # output shape = (B, L)
            else:
                ah = K.mean(ah, axis=2) # output shape = (B, L)
                
            alpha = tf.nn.softmax(ah) # output shape = (B, L)
            
        else: # weights not shared across L
            ## concatenate a and h_prev and pass them through a MLP. This is different than the theano
            ## implementation of the paper because we flatten a from (B,L,D) to (B,L*D). Hence each element
            ## of the L*D vector receives its own weight because the effective weight matrix here would be
            ## shape (L*D, num_dense_units) as compared to (D, num_dense_units) as in the shared_weights case

            ## Concatenate a and h. Final shape will be (B, L*D+n)
            ah = K.concatenate(K.batch_flatten(a), h) # dim = L*D+n
            ah = tfc.MLP(CONF.att_layers)(ah); dim = CONF.att_layers.num_units[-1]
#            for i in range(1, CONF.att_layers+1):
#                n_units = CONF['att_%d_n'%(i,)]; assert(n_units <= dim)
#                ah = Dense(n_units, activation=CONF.att_actv, batch_input_shape=(B,dim))(ah)
#                dim = n_units
            ## At this point, ah.shape = (B, dim)        
            assert K.int_shape(ah) == (B, dim)
            assert dim >= L        
            ## NOTE: An extra dense layer is not needed if dim == L. Simply a softmax activation would
            ## suffice in that case.
            alpha = self.Dense(L, activation='softmax', name='alpha')(ah) # output shape = (B, L)
        
        assert K.int_shape(alpha) == (B, L)
        return alpha

    def _build_decoder_lstm(self, Ex_t, z_t, lstm_states_t_1):
        """Represents invocation of the decoder lstm. (h_t, lstm_states_t) = *(z_t|Ex_t, lstm_states_t_1)"""
        m = self.C.m
        D = self.C.D
        B = self.C.B*self.BeamWidth
        n = self.C.n
        
        inputs_t = K.concatenate((Ex_t, z_t))
        assert K.int_shape(inputs_t) == (B, m+D)
        assert K.int_shape(lstm_states_t_1[1]) == (B, n)
        
        ## TODO: Make this multi-layered
        (h_t, lstm_states_t) = self._LSTM_cell(inputs_t, lstm_states_t_1)
        return (h_t, lstm_states_t)

    def _build_output_layer(self, Ex_t, h_t, z_t):
        
        ## Renaming HyperParams for convenience
        CONF = self.C
        B = self.C.B*self.BeamWidth
        n = self.C.n
        D = self.C.D
        m = self.C.m
        Kv =self.C.K
        
        assert K.int_shape(Ex_t) == (B, m)
        assert K.int_shape(h_t) == (B, n)
        assert K.int_shape(z_t) == (B, D)
        
        ## First layer of output MLP
        if not CONF.output_follow_paper: ## Follow the paper.
            ## Affine transformation of h_t and z_t from size n/D to bring it down to m
            o_t = Dense(m, activation='linear', batch_input_shape=(B,n+D))(tf.concat([h_t, z_t], -1)) # o_t: (B, m)
            ## h_t and z_t are both dimension m now. So they can now be added to Ex_t.
            o_t = o_t + Ex_t # Paper does not multiply this with weights - weird.
            ## non-linearity for the first layer
            o_t = Activation(CONF.output_activation)(o_t)
            dim = m
        else: ## Use a straight FC layer
            o_t = K.concatenate((Ex_t, h_t, z_t)) # (B, m+n+D)
            o_t = Dense(CONF.output_1_n, activation=CONF.output_activation, batch_input_shape=(B,D+m+n))(o_t)
            dim = CONF.output_1_n
            
        ## Subsequent MLP layers
        if CONF.decoder_out_layers > 1:
            for i in range(2, CONF.decoder_out_layers+1):
                o_t = Dense(m, activation=CONF.output_activation, 
                            batch_input_shape=(B,dim))(o_t)
                
        ## Final logits layer
        logits_t = Dense(Kv, activation=CONF.output_activation, batch_input_shape=(B,m))(o_t) # shape = (B,K)
        assert K.int_shape(logits_t) == (B, Kv)
        
        # return tf.nn.softmax(logits_t), logits_t
        return logits_t

    def call(self, inputs, state):
        """
        TODO: Incorporate Dropout
        Builds/threads tf graph for one RNN iteration.
        Takes in previous lstm states (h and c),
        the current input and the image annotations (a) as input and outputs the states and outputs for the
        current timestep.
        Note that input(t) = Ey(t-1). Input(t=0) = Null. When training, the target output is used for Ey
        whereas at prediction time (via. beam-search for e.g.) the actual output is used.
        """

        Ex_t = inputs                          # shape = (B,m)
        state = Im2LatexRNNStateTuple(state[0], state[1])
        lstm_states_t_1 = state.lstm_state   # shape = ((B,n), (B,n)) = (c_t_1, h_t_1)
        alpha_t_1 = state.alpha            # shape = (B, L)
        h_t_1 = lstm_states_t_1[1]
        a = self._a

        CONF = self.C
        B = CONF.B*self.BeamWidth
        m = CONF.m
        n = CONF.n
        L = CONF.L
        Kv =CONF.K

        print 'shape(Ex_t) = ', K.int_shape(Ex_t)
        assert K.int_shape(Ex_t) == (B,m)
        assert K.int_shape(h_t_1) == (B, n)
        assert K.int_shape(lstm_states_t_1[1]) == (B, n)
        assert K.int_shape(alpha_t_1) == (B, L)

        ################ Attention Model ################
        with tf.variable_scope('Attention'):
            alpha_t = self._attention_model(a, h_t_1) # alpha.shape = (B, L)

        ################ Soft deterministic attention: z = alpha-weighted mean of a ################
        ## (B, L) batch_dot (B,L,D) -> (B, D)
        with tf.variable_scope('Phi'):
            z_t = K.batch_dot(alpha_t, a, axes=[1,1]) # z_t.shape = (B, D)

        ################ Decoder Layer ################
        with tf.variable_scope("Decoder_LSTM"):
            (h_t, lstm_states_t) = self._build_decoder_lstm(Ex_t, z_t, lstm_states_t_1) # h_t.shape=(B,n)

        ################ Decoder Layer ################
        with tf.variable_scope('Output_Layer'):
            yLogits_t = self._build_output_layer(Ex_t, h_t, z_t) # yProbs_t.shape = (B,K)

        assert K.int_shape(h_t) == (B, n)
        assert K.int_shape(lstm_states_t.h) == (B, n)
        assert K.int_shape(lstm_states_t.c) == (B, n)
        #assert K.int_shape(yProbs_t) == (B, Kv)
        assert K.int_shape(yLogits_t) == (B, Kv)
        assert K.int_shape(alpha_t) == (B, L)

        return yLogits_t, (tuple(lstm_states_t), alpha_t)
