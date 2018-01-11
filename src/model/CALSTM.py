#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
    Conditioned Attentive LSTM

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
import tensorflow as tf
# from keras import backend as K
from tensorflow.contrib.keras import backend as K
import tf_commons as tfc
from hyper_params import CALSTMParams


CALSTMState = collections.namedtuple("CALSTMState", ('lstm_state', 'alpha', 'ztop', 'beta'))

class CALSTM(tf.nn.rnn_cell.RNNCell):
    """
    One timestep of the ConditionedAttentiveLSTM. The entire function is a complex RNN-cell
    that includes one LSTM-Stack conditioned by image features and an attention model.
    """

    def __init__(self, config, context, beamsearch_width=1, var_scope=None):
        assert K.int_shape(context) == (config.B, config.L, config.D)

        with tf.variable_scope(var_scope or 'CALSTM') as scope:
            with tf.name_scope(scope.original_name_scope):
                super(CALSTM, self).__init__(_scope=scope, name=scope.name)
                self.my_scope = scope
                self.C = CALSTMParams(config)
                ## Beam Width to be supplied to BeamsearchDecoder. It essentially broadcasts/tiles a
                ## batch of input from size B to B * BeamWidth. Set this value to 1 in the training
                ## phase.
                self._beamsearch_width = beamsearch_width

                self._a = context ## Image features from the Conv-Net
                assert self._a.get_shape().as_list() == [self.C.B, self.C.L, self.C.D]

                self._LSTM_stack = tfc.RNNWrapper(self.C.decoder_lstm,
                                                  beamsearch_width=beamsearch_width)

    @property
    def output_size(self):
        # size of h_t
        return self._LSTM_stack.output_size

    @property
    def batch_output_shape(self):
        ## batch-shape of h_t taking beamwidth into account
        return (self.ActualBatchSize, self._LSTM_stack.output_size)
    
    @property
    def state_size(self):
        L = self.C.L # sizeof alpha
        D = self.C.D # size of z
        sizeof_beta = 1 # beta is a scalar

        # must match CALSTMState
        return CALSTMState(self._LSTM_stack.state_size, L, D, sizeof_beta)

    def assertOutputShape(self, output):
        """
        Asserts that the shape of the tensor is consistent with the stack's output shape.
        For e.g. the output shape is o then the input-tensor should be of shape (B,o)
        """
        # assert (self.ActualBatchSize, self._LSTM_stack.output_size) == K.int_shape(output)
        assert K.int_shape(output) == self.batch_output_shape

    def zero_state(self, batch_size, dtype):
        with tf.variable_scope(self.my_scope) as var_scope:
            with tf.name_scope(var_scope.original_name_scope):
                with tf.variable_scope("ZeroState", values=[batch_size]):
                    return CALSTMState(
                        self._LSTM_stack.zero_state(batch_size, dtype),
                        tf.zeros((batch_size, self.C.L), dtype=dtype, name='alpha'),
                        tf.zeros((batch_size, self.C.D), dtype=dtype, name='ztop'),
                        tf.zeros((batch_size, 1), dtype=dtype, name='beta')
                        )

    @property
    def BeamWidth(self):
        return self._beamsearch_width

    @property
    def ActualBatchSize(self):
        return self.C.B*self.BeamWidth

#    def _set_beamwidth(self, beamwidth):
#        self._beamsearch_width = beamwidth
#        self._LSTM_stack._set_beamwidth(beamwidth)

    def _attention_model(self, a, h_prev, Ex_t):
        with tf.variable_scope(self.my_scope) as var_scope:
            with tf.name_scope(var_scope.original_name_scope):
                with tf.variable_scope('AttentionModel'):
                    CONF = self.C
                    B = self.ActualBatchSize
                    L = CONF.L
                    D = CONF.D
                    h = h_prev
                    n = self.output_size
                    m = CONF.m

                    self.assertOutputShape(h_prev)
                    assert K.int_shape(a) == (B, L, D)
                    assert K.int_shape(h_prev) == (B, n)
                    assert K.int_shape(Ex_t) == (B, m)

                    if (CONF.att_model == 'MLP_shared') or (CONF.att_model == '1x1_conv'):
                        """
                        Here we'll effectively create L MLP stacks all sharing the same weights. Each
                        stack receives a concatenated vector of a(l) and h as input.
                        """
                        # h.shape = (B,n). Expand it to (B,1,n) and then broadcast to (B,L,n) in order
                        # to concatenate with feature vectors of 'a' whose shape=(B,L,D)
                        h = tf.identity(K.tile(K.expand_dims(h, axis=1), (1, L, 1)), name='h_t-1')
                        a = tf.identity(a, name='a')
                        if CONF.feed_clock_to_att:
                            assert CONF.build_scanning_RNN, 'Attention model can take Ex_t only in a scanning-LSTM'
                            # Ex_t.shape = (B,m). Expand it to (B,1,m) and then broadcast to (B,L,m) in order
                            # to concatenate with feature vectors of 'a' whose shape=(B,L,D)
                            x = tf.identity(K.tile(K.expand_dims(Ex_t, axis=1), (1, L, 1)), name='Ex_t')
                            # Concatenate a, h nd x. Final shape = (B, L, D+n+m)
                            att_inp = tf.concat([a, h, x], -1, name='ai_h_x') # (B, L, D+n+m)
                            assert K.int_shape(att_inp) == (B, L, D+n+m)
                        else:
                            # Concatenate a and h. Final shape = (B, L, D+n)
                            att_inp = tf.concat([a, h], -1, name='ai_h') # (B, L, D+n)
                            assert K.int_shape(att_inp) == (B, L, D+n)

                        if CONF.att_model == 'MLP_shared':
                            ## For #layers > 1 this implementation will endup being different than the paper's implementation because they only 
                            ## Below is how it is implemented in the code released by the authors of the paper
                            ##     for i in range(1, CONF.att_a_layers+1):
                            ##         if not last_layer:
                            ##              a = Dense(CONF['att_a_%d_n'%(i,)], activation=tanh)(a)
                            ##         else: # last-layer
                            ##              a = AffineTransform(CONF['att_a_%d_n'%(i,)])(a)
                            ##     h = AffineTransform(CONF['att_h_%d_n'%(i,)])(h)
                            ##     ah = a + K.expand_dims(h, axis=1)
                            ##     ah = tanh(ah)
                            ##     alpha = Dense(softmax_layer_params, activation=softmax)(ah)

                            alpha_1_ = tfc.MLPStack(CONF.att_layers)(att_inp)  # (B, L, 1)
                            assert K.int_shape(alpha_1_) == (B, L, 1)  # (B, L, 1)
                            alpha_ = K.squeeze(alpha_1_, axis=2)  # output shape = (B, L)
                            assert K.int_shape(alpha_) == (B, L)

                        elif CONF.att_model == '1x1_conv':
                            """
                            NOTE: The above model ('MLP_shared') tantamounts to a 
                            1x1 convolution on the Lx1 shaped (L=H.W) convnet features with num_channels=D i.e. an input shape of (H,W,C) or (1,L,D).
                            Using 'dimctx' kernels of size (1,1) and stride=1 resulting in an output shape of (1,L,dimctx) [or (B, L, 1, dimctx) with the batch dimension included].
                            This option provides such a convnet layer implementation (which turns out not to be faster than MLP_shared).
                            """
                            att_inp = tf.expand_dims(att_inp, axis=1)
                            alpha_1_ = tfc.ConvStack(CONF.att_layers, (B, 1, L, D+self.output_size))(att_inp)
                            assert K.int_shape(alpha_1_) == (B,1,L,1)
                            alpha_ = tf.squeeze(alpha_1_, axis=[1,3]) # (B, L)
                            assert K.int_shape(alpha_) == (B,L)
                    
                    elif CONF.att_model == 'MLP_full':  # MLP: weights not shared across L
                        ## concatenate a and h_prev and pass them through a MLP. This is different than the theano
                        ## implementation of the paper because we flatten a from (B,L,D) to (B,L*D). Hence each element
                        ## of the L*D vector receives its own weight because the effective weight matrix here would be
                        ## shape (L*D, num_dense_units) as compared to (D, num_dense_units) as in the shared_weights case

                        ## Concatenate a and h. Final shape will be (B, L*D+n)
                        with tf.variable_scope('a_h'):
                            a_ = K.batch_flatten(a) # (B, L*D)
                            a_.set_shape((B, L*D)) # Flatten loses shape info
                            if CONF.build_scanning_RNN and CONF.feed_clock_to_att:
                                assert CONF.build_scanning_RNN, 'Attention model can take Ex_t only in a scanning-LSTM'
                                att_inp = tf.concat([a_, h, Ex_t], -1, name="a_h_x") # (B, L*D + n + m)
                                assert K.int_shape(att_inp) == (B, L*D + self.output_size + m), 'shape %s != %s'%(K.int_shape(att_inp),(B, L*D + self.output_size + m))
                            else:
                                att_inp = tf.concat([a_, h], -1, name="a_h") # (B, L*D + n)
                                assert K.int_shape(att_inp) == (B, L*D + self.output_size), 'shape %s != %s'%(K.int_shape(att_inp),(B, L*D + self.output_size))
                        alpha_ = tfc.MLPStack(CONF.att_layers)(att_inp)  # (B, L)
                        assert K.int_shape(alpha_) == (B, L)

                    else:
                        raise AttributeError('Invalid value of att_model param: %s'%CONF.att_model)

                    ## Softmax
                    alpha = tf.identity(tf.nn.softmax(alpha_), name='alpha')
                    assert K.int_shape(alpha) == (B, L)

                    ## Attention Modulator: Beta
                    if CONF.build_att_modulator:
                        beta = tfc.MLPStack(CONF.att_modulator, self.batch_output_shape)(h_prev)
                        beta = tf.identity(beta, name='beta')
                    else:
                        beta = tf.constant(1., shape=(B, 1), dtype=CONF.dtype)
                    assert K.int_shape(beta) == (B, 1)

                    return alpha, beta

    def _decoder_lstm(self, Ex_t, z_t, lstm_states_t_1):
        """Represents invocation of the decoder lstm. (h_t, lstm_states_t) = *(z_t|Ex_t, lstm_states_t_1)"""
        with tf.variable_scope(self.my_scope) as var_scope:
            with tf.name_scope(var_scope.original_name_scope):
                m = self.C.m
                D = self.C.D
                B = self.C.B*self.BeamWidth

                if (not self.C.build_scanning_RNN) or (not self.C.no_clock_to_lstm):
                    inputs_t = tf.concat((Ex_t, z_t), axis=-1, name="Ex_concat_z")
                    assert K.int_shape(inputs_t) == (B, m+D)
                else:
                    assert self.C.build_scanning_RNN, 'no_clock_to_lstm can be set only in a scanning RNN '
                    inputs_t = z_t
                    assert K.int_shape(inputs_t) == (B, D)

                self._LSTM_stack.assertStateShape(lstm_states_t_1)

                (htop_t, lstm_states_t) = self._LSTM_stack(inputs_t, lstm_states_t_1)
                return (htop_t, lstm_states_t)

    def call(self, inputs, state):
        """
        Builds/threads tf graph for one RNN iteration.
        Takes in previous lstm states (h and c),
        the current input and the image annotations (a) as input and outputs the states and outputs for the
        current timestep.
        Note that input(t) = Ey(t-1). Input(t=0) = Null. When training, the target output is used for Ey
        whereas at prediction time (via. beam-search for e.g.) the actual output is used.
        """
        with tf.variable_scope(self.my_scope) as var_scope:
            with tf.name_scope(var_scope.original_name_scope):## Ugly but only way to fix TB visuals

                ## Input
                Ex_t = inputs                          # shape = (B,m)
                ## State
                htop_1 = state.lstm_state.h if self._LSTM_stack.num_layers == 1 else state.lstm_state[-1].h
                lstm_states_t_1 = state.lstm_state   # shape = ((B,n), (B,n)) = (c_t_1, h_t_1)
                unused_alpha_t_1 = state.alpha       # shape = (B, L)
                unused_beta_t_1 = state.beta         # (B,1)
                a = self._a
                ## Broadcast context from size B to B*BeamWidth, because that's what BeamSearchDecoder does
                ## to the input batch.
                if self.BeamWidth > 1:
                    a = tf.contrib.seq2seq.tile_batch(self._a, self.BeamWidth)

                CONF = self.C
                B = CONF.B*self.BeamWidth
                m = CONF.m
                L = CONF.L
        #        Kv =CONF.K

                assert K.int_shape(Ex_t) == (B, m)
                assert K.int_shape(unused_alpha_t_1) == (B, L)
                assert K.int_shape(unused_beta_t_1) == (B, 1)
                self._LSTM_stack.assertStateShape(lstm_states_t_1)

                ################ Attention Model ################
                alpha_t, beta_t = self._attention_model(a, htop_1, Ex_t)  # alpha.shape = (B, L), beta.shape = (B,)
                assert K.int_shape(alpha_t) == (B,L)
                assert K.int_shape(beta_t) == (B, 1)

                ################ Soft deterministic attention: z = alpha-weighted mean of a ################
                with tf.variable_scope('Phi'):
                    ## (B, L) batch_dot (B,L,D) -> (B, D)
                    z_t = K.batch_dot(alpha_t, a, axes=[1,1]) # z_t.shape = (B, D)
                    z_t = tf.multiply(beta_t, z_t, name='beta_t.z') # elementwise multiply (B,1)*(B,D) -> (B,D)
                    z_t = tf.identity(z_t, name='z_t') ## For tensorboard viz.

                ################ Decoder Layer ################
                (htop_t, lstm_states_t) = self._decoder_lstm(Ex_t, z_t, lstm_states_t_1)  # h_t.shape=(B,n)

        #        ################ Output Layer ################
        #        with tf.variable_scope('Output_Layer'):
        #            yLogits_t = self._output_layer(Ex_t, h_t, z_t) # yProbs_t.shape = (B,K)

                self._LSTM_stack.assertOutputShape(htop_t)
                self._LSTM_stack.assertStateShape(lstm_states_t)
                ## assert K.int_shape(yProbs_t) == (B, Kv)
                ## assert K.int_shape(yLogits_t) == (B, Kv)
                assert K.int_shape(alpha_t) == (B, L)
                assert K.int_shape(beta_t) == (B,1)

                return htop_t, CALSTMState(lstm_states_t, alpha_t, z_t, beta_t)
