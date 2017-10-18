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
import itertools
import dl_commons as dlc
import tf_commons as tfc
import tensorflow as tf
from keras import backend as K
import collections
from hyper_params import CALSTMParams


CALSTMState = collections.namedtuple("CALSTMState", ('lstm_state', 'alpha', 'ztop'))

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
        # h_t
        return self._LSTM_stack.output_size

    @property
    def state_size(self):
        L = self.C.L # sizeof alpha
        D = self.C.D # size of z

        # must match CALSTMState
#        return CALSTMState(self._LSTM_stack.output_size, self._LSTM_stack.state_size, L, D)
        return CALSTMState(self._LSTM_stack.state_size, L, D)

    def assertOutputShape(self, output):
        """
        Asserts that the shape of the tensor is consistent with the stack's output shape.
        For e.g. the output shape is o then the input-tensor should be of shape (B,o)
        """
        assert (self.ActualBatchSize, self._LSTM_stack.output_size) == K.int_shape(output)

    def zero_state(self, batch_size, dtype):
        with tf.variable_scope(self.my_scope) as var_scope:
            with tf.name_scope(var_scope.original_name_scope):
                with tf.variable_scope("ZeroState", values=[batch_size]):
                    return CALSTMState(
                        self._LSTM_stack.zero_state(batch_size, dtype),
                        tf.zeros((batch_size, self.C.L), dtype=dtype, name='alpha'),
                        tf.zeros((batch_size, self.C.D), dtype=dtype, name='ztop')
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

    def _attention_model(self, a, h_prev):
        with tf.variable_scope(self.my_scope) as var_scope:
            with tf.name_scope(var_scope.original_name_scope):
                with tf.variable_scope('AttentionModel'):
                    CONF = self.C
                    B = CONF.B*self.BeamWidth
                    L = CONF.L
                    D = CONF.D
                    h = h_prev
                    n = self._LSTM_stack.output_size

                    self._LSTM_stack.assertOutputShape(h_prev)
                    assert K.int_shape(a) == (B, L, D)

                    ## For #layers > 1 this will endup being different than the paper's implementation
                    if (CONF.att_share_weights == 'paper') or (CONF.att_share_weights == 'convnet'):
                        """
                        Here we'll effectively create L MLP stacks all sharing the same weights. Each
                        stack receives a concatenated vector of a(l) and h as input.

                        TODO: This tantamounts to a 
                        1x1 convolution on the Lx1 shaped (L=H.W) convnet output with num_channels=D i.e. an input shape of (H,W,C) = (1,L,D).
                        Using 'dim' kernels of size (1,1) and stride=1 resulting in an output shape of (1,L,dim) [or (B, L, 1, dim) with the batch dimension included].
                        Using a convnet layer of this type may actually be more efficient (and easier to code).
                        """
                        ## h.shape = (B,n). Convert it to (B,1,n) and then broadcast to (B,L,n) in order
                        ## to concatenate with feature vectors of 'a' whose shape=(B,L,D)
                        h = tf.identity(K.tile(K.expand_dims(h, axis=1), (1,L,1)), name='h_t-1')
                        a = tf.identity(a, name='a')
                        ## Concatenate a and h. Final shape = (B, L, D+n)
                        ah = tf.concat([a,h], -1, name='a_concat_h'); # (B, L, D+n)
                        if CONF.att_share_weights == 'paper':
                            ah = tfc.MLPStack(CONF.att_layers)(ah) # (B, L, dim)
                            dim = K.int_shape(ah)[-1]
                            ## Below is roughly how it is implemented in the code released by the authors of the paper
                            ##     for i in range(1, CONF.att_a_layers+1):
                            ##         a = Dense(CONF['att_a_%d_n'%(i,)], activation=CONF.att_actv)(a)
                            ##     for i in range(1, CONF.att_h_layers+1):
                            ##         h = Dense(CONF['att_h_%d_n'%(i,)], activation=CONF.att_actv)(h)
                            ##    ah = a + K.expand_dims(h, axis=1)

                            ## Gather all activations across the features; go from (B, L, dim) to (B,L,1).
                            ## Instead, one could've just ensured that the num_units of the final MLP layer was == 1
                            ## resulting in an output dimension of (B,L,1). But the paper does it in this way - i.e.
                            ## adds a FC layer without an activation (somewhat like an embedding) so we'll keeep that as an option.
                            if CONF.att_weighted_gather:
                                with tf.variable_scope('weighted_gather'):
                                    # Gather layer may have dropout based on CONF.att_layers
                                    gather_params = tfc.FCLayerParams(CONF.att_layers).updated({'activation_fn':None, 'num_units':1})
                                    ah = tfc.FCLayer(gather_params)(ah) # output shape = (B, L, 1)
                            else:
                                # ah = K.mean(ah, axis=2, name='mean_gather') # output shape = (B, L)
                                assert dim == 1 ## (B, L, 1)

                            ah = K.squeeze(ah, axis=2) # output shape = (B, L)

                        elif CONF.att_share_weights == 'convnet':
                            ah = tfc.ConvStack(CONF.att_layers, (B,1,L,D+n))(tf.expand_dims(ah, axis=1))
                            assert K.int_shape(ah) == (B,1,L,1)
                            ah = tf.squeeze(ah, axis=[1,3]) # (B, L)
                    
                    elif CONF.att_share_weights == 'MLP': # MLP: weights not shared across L
                        ## concatenate a and h_prev and pass them through a MLP. This is different than the theano
                        ## implementation of the paper because we flatten a from (B,L,D) to (B,L*D). Hence each element
                        ## of the L*D vector receives its own weight because the effective weight matrix here would be
                        ## shape (L*D, num_dense_units) as compared to (D, num_dense_units) as in the shared_weights case

                        ## Concatenate a and h. Final shape will be (B, L*D+n)
                        with tf.variable_scope('a_flatten_concat_h'):
                            a_ = K.batch_flatten(a) # (B, L*D)
                            a_.set_shape((B, L*D)) # Flatten loses shape info
                            ah = K.concatenate([a_, h]) # (B, L*D+n)
                            assert K.int_shape(ah) == (B, L*D + self.output_size), 'shape %s != %s'%(K.int_shape(ah),(B, L*D + self.output_size))
                        ah = tfc.MLPStack(CONF.att_layers)(ah) # (B, L)
                        dim = CONF.att_layers.layers_units[-1]
                        assert dim == L
                        assert K.int_shape(ah) == (B, L)
                    else:
                        raise AttributeError('Invalid value of att_share_weights param: %s'%CONF.att_share_weights)

                    alpha = tf.nn.softmax(ah, name='alpha') # output shape = (B, L)
                    # alpha = tf.identity(alpha, name='alpha') ## For clearer visualization

                    assert K.int_shape(alpha) == (B, L)
                    return alpha

    def _decoder_lstm(self, Ex_t, z_t, lstm_states_t_1):
        """Represents invocation of the decoder lstm. (h_t, lstm_states_t) = *(z_t|Ex_t, lstm_states_t_1)"""
        with tf.variable_scope(self.my_scope) as var_scope:
            with tf.name_scope(var_scope.original_name_scope):
                m = self.C.m
                D = self.C.D
                B = self.C.B*self.BeamWidth

                inputs_t = tf.concat((Ex_t, z_t), axis=-1, name="Ex_concat_z")
                assert K.int_shape(inputs_t) == (B, m+D)
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
                alpha_t_1 = state.alpha            # shape = (B, L)
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
                assert K.int_shape(alpha_t_1) == (B, L)
                self._LSTM_stack.assertStateShape(lstm_states_t_1)

                ################ Attention Model ################
                alpha_t = self._attention_model(a, htop_1) # alpha.shape = (B, L)

                ################ Soft deterministic attention: z = alpha-weighted mean of a ################
                ## (B, L) batch_dot (B,L,D) -> (B, D)
                with tf.variable_scope('Phi'):
                    z_t = K.batch_dot(alpha_t, a, axes=[1,1]) # z_t.shape = (B, D)
                    z_t = tf.identity(z_t, name='z_t') ## For tensorboard viz.

                ################ Decoder Layer ################
                (htop_t, lstm_states_t) = self._decoder_lstm(Ex_t, z_t, lstm_states_t_1) # h_t.shape=(B,n)

        #        ################ Output Layer ################
        #        with tf.variable_scope('Output_Layer'):
        #            yLogits_t = self._output_layer(Ex_t, h_t, z_t) # yProbs_t.shape = (B,K)

                self._LSTM_stack.assertOutputShape(htop_t)
                self._LSTM_stack.assertStateShape(lstm_states_t)
                ## assert K.int_shape(yProbs_t) == (B, Kv)
                ## assert K.int_shape(yLogits_t) == (B, Kv)
                assert K.int_shape(alpha_t) == (B, L)

                return htop_t, CALSTMState(lstm_states_t, alpha_t, z_t)
