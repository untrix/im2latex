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
import tensorflow as tf
from tensorflow.contrib.keras import backend as K
from tensorflow.contrib.keras import applications as K_apps
from tensorflow.contrib.framework import nest as tf_nest
import dl_commons as dlc
import tf_commons as tfc
from data_reader import InpTup
from CALSTM import CALSTM, CALSTMState
from hyper_params import Im2LatexModelParams
from tf_tutorial_code import average_gradients
from tf_dynamic_decode import dynamic_decode

def build_vgg_context(params, image_batch):
    ## Conv-net
    ## params.logger.debug('image_batch shape = %s', K.int_shape(image_batch))
    assert K.int_shape(image_batch) == (params.B,) + params.image_shape
    ################ Build VGG Net ################
    with tf.variable_scope('VGGNet'):
        K.set_image_data_format('channels_last')
        convnet = K_apps.vgg16.VGG16(include_top=False, weights='imagenet', pooling=None,
                        input_shape=params.image_shape,
                        input_tensor = image_batch)
        convnet.trainable = False
        for layer in convnet.layers:
            layer.trainable = False

        # print 'convnet output_shape = ', convnet.output_shape
        ## a = convnet(image_batch)
        a = convnet.output
        assert K.int_shape(a)[1:] == (params.H, params.W, params.D)

        ## Combine HxW into a single dimension L
        a = tf.reshape(a, shape=(params.B or -1, params.L, params.D))
        assert K.int_shape(a) == (params.B, params.L, params.D)

    return a

def build_convnet(params, image_batch, reuse_vars=True):
    ## Conv-net
    ## params.logger.debug('image_batch shape = %s', K.int_shape(image_batch))
    assert K.int_shape(image_batch) == (params.B,) + params.image_shape
    ################ Build Conv Net ################
    with tf.variable_scope('Convnet', reuse=reuse_vars):
        convnet = tfc.ConvStack(params.CONVNET, (params.B,)+params.image_shape)
        a = convnet(image_batch)
        assert K.int_shape(a)[1:] == (params.H0, params.W0, params.D0), 'Expected shape %s, got %s'%((params.H0, params.W0, params.D0), K.int_shape(a)[1:])

        ## Combine HxW into a single dimension L
        a = tf.reshape(a, shape=(params.B or -1, params.L0, params.D0))
        assert K.int_shape(a) == (params.B, params.L0, params.D0)

    return a

Im2LatexState = collections.namedtuple('Im2LatexState', ('calstm_states', 'yProbs'))
class Im2LatexModel(tf.nn.rnn_cell.RNNCell):
    """
    One timestep of the decoder model. The entire function can be seen as a complex RNN-cell
    that includes a LSTM stack and an attention model.
    """
    def __init__(self, params, inp_q, opt=None, seq2seq_beam_width=1, reuse=True):
        """
        Args:
            params (Im2LatexModelParams)
            opt: optimizer instance - e.g. tf.train.AdamOptimizer(...). A value of None will make the model switch
                to evaluation mode. Otherwise it will build a training mode graph.
            seq2seq_beam_width (integer): Only used when inferencing with beamsearch. Otherwise set it to 1.
                Will cause the batch_size in internal assert statements to get multiplied by beamwidth.
            reuse: Sets the variable_reuse setting for the entire object.
        """
        self._params = self.C = Im2LatexModelParams(params)
        self._opt = opt
        with tf.variable_scope('I2L', reuse=reuse) as outer_scope:
            self.outer_scope = outer_scope
            with tf.variable_scope('Inputs') as scope:
                self._inp_q = inp_q
                inp_tup = InpTup(*self._inp_q.dequeue())
                self._y_s = tf.identity(inp_tup.y_s, name='y_s')
                self._seq_len = tf.identity(inp_tup.seq_len, name='seq_len')
                self._y_ctc = tf.identity(inp_tup.y_ctc, name='y_ctc')
                self._ctc_len = tf.identity(inp_tup.ctc_len, name='ctc_len')
                self._image_name = tf.identity(inp_tup.image_name, name="image_name")
                ## Set tensor shapes because they get forgotten in the queue
                self._y_s.set_shape((self.C.B, None))
                self._seq_len.set_shape((self.C.B,))
                self._y_ctc.set_shape((self.C.B, None))
                self._ctc_len.set_shape((self.C.B,))
                self._image_name.set_shape((self.C.B,))

                ## Image features/context
                if self._params.build_image_context == 0:
                    ## self._a = tf.placeholder(dtype=self.C.dtype, shape=(self.C.B, self.C.L, self.C.D), name='a')
                    self._a = tf.identity(inp_tup.im, name='a')
                    ## Set tensor shape because it gets forgotten in the queue
                    self._a.set_shape((self.C.B, self.C.L0, self.C.D0))
                else:
                    ## self._im = tf.placeholder(dtype=self.C.dtype, shape=((self.C.B,)+self.C.image_shape), name='image')
                    self._im = tf.identity(inp_tup.im, name='im')
                    ## Set tensor shape because it gets forgotten in the queue
                    self._im.set_shape((self.C.B,)+self.C.image_shape)

            # Build Convnet if needed
            if self._params.build_image_context != 0:
                ## Image features/context from the Conv-Net
                if self._params.build_image_context == 2:
                    self._a = build_convnet(params, self._im, reuse)
                elif self._params.build_image_context == 1:
                    self._a = build_vgg_context(params, self._im)
                else:
                    raise AttributeError('build_image_context should be in the range [0,2]. Instead, it is %s'%self._params.build_image_context)

            ## Regroup image features if needed
            if self.C.REGROUP_IMAGE is not None:
                with tf.variable_scope('FuseImageFeatures'):
                    self._a = tf.reshape(self._a, [self.C.B, self.C.H0, self.C.W0, self.C.D0])
                    self._a = tfc.group2D(self._a, self.C.REGROUP_IMAGE)
                    assert K.int_shape(self._a) == (self.C.B, self.C.H, self.C.W, self.C.D)
                    self._a = tf.reshape(self._a, [self.C.B, self.C.L, self.C.D], name='regrouped_im_feats')

            ## RNN portion of the model
            with tf.variable_scope('I2L_RNN') as scope:
                super(Im2LatexModel, self).__init__(_scope=scope, name=scope.name)
                self._rnn_scope = scope

                ## Beam Width to be supplied to BeamsearchDecoder. It essentially broadcasts/tiles a
                ## batch of input from size B to B * Seq2SeqBeamWidth. Set this value to 1 in the training
                ## phase.
                self._seq2seq_beam_width = seq2seq_beam_width

                ## First step of x_s is 1 - the begin-sequence token. Shape = (T, B); T==1
                self._x_0 = tf.ones(shape=(1, self.C.B*seq2seq_beam_width),
                                    dtype=self.C.int_type,
                                    name='begin_sequence')*self.C.StartTokenID

                self._calstms = []
                for i, rnn_params in enumerate(self.C.CALSTM_STACK, start=1):
                    with tf.variable_scope('CALSTM_%d'%i) as var_scope:
                        self._calstms.append(CALSTM(rnn_params, self._a, seq2seq_beam_width, var_scope))
                self._CALSTM_stack = tf.nn.rnn_cell.MultiRNNCell(self._calstms)
                self._num_calstm_layers = len(self.C.CALSTM_STACK)

                if not self.C.build_scanning_RNN:
                    # We need embedding only in the case of regular RNN (i.e. not scanning-RNN)
                    with tf.variable_scope('Embedding') as embedding_scope:
                        self._embedding_scope = embedding_scope
                        self._embedding_matrix = tf.get_variable('Embedding_Matrix',
                                                                 (self.C.K, self.C.m),
                                                                 initializer=self.C.embeddings_initializer,
                                                                 ## regularizer=self.C.embeddings_regularizer,
                                                                 trainable=True)
                        tfc.add_to_collection('REGULARIZED_WEIGHTS', self._embedding_matrix)
                        assert self.C.embeddings_regularizer is not None

            ## Init State Model
            self._init_state_model = self._init_state()

    @property
    def Seq2SeqBeamWidth(self):
        return self._seq2seq_beam_width

    @property
    def RuntimeBatchSize(self):
        return self.C.B * self.Seq2SeqBeamWidth

    @property
    def output_size(self):
        return self.C.K

    @property
    def state_size(self):
        return Im2LatexState(self._CALSTM_stack.state_size, self.C.K)

    def zero_state(self, batch_size, dtype):
        return Im2LatexState(self._CALSTM_stack.zero_state(batch_size, dtype),
                             tf.zeros((batch_size, self.C.K), dtype=dtype, name='yProbs'))

   # def _set_beamwidth(self, beamwidth):
   #     self._beamsearch_width = beamwidth
   #     for calstm in self._calstms:
   #         calstm._set_beamwidth(beamwidth)

    def _output_layer(self, Ex_t, h_t, z_t):
        with tf.variable_scope(self._rnn_scope) as var_scope:
            with tf.name_scope(var_scope.original_name_scope):
                with tf.variable_scope('Output_Layer'):
                    ## Renaming HyperParams for convenience
                    CONF = self.C
                    B = self.RuntimeBatchSize
                    D = self.C.D
                    m = self.C.m
                    Kv =self.C.K
                    n = self._CALSTM_stack.output_size

                    assert K.int_shape(Ex_t) == (B, m)
                    assert K.int_shape(h_t) == (B, self._CALSTM_stack.output_size)
                    assert K.int_shape(z_t) == (B, D)

                    ## First layer of output MLP
                    if CONF.output_reuse_embeddings: ## Follow the paper.
                        with tf.variable_scope('FirstLayer'):
                            affine_params = tfc.FCLayerParams(CONF.output_first_layer).updated({'activation_fn':None, 'dropout': None})
                            ## Affine transformation of h_t and z_t from size n/D to bring it down to m
                            h_z_t = tfc.FCLayer(affine_params)(tf.concat([h_t, z_t], -1)) # o_t: (B, m)
                            ## h_t and z_t are both dimension m now. So they can now be added to Ex_t.
                            o_t = h_z_t + Ex_t # Paper does not multiply this with weights presumably becasue its been already transformed by the embedding-weights.
                            ## non-linearity for the first layer
                            o_t = tfc.Activation(tfc.ActivationParams(CONF.output_first_layer))(o_t)
                            dim = CONF.output_first_layer.num_units
                            # ## DROPOUT: Paper has one dropout layer here
                            # if CONF.output_first_layer.dropout is not None:
                            #     o_t = tfc.DropoutLayer(CONF.output_first_layer.dropout, batch_input_shape=(B,m))(o_t)
                    else: ## Use a straight MLP Stack
                        if CONF.outputMLP_skip_connections:
                            o_t = K.concatenate((Ex_t, h_t, z_t)) # (B, m+n+D)
                            dim = m + n + D
                        else:
                            o_t = h_t
                            dim = n

                    ## Regular MLP layers
                    assert CONF.output_layers.layers[-1].num_units == Kv
                    logits_t = tfc.MLPStack(CONF.output_layers, batch_input_shape=(B,dim))(o_t)
                    ## DROPOUT: Paper has a dropout layer after each hidden FC layer (i.e. all excluding the softmax layer)

                    assert K.int_shape(logits_t) == (B, Kv)
                    yProbs = tf.identity(tf.nn.softmax(logits_t), name='yProbs')
                    return yProbs, logits_t

    def _recur_init_FCLayers(self, zero_state, counter, params, inp):
        """
        Creates FC layers for lstm_states (init_c and init_h) which will sit atop the init-state MLP.
        It does this by replacing each instance of 'h' or 'c' with a FC layer using the given params.
        This recursive function is only intended to be invoked by _init_state() and therefore is
        scoped under OutputLayers.
        """
        with tf.variable_scope(self._init_FC_scope) as var_scope:
            with tf.name_scope(var_scope.original_name_scope):
                if counter is None:
                    counter = itertools.count(1)

                assert dlc.issequence(zero_state)

                s = zero_state
                if hasattr(s, 'h'):
                    num_units = K.int_shape(s.h)[1]
                    layer_params = tfc.FCLayerParams(params).updated({'num_units':num_units})
                    s = s._replace(h = tfc.FCLayer(layer_params)(inp, counter.next()))
                if hasattr(s, 'c'):
                    num_units = K.int_shape(s.c)[1]
                    layer_params = tfc.FCLayerParams(params).updated({'num_units':num_units})
                    s = s._replace(c=tfc.FCLayer(layer_params)(inp, counter.next()))

                ## Create a mutable list from the immutable tuple
                lst = []
                for i in xrange(len(s)):
                    if dlc.issequence(s[i]):
                        lst.append(self._recur_init_FCLayers(s[i], counter, params, inp))
                    else:
                        lst.append(s[i])

                ## Create the tuple back from the list
                if hasattr(s, '_make'):
                    s = s._make(lst)
                else:
                    s = tuple(lst)

                ## Set htop to the topmost 'h' of the LSTM stack
        #        if hasattr(s, 'htop') and isinstance(s, CALSTMState):
        #            ## s.lstm_state can be a single LSTMStateTuple or a tuple of LSTMStateTuples
        #            s.htop = s.lstm_state.h if hasattr(s.lstm_state, 'h') else s.lstm_state[-1].h

                return s

    def _init_state(self):
        ################ Initializer MLP ################
        with tf.variable_scope(self.outer_scope):
            with tf.name_scope(self.outer_scope.original_name_scope):
                zero_state = self.zero_state(self.RuntimeBatchSize, dtype=self.C.dtype)
                if self.C.build_init_model:
                    with tf.variable_scope('Initializer_MLP'):

                        ## Broadcast im_context to Seq2SeqBeamWidth
                        if self.Seq2SeqBeamWidth > 1:
                            a = tf.contrib.seq2seq.tile_batch(self._a, self.Seq2SeqBeamWidth)
                        else:
                            a = self._a
                        ## As per the paper, this is a multi-headed MLP. It has a base stack of common layers, plus
                        ## one additional output layer for each of the h and c LSTM states. So if you had
                        ## configured - say 3 CALSTM-stacks with 2 LSTM cells per CALSTM-stack you would end-up with
                        ## 6 top-layers on top of the base MLP stack. Base MLP stack is specified in param 'init_model_hidden'

                        if self.C.init_model_input_transform == 'mean':  # take mean of the context-tensor across L
                            a = K.mean(a, axis=1)  # (self.RuntimeBatchSize, D)
                        elif self.C.init_model_input_transform == 'full':  # use the entire context tensor
                            a = tf.reshape(a, shape=(self.RuntimeBatchSize, self.C.L*self.C.D))
                        else:
                            raise ValueError('Unsupported value of param init_model_input: %s'%self.C.init_model_input)

                        # DROPOUT: The paper has a dropout layer after each hidden layer
                        if ('init_model_hidden' in self.C) and (self.C.init_model_hidden is not None):
                            a = tfc.MLPStack(self.C.init_model_hidden)(a)

                        counter = itertools.count(0)

                        def zero_to_init_state(zs, counter):
                            assert isinstance(zs, Im2LatexState)
                            cs = zs.calstm_states
                            assert isinstance(cs, tuple) and not isinstance(cs, CALSTMState)
                            lst = []
                            for i in xrange(len(cs)):
                                assert isinstance(cs[i], CALSTMState)
                                lst.append(self._recur_init_FCLayers(cs[i],
                                                                     counter,
                                                                     self.C.init_model_final_layers,
                                                                     a))

                            cs = tuple(lst)

                            return zs._replace(calstm_states=cs)

                        with tf.variable_scope('Top_Layers'):
                            self._init_FC_scope = tf.get_variable_scope()
                            # zero_state = self.zero_state(self.RuntimeBatchSize, dtype=self.C.dtype)
                            init_state = zero_to_init_state(zero_state, counter)
                else:
                    init_state = zero_state

        return init_state

    def _embedding_lookup(self, ids):
        assert not self.C.build_scanning_RNN  # embedding is turned off in case of scanning-RNN
        with tf.variable_scope(self._embedding_scope) as scope:
            with tf.name_scope(scope.original_name_scope):
                m = self.C.m
                assert self._embedding_matrix is not None
                #assert K.int_shape(ids) == (B,)
                shape = list(K.int_shape(ids))
                # with tf.device(None): ## Unset device placement to keep Tensorflow happy
                embedded = tf.nn.embedding_lookup(self._embedding_matrix, ids)
                shape.append(m)
                ## Embedding lookup forgets the leading dimensions (e.g. (B,))
                ## Fix that here.
                embedded.set_shape(shape) # (...,m)
                return embedded

    def call(self, Ex_t, state):
        """
        One step of the RNN API of this class.
        Layers a deep-output layer on top of CALSTM
        """
        with tf.variable_scope(self._rnn_scope) as var_scope:
            with tf.name_scope(var_scope.original_name_scope):## ugly, but only option to get pretty tensorboard visuals
                ## State
                calstm_states_t_1 = state.calstm_states
                ## CALSTM stack
                htop_t, calstm_states_t = self._CALSTM_stack(Ex_t, calstm_states_t_1)
                ## DROPOUT: Paper has one dropout layer in between CALSTM stack and output layer
                ## Output layer
                yProbs_t, yLogits_t = self._output_layer(Ex_t, htop_t, calstm_states_t[-1].ztop)

                return yLogits_t, Im2LatexState(calstm_states_t, yProbs_t)

    ScanOut = collections.namedtuple('ScanOut', ('yLogits', 'state'))
    def _scan_step_training(self, out_t_1, x_t):
        if not self.C.build_scanning_RNN:
            with tf.variable_scope('Ey'):
                Ex_t = self._embedding_lookup(x_t)
        else:
            Ex_t = x_t

        ## RNN.__call__
        ## yLogits_t, state_t = self(Ex_t, out_t_1[1], scope=self._rnn_scope)
        yLogits_t, state_t = self(Ex_t, out_t_1.state, scope=self._rnn_scope)
        return self.ScanOut(yLogits_t, state_t)

    def build_training_tower(self):
        """ Build the training graph of the model """
        B = self.C.B
        N = self._num_calstm_layers
        T = None
        # H = self.C.H
        # W = self.C.W
        # Kv =self.C.K
        # L = self.C.L

        with tf.variable_scope(self.outer_scope):
            with tf.name_scope(self.outer_scope.original_name_scope):## ugly, but only option to get pretty tensorboard visuals
                ## tf.scan requires time-dimension to be the first dimension
                # y_s = K.permute_dimensions(self._y_s, (1, 0), name='y_s')  # (T, B)
                y_s = tf.transpose(self._y_s, (1, 0), name='y_s')  # (T, B)
                assert K.int_shape(y_s) == (T,B)

                if not self.C.build_scanning_RNN:
                    ################ Build x_s for regular RNN ################
                    ## x_s is y_s time-delayed by 1 timestep. First token is 1 - the begin-sequence token.
                    ## last token of y_s which is <eos> token (zero) will not appear in x_s
                    x_s = K.concatenate((self._x_0, y_s[0:-1]), axis=0)
                    assert K.int_shape(x_s) == (T, B)
                else:
                    ################ Build x_s for the Scanning RNN ################
                    # The allotted time period S for each batch is a function of its bin-length (e.g. 2*bin_length).
                    # The model must learn to finish its job within this time period. The multiplier '2' above could
                    # be modeled as a hyper-parameter to be discovered using cross-validation.
                    #
                    # The signal x_s is comprised of two parts:
                    # 1) A linearly increasing signal - 'clock1' - from 1/S at t==1 to 1.0 at t=S. clock == t/S
                    # 2) A linearly increasing signal - 'clock2' == t/max-S where max-S is the largest possible value
                    #    of S given the data-set.
                    # 3) A scalar 's' between 0.0 and 1.0 indicating the value of S for that batch. = S/max-S

                    MaxS = int(self.C.MaxSeqLen*self.C.SFactor)
                    tf_MaxS = tf.convert_to_tensor(MaxS, dtype=self.C.int_type, name='MaxS')  # scalar

                    bin_len = tf.cast(tf.shape(y_s)[0], dtype=self.C.dtype)
                    tf_S = tf.floor(bin_len * tf.constant(self.C.SFactor, dtype=self.C.dtype, name='SFactor'))  # scalar
                    tf_S = tf.cast(tf_S, dtype=self.C.int_type, name='S')  # scalar
                    assert K.int_shape(tf_S) == ()

                    t = tf.expand_dims((tf.range(tf_S) + 1), axis=1)  # (S, 1)
                    t = tf.tile(t, [1, B])  # (S, B)
                    t = tf.cast(t, dtype=self.C.int_type, name='t')  # (S, B)
                    assert K.int_shape(t) == (None, B)

                    clock1 = tf.cast(tf.divide(t, tf_S, 'clock1'), dtype=self.C.dtype)  # (S, B)
                    assert K.int_shape(clock1) == (None, B)

                    clock2 = tf.cast(tf.divide(t, tf_MaxS, 'clock2'), dtype=self.C.dtype)  # (S, B)
                    assert K.int_shape(clock2) == (None, B)

                    s = tf.cast(tf.divide(tf_S, tf_MaxS, name='s_scalar'), dtype=self.C.dtype)  # scalar
                    s = tf.expand_dims(tf.expand_dims(s, axis=0), axis=0)  # (1, 1)
                    s = tf.tile(s, [tf_S, B], name='s_tensor')  # (S, B)
                    assert K.int_shape(s) == (None, B)

                    x_s = tf.stack([clock1, clock2, s], axis=2, name='x_s')  # (S, B, 3)
                    assert K.int_shape(x_s) == (None, B, 3)

                accum = self.ScanOut(tf.zeros(shape=(self.RuntimeBatchSize, self.C.K), dtype=self.C.dtype),  # The yLogits init-value is not used
                                     self._init_state_model)
                out_s = tf.scan(self._scan_step_training,
                                x_s,
                                initializer=accum,
                                swap_memory=self.C.swap_memory)
                # SCRATCHED: THIS IS ONLY ACCURATE FOR 1 CALSTM LAYER. GATHER ALPHAS OF LOWER CALSTM LAYERS.
                yLogits_s = out_s.yLogits
                alpha_s_n = tf.stack([cs.alpha for cs in out_s.state.calstm_states], axis=0) # (N, T, B, L)
                beta_s_n = tf.stack([cs.beta for cs in out_s.state.calstm_states], axis=0) # (N, T, B, 1)
                # Switch the batch dimension back to first position - (B, T, ...)
                yLogits = tf.transpose(yLogits_s, [1,0,2], name='yLogits')
                alpha = tf.transpose(alpha_s_n, [0,2,1,3], name='alpha')  # (N, B, T, L)
                beta = tf.transpose(beta_s_n, [0,2,1,3], name='beta')  # (N, B, T, 1)
                assert K.int_shape(beta) == (N, B, T, 1)
                beta_out = tf.squeeze(beta, axis=3, name='beta_out')  # (N, B, T, 1) -> (N, B, T)
                assert K.int_shape(beta_out) == (N, B, T)
                x_out = tf.transpose(x_s, [1, 0, 2] if self.C.build_scanning_RNN else [1, 0])
                assert K.int_shape(x_out)[:2] == (B, T)

                optimizer_ops = self._optimizer(yLogits,
                                            alpha,
                                            beta,
                                            tf.tile(tf.expand_dims(tf_S, axis=0), [B]) if self.C.build_scanning_RNN else self._seq_len,  # (B,)
                                            self._y_s,
                                            self._seq_len,
                                            self._y_ctc,
                                            self._ctc_len)

                return optimizer_ops.updated({
                                              'inp_q': self._inp_q,
                                              'image_name': self._image_name,
                                              'beta': beta_out,  # (N, B, T)
                                              'y_s': self._y_s,  # (B, T)
                                              'y_ctc': self._y_ctc,  # ((B,T),)
                                              'x_s': x_out  # (B, S, 3) or (B, T)
                                              })

    def _optimizer(self, yLogits, alpha, beta, x_len, y_s, y_len, y_ctc, ctc_len):
        with tf.variable_scope(self.outer_scope) as var_scope:
            with tf.name_scope(var_scope.original_name_scope):
                B = self.C.B
                Kv = self.C.K
                L = self.C.L
                N = self._num_calstm_layers
                H = self.C.H
                W = self.C.W
                T = None

                assert K.int_shape(yLogits) == (B, None, Kv)  # (B, T, K)
                assert K.int_shape(alpha) == (N, B, None, L)  # (N, B, T, L)
                assert K.int_shape(beta) == (N, B, None, 1)  # (N, B, T, 1)
                assert K.int_shape(y_s) == (B, None)  # (B, T)
                assert K.int_shape(x_len) == (B,)
                assert K.int_shape(y_len) == (B,)
                assert K.int_shape(y_ctc) == (B, None)  # (B, T)
                assert K.int_shape(ctc_len) == (B,)
                tf_T = tf.shape(yLogits)[1]
                bin_len = tf.shape(y_s)[1]

                ################ Loss Calculations ################
                with tf.variable_scope('Loss_Calculations') as loss_scope:

                    ################ Regularization Cost ################
                    with tf.variable_scope('Regularization_Cost'):
                        if (self.C.weights_regularizer is not None) and self.C.rLambda > 0:
                            reg_loss = self.C.rLambda * tf.reduce_sum(
                                [self.C.weights_regularizer(var) for var in tf.get_collection("REGULARIZED_WEIGHTS")])
                            _reg_wt_names = [var.name for var in tf.get_collection("REGULARIZED_WEIGHTS")]
                            assert len(dlc.get_dupes(_reg_wt_names)) == 0, \
                                'Some regularization weights seem to have been double-counted.%s' % dlc.get_dupes(
                                    _reg_wt_names)
                        else:
                            reg_loss = tf.constant(0, dtype=self.C.dtype)

                    x_mask = tf.sequence_mask(x_len,
                                              maxlen=tf_T,
                                              dtype=self.C.dtype,
                                              name='sequence_mask')  # (B, T)
                    assert K.int_shape(x_mask) == (B, None)  # (B,T)

                    ################ Build LogLoss ################
                    if not self.C.use_ctc_loss:
                        with tf.variable_scope('LogLoss'):
                            assert not self.C.build_scanning_RNN
                            sequence_mask = x_mask
                            ## Masked negative log-likelihood of the sequence.
                            ## Note that log(product(p_t)) = sum(log(p_t)) therefore taking log of
                            ## joint-sequence-probability is same as taking sum of log of probability at each time-step

                            ################ Compute Sequence Log-Loss / Log-Likelihood  ################
                            #####             == -Log( product(p_t) ) == -sum(Log(p_t))             #####
                            if self.C.sum_logloss:
                                ## Here we do not normalize the log-loss across time-steps because the
                                ## paper as well as it's source-code do not do that.
                                log_losses = tf.contrib.seq2seq.sequence_loss(logits=yLogits,
                                                                              targets=y_s,
                                                                              weights=sequence_mask,
                                                                              average_across_timesteps=False,
                                                                              average_across_batch=False,
                                                                              name='log_losses'
                                                                              )
                                # print 'shape of loss_vector = %s'%(K.int_shape(log_losses),)
                                log_losses = tf.reduce_sum(log_losses, axis=1,
                                                           name='total log-loss')  # sum along time dimension => (B,)
                                # print 'shape of loss_vector = %s'%(K.int_shape(log_losses),)
                                log_likelihood = tf.reduce_mean(log_losses, axis=0,
                                                                name='CrossEntropyPerSentence')  # scalar
                            else:  ## Standard log perplexity (average per-word log perplexity)
                                log_losses = tf.contrib.seq2seq.sequence_loss(logits=yLogits,
                                                                              targets=y_s,
                                                                              weights=sequence_mask,
                                                                              average_across_timesteps=True,
                                                                              average_across_batch=False)
                                # print 'shape of loss_vector = %s'%(K.int_shape(log_losses),)
                                log_likelihood = tf.reduce_mean(log_losses, axis=0, name='CrossEntropyPerWord')
                            assert K.int_shape(log_likelihood) == tuple()
                    else:
                        log_likelihood = tf.constant(0.0, name='no_log_likelihood')

                    ################ Build CTC Cost Function ################
                    ## Compute CTC loss/score with intermediate blanks removed. We've removed all spaces/blanks in the
                    ## target sequences (y_ctc). Hence the target (y_ctc_ sequences are shorter than the inputs (y_s/x_s).
                    ## In case of scanning-RNN, x_s is already much longer than y_s by a factor of self.C.SFactor. In
                    ## case of scanning-RNN this is the only log-loss calculated.
                    ## Using CTC loss will have the following side-effect:
                    ##  1) The network will be told that it is okay to omit blanks (spaces) or emit multiple blanks
                    ##     since CTC will ignore those. This makes the learning easier, but we'll need to insert blanks
                    ##     between tokens when printing out the predicted markup.
                    with tf.variable_scope('CTC_Cost'):
                        ## sparse tensor
                        #    y_idx =    tf.where(tf.not_equal(y_ctc, 0)) ## null-terminator/EOS is removed :((
                        ctc_mask = tf.sequence_mask(ctc_len, maxlen=tf.shape(y_ctc)[1], dtype=tf.bool)
                        assert K.int_shape(ctc_mask) == (B, None)  # (B,T)
                        y_idx = tf.where(ctc_mask)
                        y_vals = tf.gather_nd(y_ctc, y_idx)
                        y_sparse = tf.SparseTensor(y_idx, y_vals, tf.shape(y_ctc, out_type=tf.int64))
                        ctc_losses = tf.nn.ctc_loss(y_sparse,
                                                    yLogits,
                                                    x_len,
                                                    ctc_merge_repeated=(not self.C.no_ctc_merge_repeated),
                                                    time_major=False)
                        ## print 'shape of ctc_losses = %s'%(K.int_shape(ctc_losses),)
                        assert K.int_shape(ctc_losses) == (B,)
                        if self.C.sum_logloss:
                            ctc_loss = tf.reduce_mean(ctc_losses, axis=0, name='CTCSentenceLoss')  # scalar
                        else:  # mean loss per word
                            ctc_loss = tf.div(tf.reduce_sum(ctc_losses, axis=0),
                                              tf.reduce_sum(tf.cast(ctc_mask, dtype=self.C.dtype)),
                                              name='CTCWordLoss')  # scalar
                        assert K.int_shape(ctc_loss) == tuple()

                ################ CTC Beamsearch Decoding ################
                with tf.variable_scope('Accuracy_Calculation'):
                    if (self.C.CTCBlankTokenID is not None) or (self.C.SpaceTokenID is not None):
                        yLogits_s = K.permute_dimensions(yLogits, [1, 0, 2])  # (T, B, K)
                        ## ctc_beam_search_decoder sometimes produces ID values = -1
                        ## Note: returns sparse tensor therefore no EOS tokens are padded to the sequences.
                        decoded_ids_sparse, _ = tf.nn.ctc_beam_search_decoder(yLogits_s,
                                                                              x_len,
                                                                              beam_width=self.C.ctc_beam_width,
                                                                              top_paths=1,
                                                                              merge_repeated=(
                                                                              not self.C.no_ctc_merge_repeated))
                        # print 'shape of ctc_decoded_ids = ', decoded_ids_sparse[0].get_shape().as_list()

                        ## Pad EOS_Tokens to the end of the sequences. (and convert to dense form).
                        ctc_decoded_ids = tf.sparse_tensor_to_dense(decoded_ids_sparse[0],
                                                                    default_value=self.C.NullTokenID)  # (B, T)
                        ctc_decoded_ids.set_shape((B, None))
                        ## get seq-lengths including eos_token (as is the case with tf.dynamic_decode as well as our input sequences)
                        ctc_decoded_lens = tfc.seqlens(ctc_decoded_ids)
                        #                    ctc_squashed_ids, ctc_squashed_lens = tfc.squash_2d(B,
                        #                                                                        ctc_decoded_ids,
                        #                                                                        sequence_lengths,
                        #                                                                        self.C.SpaceTokenID,
                        #                                                                        padding_token=0) # ((B,T), (B,))
                        #                    ctc_ed = tfc.edit_distance2D(B, ctc_squashed_ids, ctc_squashed_lens, tf.cast(self._y_ctc, dtype=tf.int64), self._ctc_len, self.C.CTCBlankTokenID, self.C.SpaceTokenID)
                        # ctc_ed = tfc.edit_distance2D_sparse(B, decoded_ids_sparse, tf.cast(self._y_ctc, dtype=tf.int64), self._ctc_len) #(B,)
                        ctc_ed = tfc.edit_distance2D(B,
                                                     ctc_decoded_ids, ctc_decoded_lens,
                                                     tf.cast(self._y_ctc, dtype=tf.int64), self._ctc_len,
                                                     self.C.CTCBlankTokenID, self.C.SpaceTokenID,
                                                     self.C.NullTokenID)
                        mean_ctc_ed = tf.reduce_mean(ctc_ed)  # scalar
                        num_hits = tf.reduce_sum(tf.to_float(tf.equal(ctc_ed, 0)))
                        #                    pred_len_ratio = tf.divide(ctc_squashed_lens,ctc_len)
                        pred_len_ratio = tf.divide(tf.cast(ctc_decoded_lens, self.C.dtype),
                                                   tf.cast(ctc_len, self.C.dtype))
                        # target_and_predicted_ids = tf.stack([tf.cast(self._y_ctc, dtype=tf.int64), ctc_squashed_ids], axis=1)
                    else:
                        assert not self.C.build_scanning_RNN, 'CTC Beamsearch Decoding must not be turned-off in a scanning-RNN'
                        ctc_ed = None
                        mean_ctc_ed = None
                        pred_len_ratio = None
                        num_hits = None
                        ctc_decoded_ids = None
                        ctc_decoded_lens = None

                with tf.variable_scope(loss_scope):
                    ################   Calculate the alpha penalty:   ################
                    #### lambda * sum_over_i&b(square(C/L - sum_over_t(alpha_i))) ####
                    with tf.variable_scope('AlphaPenalty'):
                        if self.C.build_scanning_RNN:
                            preds_mask = tf.sequence_mask(ctc_decoded_lens,
                                                          maxlen=tf_T,
                                                          dtype=self.C.dtype,
                                                          name='preds_mask')  # (B, T)
                            preds_len = ctc_decoded_lens
                        else:
                            preds_mask = x_mask
                            preds_len = x_len
                        alpha_mask = tf.expand_dims(preds_mask, axis=2)  # (B, T, 1)

                        if self.C.MeanSumAlphaEquals1:
                            mean_sum_alpha_over_t = 1.0
                        else:
                            mean_sum_alpha_over_t = tf.div(tf.cast(preds_len, dtype=self.C.dtype),
                                                           tf.cast(self.C.L, dtype=self.C.dtype))  # (B,)
                            mean_sum_alpha_over_t = tf.expand_dims(mean_sum_alpha_over_t, axis=1)  # (B, 1)

                        # sum_over_t = tf.reduce_sum(tf.multiply(alpha,preds_mask), axis=1, keep_dims=False)# (B, L)tf_S, tf_MaxS,
                        #    squared_diff = tf.squared_difference(sum_over_t, mean_sum_alpha_over_t) # (B, L)
                        #    alpha_penalty = self.C.pLambda * tf.reduce_sum(squared_diff, keep_dims=False) # scalar
                        sum_over_t = tf.reduce_sum(tf.multiply(alpha, alpha_mask), axis=2,
                                                   keep_dims=False)  # (N, B, L)
                        squared_diff = tf.squared_difference(sum_over_t, mean_sum_alpha_over_t)  # (N, B, L)
                        abs_diff = tf.abs(tf.subtract(sum_over_t, mean_sum_alpha_over_t))  # (N, B, L)

                        alpha_squared_error = tf.reduce_sum(squared_diff, axis=2, keep_dims=False)  # (N, B)
                        alpha_abs_error = tf.reduce_sum(abs_diff, axis=2, keep_dims=False)  # (N, B)
                        # alpha_squared_error = tf.reduce_sum(squared_diff, keep_dims=False) # scalar
                        # alpha_penalty = self.C.pLambda * alpha_squared_error # scalar

                        ## Max theoretical value of alpha_squared_error = C^2 * (L-1)/L. We'll use this to normalize alpha to a value between 0 and 1
                        ase_max = tf.constant((L - 1.0) / L * 1.0) * tf.square(
                            tf.cast(preds_len, dtype=self.C.dtype))  # (B,)
                        assert K.int_shape(ase_max) == (B,)
                        ase_max = tf.expand_dims(ase_max,
                                                 axis=0)  # (1, B) ~ C^2 who's average value is 5000 for our dataset

                        # Max theoretical value of alpha_abs_error = 2C * (L-1)/L
                        aae_max = tf.constant(2. * (L - 1.0) / (L * 1.0)) * tf.cast(preds_len,
                                                                                    dtype=self.C.dtype)  # (B,)
                        assert K.int_shape(aae_max) == (B,)
                        aae_max = tf.expand_dims(aae_max, axis=0)  # (1, B)

                        normalized_ase = alpha_squared_error * 100. / ase_max  # (N, B) all values lie between 0. and 100.
                        mean_norm_ase = tf.reduce_mean(normalized_ase)  # scalar between 0. and 100.0

                        normalized_aae = alpha_abs_error * 100. / aae_max  # (N, B) all values lie between 0. and 100.
                        mean_norm_aae = tf.reduce_mean(normalized_aae)  # scalar between 0. and 100.

                        if self.C.pLambda == 0:
                            alpha_penalty = tf.constant(0.0, name='no_alpha_penalty')
                        else:
                            alpha_penalty = tf.identity(
                                self.C.pLambda * tf.abs(mean_norm_ase - self.C.target_ase),
                                name='alpha_penalty')  # scalar

                        # mean_seq_len = tf.reduce_mean(tf.cast(preds_len, dtype=tf.float32))
                        # mean_sum_alpha_over_t = tf.reduce_mean(mean_sum_alpha_over_t)
                        # mean_sum_alpha_over_t2 = tf.reduce_mean(sum_over_t)
                        assert K.int_shape(alpha_penalty) == tuple()

                        # Reshape alpha for debugging output
                        alpha_out = tf.reshape(alpha, (N, B, -1, H, W))  # (N, B, T, L) -> (N, B, T, H, W)
                        alpha_out = tf.transpose(alpha_out, perm=(0, 1, 3, 4, 2),
                                                 name='alpha_out')  # (N, B, T, H, W)->(N, B, H, W, T)
                        assert K.int_shape(alpha_out) == (N, B, H, W, T)

                        # Beta metrics for logging
                        beta  # (N, B, T, 1)
                        alpha_mask  # (B, T, 1)
                        beta_out = tf.squeeze(tf.multiply(beta, alpha_mask), axis=3)  # (N, B, T)
                        seq_lens = tf.cast(tf.expand_dims(preds_len, axis=0),
                                           dtype=self.C.dtype)  # (N, B)
                        beta_mean = tf.divide(tf.reduce_sum(beta_out, axis=2, keep_dims=False), seq_lens,
                                              name='beta_mean')  # (N, B)
                        beta_mean_tiled = tf.tile(tf.expand_dims(beta_mean, axis=2),
                                                  [1, 1, tf_T])  # (N, B, T)
                        beta_mask = tf.expand_dims(preds_mask, axis=0)  # (1, B, T)
                        beta_mean_tiled = tf.multiply(beta_mean_tiled, beta_mask)  # (N, B, T)
                        beta_std_dev = tf.sqrt(
                            tf.reduce_sum(tf.squared_difference(beta_out, beta_mean_tiled), axis=2,
                                          keep_dims=False) / seq_lens, name='beta_std_dev')  # (N, B)

                    if self.C.use_ctc_loss:
                        cost = ctc_loss + alpha_penalty + reg_loss
                    else:
                        assert not self.C.build_scanning_RNN, 'use_ctc_loss must be True when building a scanning-RNN'
                        cost = log_likelihood + alpha_penalty + reg_loss

                ################ Optimizer ################
                if self._opt is not None:
                    with tf.variable_scope('Optimizer'):
                        grads = self._opt.compute_gradients(cost)
                else:
                    grads = tf.constant(0)

                return dlc.Properties({
                    'grads': grads,
                    'alpha': alpha_out,  # (N, B, H, W, T)
                    'beta_mean': beta_mean,  # (N, B)
                    'beta_std_dev': beta_std_dev,  # (N, B)
                    'log_likelihood': log_likelihood,  # scalar
                    'ctc_loss': ctc_loss,  # scalar
                    'loss': ctc_loss if self.C.use_ctc_loss else log_likelihood,
                    'alpha_penalty': alpha_penalty,  # scalar
                    'reg_loss': reg_loss,  # scalar
                    'cost': cost,  # scalar
                    'ctc_ed': ctc_ed,  # (B,)
                    'mean_ctc_ed': mean_ctc_ed,  # scalar
                    'y_len': y_len,  # (B,)
                    'ctc_len': ctc_len,  # (B,)
                    'scan_len': x_len if self.C.build_scanning_RNN else tf.constant(0.),  # (B,)
                    'bin_len': bin_len,  # scalar
                    'pred_len_ratio': pred_len_ratio,  # (B,)
                    'num_hits': num_hits,
                    'mean_norm_ase': mean_norm_ase,  # scalar between 0. and 100.0
                    'mean_norm_aae': mean_norm_aae,  # scalar between 0. and 100.0
                    'predicted_ids': ctc_decoded_ids,  # (B,T)
                    'predicted_lens': ctc_decoded_lens  # (B,)
                })

    def _beamsearch(self):
        """ Build the prediction graph of the model using beamsearch """
        assert not self.C.build_scanning_RNN
        with tf.variable_scope(self.outer_scope):
            # assert var_scope.reuse == True
            ## ugly, but only option to get proper tensorboard visuals
            with tf.name_scope(self.outer_scope.original_name_scope):
                with tf.variable_scope('BeamSearch'):
                    # class BeamSearchDecoder2(BeamSearchDecoder):
                    #     def initialize(self, name=None):
                    #         finished, start_inputs, initial_state = BeamSearchDecoder.initialize(self, name)
                    #         return tf.expand_dims(finished, axis=2), start_inputs, initial_state

                    decoder = tf.contrib.seq2seq.BeamSearchDecoder(self,
                                                                self._embedding_matrix,  # self._embedding_lookup,
                                                                tf.ones(shape=(self.C.B,), dtype=self.C.int_type) * self.C.StartTokenID,
                                                                self.C.NullTokenID,
                                                                self._init_state_model,
                                                                beam_width=self.Seq2SeqBeamWidth,
                                                                length_penalty_weight=self.C.beamsearch_length_penalty)
                    # self.C.logger.info('Decoder.output_size=%s', decoder.output_size)
                    # self.C.logger.info('Decoder.initial_state.shape=%s', tfc.nested_tf_shape(self._init_state_model))
                    final_outputs, final_state, final_sequence_lengths, final_states = dynamic_decode(
                    # final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                                                                    decoder,
                                                                    impute_finished=False, ## Setting this to true causes error
                                                                    maximum_iterations=self.C.MaxDecodeLen,
                                                                    swap_memory=self.C.swap_memory)
                    assert K.int_shape(final_outputs.predicted_ids) == (self.C.B, None, self.Seq2SeqBeamWidth)
                    assert K.int_shape(final_outputs.beam_search_decoder_output.scores) == (self.C.B, None, self.Seq2SeqBeamWidth) # (B, T, Seq2SeqBeamWidth)
                    assert K.int_shape(final_sequence_lengths) == (self.C.B, self.Seq2SeqBeamWidth)
                    tf_nest.assert_same_structure(final_states, final_state)

                    # self.C.logger.info('dynamic_decode final_states.shape=%s', tfc.nested_tf_shape(final_states))
                    # self.C.logger.info('dynamic_decode final_state.shape=%s', tfc.nested_tf_shape(final_state))
                    # self.C.logger.info('dynamic_decode final_outputs.shape=%s', tfc.nested_tf_shape(final_outputs))

                    return dlc.Properties({
                            'ids': final_outputs.predicted_ids, # (B, T, BW)
                            'scores': final_outputs.beam_search_decoder_output.scores, # (B, T, BW)
                            'seq_lens': final_sequence_lengths, # (B, BW),
                            'states': final_states.cell_state # Im2LatexState structure. Element shape = (B, T, BW, ...)
                            })

    def build_testing_tower(self):
        """ Test one batch of input """
        assert not self.C.build_scanning_RNN
        B = self.C.B
        BW = self.Seq2SeqBeamWidth
        k = min([self.C.k, BW])
        N = self._num_calstm_layers
        T = None
        H = self.C.H
        W = self.C.W

        with tf.variable_scope(self.outer_scope):
            # assert var_scope.reuse == True
            ## ugly, but only option to get proper tensorboard visuals
            with tf.name_scope(self.outer_scope.original_name_scope):
                outputs = self._beamsearch()
                T_shape = tf.shape(outputs.ids)[1]
                bm_ids = tf.transpose(outputs.ids, perm=(0,2,1)) # (B, Seq2SeqBeamWidth, T)
                assert bm_ids.get_shape().as_list() == [B, BW, T], 'bm_ids shape %s != %s'%(bm_ids.get_shape().as_list(), [B, BW, T])
                bm_seq_lens = outputs.seq_lens # (B, BW)
                assert K.int_shape(bm_seq_lens) == (B, BW)
                bm_scores = tf.transpose(outputs.scores, perm=[0,2,1]) # (B, Seq2SeqBeamWidth, T)
                assert bm_scores.get_shape().as_list() == [B, BW, T]

                ## Extract Alpha Values for Debugging
                ## states.shape=Im2LatexState(calstm_states=(CALSTMState(lstm_state=(LSTMStateTuple(c=[B, T, BW, n], h=[20, None, 10, 1000]), LSTMStateTuple(c=[20, None, 10, 1000], h=[20, None, 10, 1000])), alpha=[B, T, BW, L], ztop=[20, None, 10, 512]),), yProbs=[20, None, 10, 557])
                alphas = []
                betas  = []
                for i in range(N):
                    alpha = tf.transpose(outputs.states.calstm_states[i].alpha, (0,2,3,1)) ## (B, T, BW, L) -> (B, BW, L, T)
                    alpha = tf.reshape(alpha, shape=(B, BW, H, W, -1), name='alpha_out') # (B, BW, L, T) -> (B, BW, H, W, T)
                    alphas.append(alpha)
                    beta = tf.squeeze(outputs.states.calstm_states[i].beta, axis=3) # (B, T, BW, 1)
                    assert K.int_shape(beta) == (B, T, BW)
                    beta = tf.transpose(beta, (0,2,1), name='beta_out') ## (B, T, BW) -> (B, BW, T)
                    betas.append(beta)
                alphas = tf.stack(alphas, axis=0, name='alphas_out') # (N, B, BW, H, W, T)
                betas  = tf.stack(betas,  axis=0, name='betas_out') # (N, B, BW, 1, T)
                assert alphas.shape.as_list() == [N, B, BW, H, W, T], '%s != %s'%(alphas.shape.as_list(),[N, B, BW, H, W, T])
                assert K.int_shape(betas) == (N, B, BW, T), '%s != %s'%(K.int_shape(betas), (N, B, BW, T))

                with tf.name_scope('Score'):
                    ## Prepare a sequence mask for EOS padding
                    seq_lens_flat = tf.reshape(bm_seq_lens, shape=[-1]) # (B*Seq2SeqBeamWidth,)
                    assert K.int_shape(seq_lens_flat) == (B*BW,)
                    seq_mask_flat = tf.sequence_mask(seq_lens_flat, maxlen=T_shape, dtype=tf.int32) # (B*Seq2SeqBeamWidth, T)
                    assert K.int_shape(seq_mask_flat) == (B*BW, T)

                    # scores are log-probabilities which are negative values
                    # zero out scores (log probabilities) of words after EOS. Tantamounts to setting their probs = 1
                    # Hence they will have no effect on the sequence probabilities
                    scores_flat = tf.reshape(bm_scores, shape=(B*BW, -1))  * tf.to_float(seq_mask_flat) # (B*Seq2SeqBeamWidth, T)
                    scores = tf.reshape(scores_flat, shape=(B, BW, -1)) # (B, Seq2SeqBeamWidth, T)
                    assert K.int_shape(scores) == (B, BW, T)
                    ## Sum of log-probabilities == Product of probabilities
                    seq_scores = tf.reduce_sum(scores, axis=2) # (B, Seq2SeqBeamWidth)

                    ## Also zero-out tokens after EOS because we set impute_finished = False and as a result we noticed
                    ## ID values = -1 after EOS tokens.
                    ## Conveniently, zero is also the EOS token hence just multiplying the ids with 0 should make them EOS.
                    # ids_flat = tf.reshape(bm_ids, shape=(B*BW, -1)) * seq_mask_flat ## False values become 0
                    # ids = tf.reshape(ids_flat, shape=(B, BW, -1)) # (B, Seq2SeqBeamWidth, T)

                ## Select the top scoring beams
                with tf.name_scope('top_1'):
                    ## Top 1. I verified that beams in beamsearch_outputs are already sorted by seq_scores.
                    top1_seq_scores = seq_scores[:,0] # (B,)
                    top1_ids = bm_ids[:,0,:] # (B, T)training
                    assert top1_ids.get_shape().as_list() == [B, T]
                    top1_seq_lens = bm_seq_lens[:,0] # (B,)
                    top1_len_ratio = tf.divide(top1_seq_lens,self._seq_len)

                ## Top K
                with tf.name_scope('top_k'):
                    ## Old code not needed because beams in beamsearch_outputs are already sorted by seq_score
                        # topK_seq_scores, topK_score_indices = tfc.batch_top_k_2D(seq_scores, k) # (B, k) and (B, k, 2) sorted
                        # topK_ids = tfc.batch_slice(ids, topK_score_indices) # (B, k, T)
                        # assert K.int_shape(topK_ids) == (B, k, None)
                        # topK_seq_lens = tfc.batch_slice(seq_lens, topK_score_indices) # (B, k)
                        # assert K.int_shape(topK_seq_lens) == (B, k)
                    self.C.logger.info('seq_scores.shape=%s', tfc.nested_tf_shape(seq_scores))
                    topK_seq_scores = seq_scores[:,:k] # (B, k)
                    topK_ids = bm_ids[:,:k,:] # (B, k, T)
                    topK_seq_lens = bm_seq_lens[:, :k] # (B, k)
                    assert K.int_shape(topK_seq_scores) == (B, k)
                    assert K.int_shape(topK_seq_lens) == (B, k)
                    assert K.int_shape(topK_ids) == (B, k, T)


                ## BLEU scores
                # with tf.name_scope('BLEU'):
                #     ## BLEU score is calculated outside of TensorFlow and then injected back in via. a placeholder
                #     ph_bleu = tf.placeholder(tf.float32, shape=(self.C.B,), name="BLEU_placeholder")
                #     tf.summary.histogram( 'predicted/bleu', ph_bleu, collections=['bleu'])BW
                #     tf.summary.scalar( 'predicted/bleuH', tf.reduce_mean(ph_bleu), collections=['bleu'])
                #     logs_b = tf.summary.merge(tf.get_collection('bleu'))# (N, B)

                ## Levenshtein Distance metric
                with tf.name_scope('LevenshteinDistance'):
                    y_ctc_beams = tf.tile(tf.expand_dims(self._y_ctc, axis=1), multiples=[1,k,1])
                    ctc_len_beams = tf.tile(tf.expand_dims(self._ctc_len, axis=1), multiples=[1,k])

                    with tf.name_scope('top_1'):
                        top1_ed = tfc.edit_distance2D(B,
                                                      top1_ids, top1_seq_lens,
                                                      self._y_ctc, self._ctc_len,
                                                      self._params.CTCBlankTokenID, self.C.SpaceTokenID, self.C.NullTokenID) #(B,)
                        top1_mean_ed = tf.reduce_mean(top1_ed) # scalar
                        top1_hits = tf.to_float(tf.equal(top1_ed, 0))
                        top1_accuracy = tf.reduce_mean(top1_hits) # scalar
                        top1_num_hits = tf.reduce_sum(top1_hits) # scalar

                    with tf.name_scope('bestof_%d'%k):
                        ed = tfc.edit_distance3D(B, k,
                                                 topK_ids, topK_seq_lens,
                                                 y_ctc_beams, ctc_len_beams,
                                                 self._params.CTCBlankTokenID, self.C.SpaceTokenID, self.C.NullTokenID) #(B,k)
                        ## Best of top_k
                        # bok_ed = tf.reduce_min(ed, axis=1) # (B, 1)
                        bok_ed, bok_indices = tfc.batch_bottom_k_2D(ed, 1) # (B, 1)
                        bok_mean_ed = tf.reduce_mean(bok_ed)
                        bok_accuracy = tf.reduce_mean(tf.to_float(tf.equal(bok_ed, 0)))
                        bok_seq_lens =  tf.squeeze(tfc.batch_slice(topK_seq_lens, bok_indices), axis=1) # (B, 1).squeeze => (B,)
                        bok_seq_scores = tf.squeeze(tfc.batch_slice(topK_seq_scores, bok_indices), axis=1) # (B, 1).squeeze => (B,)
                        bok_ids = tf.squeeze(tfc.batch_slice(topK_ids, bok_indices), axis=1) # (B, 1, T).squeeze => (B,T)

                return dlc.Properties({
                    'inp_q': self._inp_q,
                    'image_name': self._image_name, #[(B,), ...]
                    'y_s': self._y_s, # (B, T)
                    'y_ctc': self._y_ctc, # (B,T)
                    'ctc_len': self._ctc_len, # (B,)
                    'top1_ids': top1_ids, # (B, T)
                    'top1_scores': top1_seq_scores, # (B,)
                    'top1_lens': top1_seq_lens, # (B,)
                    'top1_len_ratio': top1_len_ratio,  # (B,)
                    'top1_alpha': alphas[:, :, 0, :, :, :],  # (N, B, BW, H, W, T) -> (N, B, H, W, T)
                    'top1_beta': betas[:, :, 0, :], ## (N, B, BW, T) -> (N, B, T)
                    'topK_ids': topK_ids, # (B, k, T)
                    'topK_scores': topK_seq_scores, # (B, k)
                    'topK_lens': topK_seq_lens, # (B,k)
                    'output_ids': outputs.ids, # (B, T, Seq2SeqBeamWidth)
                    'all_ids': bm_ids, # (B, Seq2SeqBeamWidth, T)
                    'all_id_scores': scores,
                    'all_seq_lens': bm_seq_lens, # (B, Seq2SeqBeamWidth)
                    'top1_num_hits': top1_num_hits, # scalar
                    'top1_accuracy': top1_accuracy, # scalar
                    'top1_ed': top1_ed, #(B,)
                    'top1_mean_ed': top1_mean_ed, # scalar
                    'bok_accuracy': bok_accuracy, # scalar
                    'bok_ed': tf.squeeze(bok_ed, axis=[1], name='bok_ed'), #(B,)
                    'bok_mean_ed': bok_mean_ed, # scalar
                    'bok_seq_lens': bok_seq_lens, # (B,)
                    'bok_seq_scores': bok_seq_scores, #(B,)
                    'bok_ids': bok_ids # (B,T)
                    # 'logs_tr_acc_top1': logs_tr_acc_top1,
                    # 'logs_tr_acc_topK': logs_tr_acc_topK,
                    # 'logs_agg_top1': logs_agg_top1,
                    # 'logs_agg_bok': logs_agg_bok,
                    # 'ph_top1_seq_lens': ph_top1_seq_lens,
                    # 'ph_edit_distance': ph_edit_distance,
                    # 'ph_BoK_distance': ph_BoK_distance,
                    # 'ph_accuracy': ph_accuracy,
                    # 'ph_BoK_accuracy': ph_BoK_accuracy,
                    # 'ph_valid_time': ph_valid_time,
                    # 'ph_num_hits': ph_num_hits
                    })


def sync_training_towers(hyper, tower_ops, global_step, run_tag='training', optimizer=None):
    if run_tag == 'training':
        assert optimizer is not None
    elif run_tag == 'validation':
        assert optimizer is None
    else:
        raise ValueError('run_tag can only have value "training" or "validation"')

    with tf.variable_scope('SyncTowers') as var_scope:
        zero = tf.constant(0.0, shape=(1,))
        zero_list = [zero]
        def allNone(seq):
            return all([item is None for item in seq])

        def gather(prop_name):
            tensor_list = [o[prop_name] for o in tower_ops]
            return tensor_list

        def get_mean(prop_name, op_name=None):
            tensor_list = gather(prop_name)
            return tf.reduce_mean(tensor_list, name=op_name or '%s_out'%prop_name)

        def get_sum(prop_name, op_name=None):
            tensor_list = gather(prop_name)
            return tf.reduce_sum(tensor_list, name=op_name or '%s_out'%prop_name)

        def concat(prop_name, axis=0):
            tensor_list = gather(prop_name)
            return tf.concat(tensor_list, axis=axis, name='%s_out'%prop_name)

        def stack(prop_name, op_name=None):
            tensor_list = gather(prop_name)
            return tf.stack(tensor_list, axis=0, name=op_name or '%s_out'%prop_name)

        log_likelihood = get_mean('log_likelihood')
        ctc_loss = get_mean('ctc_loss')
        alpha_penalty = get_mean('alpha_penalty')
        reg_loss = tower_ops[0]['reg_loss'] # RegLoss should be same for all towers

        if hyper.use_ctc_loss:
            cost_log = ctc_loss + alpha_penalty + reg_loss
        else:
            assert not hyper.build_scanning_RNN
            cost_log = log_likelihood + alpha_penalty + reg_loss

        if optimizer is not None:
            # print gather('grads')
            grads = average_gradients(gather('grads'))
            with tf.variable_scope('optimizer'):
                apply_grads = optimizer.apply_gradients(grads, global_step=global_step)
        else:
            apply_grads = tf.constant(0.0, name='no_training')

        with tf.variable_scope('Instrumentation') as var_scope:
#            tf.summary.scalar('training/regLoss/', reg_loss, collections=['training'])
#            tf.summary.scalar('training/logloss/', log_likelihood, collections=['training'])
#            tf.summary.scalar('training/ctc_loss/', ctc_loss, collections=['training'])
#            tf.summary.scalar('training/alpha_penalty/', alpha_penalty, collections=['training'])
#            tf.summary.scalar('training/total_cost_log/', cost_log, collections=['training'])
#            tf.summary.scalar('training/mean_norm_ase/', get_mean('mean_norm_ase'), collections=['training'])
#            tf.summary.histogram('training/seq_len/', concat('sequence_lengths'), collections=['training'])
#            tf.summary.histogram('training/pred_len_ratio/', concat('pred_len_ratio'), collections=['training'])
#            tf.summary.scalar('training/mean_ctc_ed/', get_mean('mean_ctc_ed'), collections=['training'])
#            tf.summary.scalar('training/num_hits/', get_sum('num_hits'), collections=['training'])

#            tb_logs = tf.summary.merge(tf.get_collection('training'))

            logs = []
            for var in tf.trainable_variables():
                logs.append(tf.summary.histogram('%s/step/%s/'%(run_tag, var.name), var))
            tb_step_logs = tf.summary.merge(logs)

            ## Placeholder for injecting training-time from outside
            ph_train_time = tf.placeholder(hyper.dtype)
            ph_bleu_scores = tf.placeholder(hyper.dtype)
            ph_bleu_score2 = tf.placeholder(hyper.dtype)  # scalar
            ph_ctc_eds = tf.placeholder(hyper.dtype)
            ph_loglosses = tf.placeholder(hyper.dtype)
            ph_ctc_losses = tf.placeholder(hyper.dtype)
            ph_alpha_penalties = tf.placeholder(hyper.dtype)
            ph_costs = tf.placeholder(hyper.dtype)
            ph_mean_norm_ases = tf.placeholder(hyper.dtype)
            ph_mean_norm_aaes = tf.placeholder(hyper.dtype)
            ph_beta_mean = tf.placeholder(hyper.dtype)  # (num_steps, N, B*num_towers)
            ph_beta_std_dev = tf.placeholder(hyper.dtype)
            ph_pred_len_ratios = tf.placeholder(hyper.dtype)
            ph_num_hits = tf.placeholder(hyper.int_type)
            ph_reg_losses = tf.placeholder(hyper.dtype)
            ph_scan_lens = tf.placeholder(hyper.int_type)

            aggs = []
            aggs.append(tf.summary.scalar('%s/time_per100/'%run_tag, ph_train_time))
            aggs.append(tf.summary.histogram('%s/bleu_dist/'%run_tag, ph_bleu_scores))
            aggs.append(tf.summary.scalar('%s/bleu/'%run_tag, tf.reduce_mean(ph_bleu_scores)))
            aggs.append(tf.summary.scalar('%s/bleu2/'%run_tag, ph_bleu_score2))
            aggs.append(tf.summary.histogram('%s/ctc_ed_dist/'%run_tag, ph_ctc_eds))
            aggs.append(tf.summary.scalar('%s/ctc_ed/'%run_tag, tf.reduce_mean(ph_ctc_eds)))
            if not hyper.use_ctc_loss:
                aggs.append(tf.summary.scalar('%s/logloss_mean/'%run_tag, tf.reduce_mean(ph_loglosses)))
                min_logloss = tf.reduce_min(ph_loglosses)
                aggs.append(tf.summary.scalar('%s/logloss/'%run_tag, min_logloss))
                aggs.append(tf.summary.scalar('%s/batch_logloss_min/'%run_tag, min_logloss))
                aggs.append(tf.summary.scalar('%s/batch_logloss_max/'%run_tag, tf.reduce_max(ph_loglosses)))
            aggs.append(tf.summary.scalar('%s/alpha_penalty/'%run_tag, tf.reduce_mean(ph_alpha_penalties)))
            aggs.append(tf.summary.scalar('%s/ctc_loss_mean/'%run_tag, tf.reduce_mean(ph_ctc_losses)))
            min_ctc_loss = tf.reduce_min(ph_ctc_losses)
            aggs.append(tf.summary.scalar('%s/ctc_loss/'%run_tag, min_ctc_loss))
            aggs.append(tf.summary.scalar('%s/batch_ctc_loss_min/'%run_tag, min_ctc_loss))
            aggs.append(tf.summary.scalar('%s/batch_ctc_loss_max/'%run_tag, tf.reduce_max(ph_ctc_losses)))
            aggs.append(tf.summary.scalar('%s/total_cost/'%run_tag, tf.reduce_mean(ph_costs)))
            aggs.append(tf.summary.scalar('%s/mean_norm_ase/'%run_tag, tf.reduce_mean(ph_mean_norm_ases)))
            aggs.append(tf.summary.histogram('%s/mean_norm_ase_dist/'%run_tag, ph_mean_norm_ases))
            aggs.append(tf.summary.scalar('%s/mean_norm_aae/'%run_tag, tf.reduce_mean(ph_mean_norm_aaes)))
            aggs.append(tf.summary.histogram('%s/mean_norm_aae_dist/'%run_tag, ph_mean_norm_aaes))
            aggs.append(tf.summary.scalar('%s/beta_mean/'%run_tag, tf.reduce_mean(ph_beta_mean)))
            aggs.append(tf.summary.scalar('%s/beta_std_dev/'%run_tag, tf.reduce_mean(ph_beta_std_dev)))
            aggs.append(tf.summary.histogram('%s/beta_mean_dist/'%run_tag, ph_beta_mean))
            aggs.append(tf.summary.histogram('%s/beta_std_dev_dest/'%run_tag, ph_beta_std_dev))
            aggs.append(tf.summary.scalar('%s/pred_len_ratio_avg/'%run_tag, tf.reduce_mean(ph_pred_len_ratios)))
            aggs.append(tf.summary.histogram('%s/pred_len_ratio_dist/'%run_tag, ph_pred_len_ratios))
            mean_hits = tf.reduce_mean(tf.cast(ph_num_hits, hyper.dtype))
            aggs.append(tf.summary.scalar('%s/num_hits/'%run_tag, mean_hits))
            aggs.append(tf.summary.scalar('%s/accuracy/'%run_tag, mean_hits / tf.constant(hyper.num_towers * hyper.B*1.0)))
            aggs.append(tf.summary.scalar('%s/regLoss/'%run_tag, tf.reduce_mean(ph_reg_losses)))
            if hyper.build_scanning_RNN:
                aggs.append(tf.summary.scalar('%s/scanLen/'%run_tag, tf.reduce_mean(tf.cast(ph_scan_lens, dtype=hyper.dtype))))
                aggs.append(tf.summary.histogram('%s/scanLen_dist/'%run_tag, ph_scan_lens))


            tb_agg_logs = tf.summary.merge(aggs)

            if (hyper.CTCBlankTokenID is None) and (hyper.SpaceTokenID is None):
                ctc_ed = zero
                mean_ctc_ed = zero
                pred_len_ratio = zero
                num_hits = zero
                predicted_ids_list = zero
                predicted_lens = zero
            else:
                ctc_ed = concat('ctc_ed') # (num_towers*B,)
                pred_len_ratio = concat('pred_len_ratio') # (num_towers*B,)
                num_hits = get_sum('num_hits')  # scalar
                predicted_ids_list = gather('predicted_ids')  # [(B,T), ...]
                predicted_lens = concat('predicted_lens')  # (num_towers*B,)

            if hyper.build_scanning_RNN:
                scan_len = concat('scan_len')  # (num_towers*B,)
            else:
                scan_len = zero

        return dlc.Properties({
            'train': apply_grads,  # op
            'global_step': global_step,  # scalar
            'image_name_list': gather('image_name'),  # [(B,), ...]
            'alpha': concat('alpha', axis=1),  # (N, num_towers*B, H, W, T)
            'beta': concat('beta', axis=1),  # (N, num_towers*B, T)
            'log_likelihood': log_likelihood,  # scalar
            'ph_loglosses': ph_loglosses,  # (num_steps,)
            'ctc_loss': ctc_loss,  # scalar
            'ph_ctc_losses': ph_ctc_losses,  # (num_steps,)
            'loss': ctc_loss if hyper.use_ctc_loss else log_likelihood,  # scalar
            'alpha_penalty': alpha_penalty,  # scalarconcat_scalar
            'ph_alpha_penalties': ph_alpha_penalties,  # (num_steps,)
            'mean_norm_ase': get_mean('mean_norm_ase'),  # scalar between 0. and 100.
            'ph_mean_norm_ases': ph_mean_norm_ases,  # (num_steps,)
            'mean_norm_aae': get_mean('mean_norm_aae'),  # scalar between 0. and 100.
            'ph_mean_norm_aaes': ph_mean_norm_aaes,  # (num_steps,)
            'beta_mean': concat('beta_mean', axis=1),  # (N, B*num_towers)
            'ph_beta_mean': ph_beta_mean,  # (num_steps, N, B*num_towers)
            'beta_std_dev': concat('beta_std_dev', axis=1),  # (N, B*num_towers)
            'ph_beta_std_dev': ph_beta_std_dev,  # (num_steps, N, B*num_towers)
            'num_hits': num_hits,  # scalar
            'ph_num_hits': ph_num_hits,  # (num_steps,)
            'reg_loss': reg_loss,  # scalar
            'ph_reg_losses': ph_reg_losses,  # (num_steps,)
            'cost': cost_log,  # scalar
            'ph_costs': ph_costs,  # (num_steps,)
            'ctc_ed': ctc_ed,  # (num_towers*B,)
            'ph_ctc_eds': ph_ctc_eds,  # (num_steps*num_towers*B,)
            'predicted_ids_list': predicted_ids_list,  # [(B,T), ...]
            'predicted_lens': predicted_lens,  # (num_towers*B,)
            'pred_len_ratio': pred_len_ratio,  # (num_towers*B,)
            'ph_pred_len_ratios': ph_pred_len_ratios,  # (num_steps*num_towers*B,)
            'x_s_list': gather('x_s'),  # [(B,T),...] or [(B,S,3),...]
            'y_s_list': gather('y_s'),  # [(B,T),...]
            'y_ctc_list': gather('y_ctc'),  # [(B,T),...]
            'ctc_len': concat('ctc_len'),  # (num_towers*B,)
            'scan_len': scan_len,  # (num_towers*B,)
            'bin_len': stack('bin_len'),  # (num_towers*B,)
            'ph_scan_lens': ph_scan_lens,  # (num_steps*num_towers*B,)
            'ph_train_time': ph_train_time,  # scalar
            'ph_bleu_scores': ph_bleu_scores,  # (num_steps*num_towers*B,)
            'ph_bleu_score2': ph_bleu_score2,  # scalar
            'tb_agg_logs': tb_agg_logs,  # summary string
            'tb_step_logs': tb_step_logs,  # summary string
        })

def sync_testing_towers(hyper, tower_ops, run_tag='validation'):
    # n = len(tower_ops)
    BW = hyper.seq2seq_beam_width
    k = min([hyper.k, BW])

    with tf.variable_scope('SyncTowers') as var_scope:
        def gather(prop_name):
            return [o[prop_name] for o in tower_ops]
        def get_mean(prop_name, op_name=None):
            return tf.reduce_mean(gather(prop_name), name=op_name or prop_name)
        def get_sum(prop_name, op_name=None):
            return tf.reduce_sum(gather(prop_name), name=op_name or prop_name)
        def concat(prop_name, axis=0):
            return tf.concat(gather(prop_name), axis=axis)

        top1_lens = concat('top1_lens') # (n*B,)
        top1_len_ratio = concat('top1_len_ratio') #(n*B,)
        bok_seq_lens = concat('bok_seq_lens') # (n*B,)
        top1_scores = concat('top1_scores') # (n*B,)
        bok_seq_scores = concat('bok_seq_scores') # (n*B,)
        top1_mean_ed = get_mean('top1_mean_ed') # scalar
        bok_mean_ed = get_mean('bok_mean_ed') # scalar
        top1_accuracy = get_mean('top1_accuracy') # scalar
        bok_accuracy = get_mean('bok_accuracy') # scalar
        top1_num_hits = get_sum('top1_num_hits') # scalar
        # top1_ids_list = gather('top1_ids') # (B, T)
        # bok_ids_list = gather('bok_ids') # (B, T)
        # y_s_list = gather('y_s') # (B, T)

        ## Batch Metrics
        tf.summary.histogram( 'scores/top_1', top1_scores, collections=['top1'])
        tf.summary.histogram( 'top1_len_ratio/top_1', top1_len_ratio, collections=['top1'])
        tf.summary.scalar( 'edit_distances/top_1', top1_mean_ed, collections=['top1'])
        ## tf.summary.scalar( 'accuracy/top_1', top1_accuracy, collections=['top1'])
        tf.summary.scalar('num_hits/top_1', top1_num_hits, collections=['top1'])

        tf.summary.scalar( 'edit_distances/top_%d'%k, bok_mean_ed, collections=['top_k'])
        tf.summary.scalar( 'accuracy/top_%d'%k, bok_accuracy, collections=['top_k'])
        tf.summary.histogram('seq_lens/top_%d'%k, bok_seq_lens, collections=['top_k'])
        tf.summary.histogram('scores/top_%d'%k, bok_seq_scores, collections=['top_k'])

        logs_top1 = tf.summary.merge(tf.get_collection('top1'))
        logs_topK = tf.summary.merge(tf.get_collection('top_k'))

        ## Aggregate metrics injected into the graph from outside
        ## BLEU scores
        # with tf.name_scope('BLEU'):
        #     ## BLEU score is calculated outside of TensorFlow and then injected back in via. a placeholder
        #     ph_bleu = tf.placeholder(tf.float32, shape=(self.C.B,), name="BLEU_placeholder")
        #     tf.summary.histogram( 'predicted/bleu', ph_bleu, collections=['bleu'])BW
        #     tf.summary.scalar( 'predicted/bleuH', tf.reduce_mean(ph_bleu), collections=['bleu'])
        #     logs_b = tf.summary.merge(tf.get_collection('bleu'))

        with tf.name_scope('AggregateMetrics'):
            ph_top1_seq_lens = tf.placeholder(hyper.int_type) # (num_batches,n*B,)
            ph_top1_len_ratio = tf.placeholder(hyper.dtype) # (num_batches,n*B,)
            ph_edit_distance = tf.placeholder(hyper.dtype) # (num_batches,)
            ph_bleus = tf.placeholder(hyper.dtype) # (num_gpus*B,)
            ph_bleu2 = tf.placeholder(hyper.dtype) # scalar
            ph_num_hits =  tf.placeholder(hyper.dtype) # (num_batches,)
            ph_accuracy =  tf.placeholder(hyper.dtype) # (num_batches,)
            ph_valid_time =  tf.placeholder(hyper.dtype) # scalar
            ph_full_validation = tf.placeholder(dtype=tf.uint8)

            ph_BoK_distance =  tf.placeholder(hyper.dtype) # (num_batches,)
            ph_BoK_accuracy =  tf.placeholder(hyper.dtype) # (num_batches,)
            ph_BoK_bleus = tf.placeholder(hyper.dtype) # (num_gpus*B,)
            ph_BoK_bleu2 = tf.placeholder(hyper.dtype) # scalar

            agg_num_hits = tf.reduce_sum(ph_num_hits)
            agg_accuracy = tf.reduce_mean(ph_accuracy)
            agg_ed = tf.reduce_mean(ph_edit_distance)
            agg_bok_ed = tf.reduce_mean(ph_BoK_distance)
            agg_bok_accuracy = tf.reduce_mean(ph_BoK_accuracy)

            top1 = []
            # tf.summary.histogram( 'top_1/seq_lens', ph_top1_seq_lens, collections=['aggregate_top1'])
            # top1.append(tf.summary.scalar('full_validation', ph_full_validation))
            top1.append(tf.summary.histogram('%s/top_1/top1_len_ratio/'%run_tag, ph_top1_len_ratio))
            top1.append(tf.summary.histogram('%s/top_1/edit_distances/'%run_tag, ph_edit_distance))
            top1.append(tf.summary.scalar('%s/top_1/edit_distance/'%run_tag, agg_ed))
            top1.append(tf.summary.scalar('%s/top_1/num_hits/'%run_tag, agg_num_hits))
            top1.append(tf.summary.scalar('%s/top_1/accuracy/'%run_tag, agg_accuracy))
            top1.append(tf.summary.scalar('%s/time_per100/'%run_tag, ph_valid_time))
            top1.append(tf.summary.histogram('%s/top_1/bleu_dist/'%run_tag, ph_bleus))
            top1.append(tf.summary.scalar('%s/top_1/bleu/'%run_tag, tf.reduce_mean(ph_bleus)))
            top1.append(tf.summary.scalar('%s/top_1/bleu2/'%run_tag, ph_bleu2))

            bok  = []
            # bok.append( tf.summary.histogram('%s/bestof_%d/edit_distances/'%(run_tag, k), ph_BoK_distance))
            bok.append(tf.summary.scalar('%s/bestof_%d/accuracy/'%(run_tag, k), agg_bok_accuracy))
            bok.append(tf.summary.scalar('%s/bestof_%d/edit_distance/'%(run_tag, k), agg_bok_ed))
            bok.append(tf.summary.scalar('%s/bestof_%d/bleu/' % (run_tag, k), tf.reduce_mean(ph_BoK_bleus)))
            bok.append(tf.summary.scalar('%s/bestof_%d/bleu2/' % (run_tag, k), ph_BoK_bleu2))

            logs_agg_top1 = tf.summary.merge(top1)
            logs_agg_bok = tf.summary.merge(bok)


        return dlc.Properties({
            'image_name_list': gather('image_name'),
            'top1_lens': top1_lens,  # (n*B,)
            'top1_len_ratio': top1_len_ratio,  #(n*B,)
            'top1_ed': concat('top1_ed'),  # (n*B,)
            'top1_mean_ed': top1_mean_ed,  # scalar
            'top1_accuracy': top1_accuracy,  # scalar
            'top1_num_hits': top1_num_hits,  # scalar
            'top1_ids_list': gather('top1_ids'),  # ((B,T),...)
            'top1_alpha_list': gather('top1_alpha'),  # [(N, B, H, W, T), ...]
            'top1_beta_list': gather('top1_beta'),  # [(N, B, T), ...]
            'bok_ed': concat('bok_ed'),  # (n*B,)
            'bok_mean_ed': bok_mean_ed,  # scalar
            'bok_accuracy': bok_accuracy,  # scalar
            'bok_seq_lens': bok_seq_lens,  # (n*B,)
            'bok_ids_list': gather('bok_ids'),  # ((B,T),...)
            'y_s_list': gather('y_s'),  # ((B,T),...)
            'y_ctc_list': gather('y_ctc'),  # [(B,T),...]
            'ctc_len': concat('ctc_len'),  # (num_gpus*B,)
            'all_ids_list': gather('all_ids'),  # ((B, BW, T), ...)
            'output_ids_list': gather('output_ids'),
            'logs_top1': logs_top1,
            'logs_topK': logs_topK,
            'ph_top1_seq_lens': ph_top1_seq_lens,
            'ph_top1_len_ratio': ph_top1_len_ratio,
            'ph_edit_distance': ph_edit_distance,
            'ph_bleus': ph_bleus,
            'ph_BoK_bleus': ph_BoK_bleus,
            'ph_bleu2': ph_bleu2,
            'ph_BoK_bleu2': ph_BoK_bleu2,
            'ph_num_hits': ph_num_hits,
            'ph_accuracy': ph_accuracy,
            'ph_valid_time': ph_valid_time,
            'ph_BoK_distance': ph_BoK_distance,
            'ph_BoK_accuracy': ph_BoK_accuracy,
            'logs_agg_top1': logs_agg_top1,
            'logs_agg_bok': logs_agg_bok,
            'ph_full_validation': ph_full_validation
            })
