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
import dl_commons as dlc
import tf_commons as tfc
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Embedding, Dense, Activation, Dropout, Concatenate, Permute
from keras import backend as K
from dl_commons import PD, mandatory, boolean, integer, decimal, equalto, instanceof
from Im2LatexRNN import Im2LatexDecoderRNN, Im2LatexDecoderRNNParams

HYPER_PD = (
        ## Global Properties, used at various places
        PD('image_shape',
           'Shape of input images. Should be a python sequence.',
           None,
           (120,1075,3)
           ),
        PD('Max_Seq_Len',
           "Max sequence length including the end-of-sequence marker token. Is used to " 
            "limit the number of decoding steps.",
           integer(151,200),
           151 #get_max_seq_len(data_folder)
           ),
        PD('B',
           '(integer or None): Size of mini-batch for training, validation and testing.',
           (None, 128),
           128
           ),
        PD('K',
           'Vocabulary size including zero',
           xrange(500,1000),
           556 #get_vocab_size(data_folder)
           ),
        PD('m',
           '(integer): dimensionality of the embedded input vector (Ey / Ex)', 
           xrange(50,250),
           64
           ),
        PD('H', 'Height of feature-map produced by conv-net. Specific to the dataset image size.', None, 3),
        PD('W', 'Width of feature-map produced by conv-net. Specific to the dataset image size.', None, 33),
        PD('L',
           '(integer): number of pixels in an image feature-map = HxW (see paper or model description)', 
           integer(1),
           lambda _, d: d['H'] * d['W']),
        PD('D', 
           '(integer): number of features coming out of the conv-net. Depth/channels of the last conv-net layer.'
           'See paper or model description.', 
           integer(1),
           512),
        PD('keep_prob', '(decimal): Value between 0.1 and 1.0 indicating the keep_probability of dropout layers.'
           'A value of 1 implies no dropout.',
           decimal(0.1, 1), 
           1.0),
        PD('tb', "Tensorboard Params.",
           instanceof(tfc.TensorboardParams),
           tfc.TensorboardParams()),
        PD('dropout', 
           'Dropout parameters if any - global at the level of the RNN.',
           instanceof(tfc.DropoutParams, True)),
    ### Attention Model Params ###
        PD('att_layers', 'Number of layers in the attention_a model', xrange(1,10), 1),
        PD('att_1_n', 'Number of units in first layer of the attention model. Defaults to D as it is in the paper"s source-code.', 
           xrange(1,10000),
           equalto('D')),
        PD('att_share_weights', 'Whether the attention model should share weights across the "L" image locations or not.'
           'Choosing "True" conforms to the paper resulting in a (D+n,att_1_n) weight matrix. Choosing False will result in a MLP with (L*D+n,att_1_n) weight matrix. ',
           boolean,
           True),
        PD('att_activation', 
           'Activation to use for the attention MLP model. Defaults to tanh as in the paper source.',
           None,
           'tanh'),
        PD('att_weighted_gather', 'The paper"s source uses an affine transform with trainable weights, to narrow the output of the attention'
           "model from (B,L,dim) to (B,L,1). I don't think this is helpful since there is no nonlinearity here." 
           "Therefore I have an alternative implementation that simply averages the matrix (B,L,dim) to (B,L,1)." 
           "Default value however, is True in conformance with the paper's implementation.",
           (True, False),
           True),
        PD('att_weights_initializer', 'weights initializer to use for the attention model', None,
           'glorot_normal'),
    ### Embedding Layer ###
        PD('embeddings_initializer', 'Initializer for embedding weights', None, 'glorot_uniform'),
        PD('embeddings_initializer_tf', 'Initializer for embedding weights', dlc.iscallable(), 
           tf.contrib.layers.xavier_initializer()),
    ### Decoder LSTM Params ###
        PD('n',
           '(integer): Number of hidden-units of the LSTM cell',
           integer(100,10000),
           1000),
        PD('decoder_lstm_peephole',
           '(boolean): whether to employ peephole connections in the decoder LSTM',
           (True, False),
           False),

        PD('output_follow_paper',
           '(boolean): Output deep layer uses some funky logic in the paper instead of a straight MLP'
           'Setting this value to True (default) will follow the paper"s logic. Otherwise'
           "a straight MLP will be used.", 
           boolean, 
           True),
        PD('output_layers',
           "(MLPParams): Parameters for the output MLP. The last layer outputs the logits and therefore "
           "must have num_units = K. If output_follow_paper==True, an additional initial layer is created " 
           "with num_units = m and activtion tanh. Note: In the paper all layers have num_units=m",
           instanceof(tfc.MLPParams)),

        PD('decoder_out_layers',
           'Number of layers in the decoder output MLP. defaults to 1 as in the papers source',
           xrange(1,10), 1),
        PD('output_activation', 'Activtion function for deep output layer', None,
           'tanh'),
        PD('output_1_n', 
           'Number of units in the first hidden layer of the output MLP. Used only if output_follow_paper == False'
           "Default's to 'm' - same as when output_follow_paper == True", None,
           equalto('m')),
    ### Initializer MLP ###
        PD('init_layers', 'Number of layers in the initializer MLP', xrange(1,10),
           1),
        PD('init_dropout_rate', '(decimal): Global dropout_rate variable for init_layer',
           decimal(0.0, 0.9), 
           0.2),
        PD('init_h_activation', '', None, 'tanh'),
        PD('init_h_dropout_rate', '', 
           decimal(0.0, 0.9), 
           equalto('init_dropout_rate')),
        PD('init_c_activation', '', None, 'tanh'),
        PD('init_c_dropout_rate', '', 
           decimal(0.0, 0.9),
           equalto('init_dropout_rate')),
        PD('init_1_n', 'Number of units in hidden layer 1. The paper sets it to D',
           integer(1, 10000), 
           equalto('D')),
        PD('init_1_dropout_rate', '(decimal): dropout rate for the layer', 
           decimal(0.0, 0.9), 
           0.),
        PD('init_1_activation', 
           'Activation function of the first layer. In the paper, the final' 
           'layer has tanh and all penultinate layers have relu activation', 
           None,
           'tanh'),
    ### Loss / Cost Layer ###
        PD('sum_logloss',
           'Whether to normalize log-loss per sample as in standard log perplexity ' 
           'calculation or whether to just sum up log-losses as in the paper. Defaults' 
           'to True in conformance with the paper.',
           boolean,
           True
          ),
        PD('MeanSumAlphaEquals1',
          '(boolean): When calculating the alpha penalty, the paper uses the term: '
           'square{1 - sum_over_t{alpha_t_i}}). This assumes that the mean sum_over_t should be 1. '
           "However, that's not true, since the mean of sum_over_t term should be C/L. This "
           "variable if set to True, causes the term to change to square{C/L - sum_over_t{alpha_t_i}}). "
           "The default value is True in conformance with the paper.",
          boolean,
          False),
        PD('pLambda', 'Lambda value for alpha penalty',
           decimal(0),
           0.0001)   
)

HYPER = dlc.HyperParams(HYPER_PD,
        ## Overrides of default values.
        ## FYI: By convention, all boolean params' default value is True
        {
            'att_weighted_gather': True,
            'sum_logloss': True,
            'MeanSumAlphaEquals1': True,
            'output_follow_paper': True
        })
HYPER.output_layers = tfc.MLPParams({
        ## One layer with num_units = m is added if output_follow_paper == True
        ## Last layer must have num_units = K because it outputs logits.
        ## paper has all layers with num_units = m. I've noteiced that they build rectangular MLPs, i.e. not triangular.
        'layers_units': (HYPER.m, HYPER.K),
        'activation_fn': tf.nn.relu, # paper has it set to relu
        'tb': HYPER.tb,
        'dropout': tfc.DropoutParams({'keep_prob': 0.9})
        })


print 'HYPER = ', HYPER

class Im2LatexModel(object):
    """
    One timestep of the decoder model. The entire function can be seen as a complex RNN-cell
    that includes a LSTM stack and an attention model.
    """
    def __init__(self):
        self._define_params()
        self._numSteps = 0

    def _define_attention_params(self):
        """Define Shared Weights for Attention Model"""
        ## 1) Dense layer, 2) Optional gather layer and 3) softmax layer

        ## Renaming HyperParams for convenience
        B = HYPER.B
        n = HYPER.n
        L = HYPER.L
        D = HYPER.D
        
        ## _att_dense array indices start from 1
        self._att_dense_layer = []

        if HYPER.att_share_weights:
        ## Here we'll effectively create L MLP stacks all sharing the same weights. Each
        ## stack receives a concatenated vector of a(l) and h as input.
            dim = D+n
            for i in range(1, HYPER.att_layers+1):
                n_units = HYPER['att_%d_n'%(i,)]; assert(n_units <= dim)
                self._att_dense_layer.append(Dense(n_units, activation=HYPER.att_activation,
                                                   batch_input_shape=(B,L,dim)))
                dim = n_units
            ## Optional gather layer (that comes after the Dense Layer)
            if HYPER.att_weighted_gather:
                self._att_gather_layer = Dense(1, activation='linear') # output shape = (B, L, 1)
        else:
            ## concatenate a and h_prev and pass them through a MLP. This is different than the theano
            ## implementation of the paper because we flatten a from (B,L,D) to (B,L*D). Hence each element
            ## of the L*D vector receives its own weight because the effective weight matrix here would be
            ## shape (L*D, num_dense_units) as compared to (D, num_dense_units) as in the shared_weights case
            dim = L*D+n        
            for i in range(1, HYPER.att_layers+1):
                n_units = HYPER['att_%d_n'%(i,)]; assert(n_units <= dim)
                self._att_dense_layer.append(Dense(n_units, activation=HYPER.att_actv,
                                                   batch_input_shape=(B,dim)))
                dim = n_units
        
        assert dim >= L
        self._att_softmax_layer = Dense(L, activation='softmax', name='alpha')
        
    def _build_attention_model(self, a, h_prev):
        B = HYPER.B
        n = HYPER.n
        L = HYPER.L
        D = HYPER.D
        h = h_prev

        assert K.int_shape(h_prev) == (B, n)
        assert K.int_shape(a) == (B, L, D)

        ## For #layers > 1 this will endup being different than the paper's implementation
        if HYPER.att_share_weights:
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
            ah = tf.concat([a,h], -1)
            for i in range(HYPER.att_layers) :
                ah = self._att_dense_layer[i](ah)

            ## Below is roughly how it is implemented in the code released by the authors of the paper
#                 for i in range(1, HYPER.att_a_layers+1):
#                     a = Dense(HYPER['att_a_%d_n'%(i,)], activation=HYPER.att_actv)(a)
#                 for i in range(1, HYPER.att_h_layers+1):
#                     h = Dense(HYPER['att_h_%d_n'%(i,)], activation=HYPER.att_actv)(h)    
#                ah = a + K.expand_dims(h, axis=1)

            ## Gather all activations across the features; go from (B, L, dim) to (B,L,1).
            ## One could've just summed/averaged them all here, but the paper uses yet
            ## another set of weights to accomplish this. So we'll keeep that as an option.
            if HYPER.att_weighted_gather:
                ah = self._att_gather_layer(ah) # output shape = (B, L, 1)
                ah = K.squeeze(ah, axis=2) # output shape = (B, L)
            else:
                ah = K.mean(ah, axis=2) # output shape = (B, L)

        else: # weights not shared across L
            ## concatenate a and h_prev and pass them through a MLP. This is different than the theano
            ## implementation of the paper because we flatten a from (B,L,D) to (B,L*D). Hence each element
            ## of the L*D vector receives its own weight because the effective weight matrix here would be
            ## shape (L*D, num_dense_units) as compared to (D, num_dense_units) as in the shared_weights case

            ## Concatenate a and h. Final shape will be (B, L*D+n)
            ah = K.concatenate(K.batch_flatten(a), h)
            for i in range(HYPER.att_layers):
                ah = self._att_dense_layer(ah)
            ## At this point, ah.shape = (B, dim)

        alpha = self._att_softmax_layer(ah) # output shape = (B, L)
        assert K.int_shape(alpha) == (B, L)
        return alpha
            
    def _define_output_params(self):
        ## Renaming HyperParams for convenience
        B = HYPER.B
        n = HYPER.n
        D = HYPER.D
        m = HYPER.m
        Kv= HYPER.K

        ## First layer of output MLP
        ## Affine transformation of h_t and z_t from size n/D to m followed by a summation
        self._output_affine = Dense(m, activation='linear', batch_input_shape=(B,n+D)) # output size = (B, m)
        ## non-linearity for the first layer - will be chained by the _call function after adding Ex / Ey
        self._output_activation = Activation(HYPER.output_activation)

        ## Additional layers if any
        if HYPER.decoder_out_layers > 1:
            self._output_dense = []
            for i in range(1, HYPER.decoder_out_layers):
                self._output_dense.append(Dense(m, activation=HYPER['output_%d_activation'%i], 
                                           batch_input_shape=(B,m))
                                         )

        ## Final softmax layer
        self._output_softmax = Dense(Kv, activation='softmax', batch_input_shape=(B,m))
        
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
        n = self._LSTM_cell.output_size
        
        assert K.int_shape(Ex_t) == (B, m)
        self._LSTM_cell.assertOutputShape(h_t)
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

    def _define_init_params(self):
        ## As per the paper, this is a two-headed MLP. It has a stack of common layers at the bottom
        ## two output layers at the top - one each for h and c LSTM states.
        self._init_layer = []
        self._init_dropout = []
        for i in xrange(1, HYPER.init_layers):
            key = 'init_%d_'%i
            self._init_layer.append(Dense(HYPER[key+'n'], activation=HYPER[key+'activation']))
            if HYPER[key+'dropout_rate'] > 0.0:
                self._init_dropout.append(Dropout(HYPER[key+'dropout_rate']))

        ## Final layer for h
        self._init_h = Dense(HYPER['n'], activation=HYPER['init_h_activation'])
        if HYPER['init_h_dropout_rate'] > 0.0:
            self._init_h_dropout = Dropout(HYPER['init_h_dropout_rate'])

        ## Final layer for c
        self._init_c = Dense(HYPER['n'], activation=HYPER['init_c_activation'])
        if HYPER['init_c_dropout_rate'] > 0.0:
            self._init_c_dropout = Dropout(HYPER['init_c_dropout_rate'])

    def _build_init_layer(self, a):
        assert K.int_shape(a) == (HYPER.B, HYPER.L, HYPER.D)
        
        ################ Initializer MLP ################
        with tf.variable_scope('Initializer_MLP'):

            ## As per the paper, this is a two-headed MLP. It has a stack of common layers at the bottom,
            ## two output layers at the top - one each for h and c LSTM states.
            a = K.mean(a, axis=1) # final shape = (B, D)
            for i in xrange(1, HYPER.init_layers):
                key = 'init_%d_'%i
                a = self._init_layer[i](a)
                if HYPER[key+'dropout_rate'] > 0.0:
                    a = self._init_dropout[i](a)

            init_c = self._init_c(a)
            if HYPER['init_c_dropout_rate'] > 0.0:
                init_c = self._init_c_dropout(init_c)

            init_h = self._init_h(a)
            if HYPER['init_h_dropout_rate'] > 0.0:
                init_h = self._init_h_dropout(init_h)

            assert K.int_shape(init_c) == (HYPER.B, HYPER.n)
            assert K.int_shape(init_h) == (HYPER.B, HYPER.n)

        return init_c, init_h
            
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

        ################ Attention Model ################
        with tf.variable_scope('Attention'):
            self._define_attention_params()
                
        ################ Embedding Layer ################
        with tf.variable_scope('Ey'):
            self._embedding = Embedding(Kv, m, 
                                        embeddings_initializer=e_init, 
                                        mask_zero=True, 
                                        input_length=1,
                                        batch_input_shape=(B,1)
                                        #input_shape=(1,)
                                        ) ## (B, 1, m)
            
            ## Above Embedding layer will get replaced by this one.
            self._embedding_matrix = tf.get_variable('Embedding_Matrix', (Kv, m))
        
        ################ Decoder LSTM Cell ################
        with tf.variable_scope('Decoder_LSTM'):
            self._decoder_lstm = tf.contrib.rnn.LSTMBlockCell(n, forget_bias=1.0, 
                                                              use_peephole=HYPER.decoder_lstm_peephole)
            
        ################ Output Layer ################
        with tf.variable_scope('Decoder_Output_Layer'):
            self._define_output_params()

        ################ Initializer MLP ################
        with tf.variable_scope('Initializer_MLP'):
            self._define_init_params()
            
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
    
    Im2LatexState = collections.namedtuple('Im2LatexState', ('step', 'h', 'lstm_states', 'a', 'yProbs', 
                                                             'yLogits', 'alpha'))
    def _build_rnn_step(self, out_t_1, x_t, isStep1=False, testing=False):
        """
        TODO: Incorporate Dropout
        Builds/threads tf graph for one RNN iteration.
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
        #x_t = input at t             # shape = (B,)
        step = out_t_1[0] + 1
        h_t_1 = out_t_1[1]            # shape = (B,n)
        lstm_states_t_1 = out_t_1[2]  # shape = ((B,n), (B,n)) = (c_t_1, h_t_1)
        a = out_t_1[3]                # shape = (B, L, D)
        if not isStep1: ## init_accum does not have everything
            yProbs_t_1 = out_t_1[4]           # shape = (B, Kv)
        #yLogits_t_1 = out_t_1[5]          # shape = (B, Kv)
        #alpha_t_1 = out_t_1[6]
        
        B = HYPER.B
        m = HYPER.m
        n = HYPER.n
        L = HYPER.L
        D = HYPER.D
        Kv = HYPER.K
        
        assert K.int_shape(h_t_1) == (B, n)
        assert K.int_shape(a) == (B, L, D)
        assert K.int_shape(lstm_states_t_1[1]) == (B, n)
        
        if not isStep1:
            assert K.int_shape(yProbs_t_1) == (B, Kv)
            tf.get_variable_scope().reuse_variables()
            if testing:
                x_t = tf.argmax(yProbs_t_1, axis=1)
        elif testing:
            tf.get_variable_scope().reuse_variables()
        
        ################ Attention Model ################
        with tf.variable_scope('Attention'):
            alpha_t = self._build_attention_model(a, h_t_1) # alpha.shape = (B, L)

        ################ Soft deterministic attention: z = alpha-weighted mean of a ################
        ## (B, L) batch_dot (B,L,D) -> (B, D)
        with tf.variable_scope('Phi'):
            z_t = K.batch_dot(alpha_t, a, axes=[1,1]) # z_t.shape = (B, D)

        ################ Embedding layer ################
        with tf.variable_scope('Ey'):
            Ex_t = self._embedding(K.expand_dims(x_t, axis=-1) ) # output.shape= (None,1,m)
            Ex_t = K.squeeze(Ex_t, axis=1) # output.shape= (None,m)
            Ex_t = K.reshape(Ex_t, (B,m)) # (B,m)
            
        ################ Decoder Layer ################
        with tf.variable_scope("Decoder_LSTM"):
            (h_t, lstm_states_t) = self._decoder_lstm(K.concatenate((Ex_t, z_t)), lstm_states_t_1) # h_t.shape=(B,n)
            
        ################ Output Layer ################
        with tf.variable_scope('Output_Layer'):
#            yProbs_t, yLogits_t = self._build_output_layer(Ex_t, h_t, z_t) # yProbs_t.shape = (B,K)
            yProbs_t, yLogits_t = self._output_layer(Ex_t, h_t, z_t) # yProbs_t.shape = (B,K)
        
        assert K.int_shape(h_t) == (B, n)
        assert K.int_shape(a) == (B, L, D)
        assert K.int_shape(lstm_states_t[1]) == (B, n)
        assert K.int_shape(yProbs_t) == (B, Kv)
        assert K.int_shape(yLogits_t) == (B, Kv)
        assert K.int_shape(alpha_t) == (B, L)
        
        return step, h_t, lstm_states_t, a, yProbs_t, yLogits_t, alpha_t
        
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
        
