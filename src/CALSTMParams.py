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

Created on Wed Jul 12 14:50:29 2017
@author: Sumeet S Singh
"""
import dl_commons as dlc
import tf_commons as tfc
from dl_commons import PD, instanceof, integer, decimal, boolean, equalto
import tensorflow as tf

class CALSTMParams(dlc.HyperParams):
    proto = (
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
        PD('n',
           "The variable n in the paper. The number of units in the decoder_lstm cell(s). "
           "The paper uses a value of 1000.",
           integer(),
           1000),
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
        PD('tb', "Tensorboard Params.",
           instanceof(tfc.TensorboardParams),
           tfc.TensorboardParams()),
        PD('dropout', 
           'Dropout parameters if any - global at the level of the RNN.',
           instanceof(tfc.DropoutParams, True)),
#        PD('keep_prob', '(decimal): Value between 0.1 and 1.0 indicating the keep_probability of dropout layers.'
#           'A value of 1 implies no dropout.',
#           decimal(0.1, 1), 
#           1.0),
    ### Attention Model Params ###
        PD('att_layers', 'MLP parameters', instanceof(tfc.MLPParams)),
        PD('att_share_weights', 'Whether the attention model should share weights across the "L" image locations or not.'
           'Choosing "True" conforms to the paper resulting in a (D+n,att_1_n) weight matrix. Choosing False will result in a MLP with (L*D+n,att_1_n) weight matrix. ',
           boolean,
           True),
        PD('att_weighted_gather', 'The paper"s source uses an affine transform with trainable weights, to narrow the output of the attention'
           "model from (B,L,dim) to (B,L,1). I don't think this is helpful since there is no nonlinearity here." 
           "Therefore I have an alternative implementation that simply averages the matrix (B,L,dim) to (B,L,1)." 
           "Default value however, is True in conformance with the paper's implementation.",
           (True, False),
           True),
    ### Embedding Layer ###
#        PD('embeddings_initializer', 'Initializer for embedding weights', None, 'glorot_uniform'),
#        PD('embeddings_initializer_tf', 'Initializer for embedding weights', dlc.iscallable(), 
#           tf.contrib.layers.xavier_initializer()),
    ### Decoder LSTM Params ###
        PD('decoder_lstm', 'Decoder LSTM parameters. At this time only one layer is supported.',
           instanceof(tfc.RNNParams)),
#        PD('output_follow_paper',
#           '(boolean): Output deep layer uses some funky logic in the paper instead of a straight MLP'
#           'Setting this value to True (default) will follow the paper"s logic. Otherwise'
#           "a straight MLP will be used.", 
#           boolean, 
#           True),
#        PD('output_layers',
#           "(MLPParams): Parameters for the output MLP. The last layer outputs the logits and therefore "
#           "must have num_units = K. If output_follow_paper==True, an additional initial layer is created " 
#           "with num_units = m and activtion tanh. Note: In the paper all layers have num_units=m",
#           instanceof(tfc.MLPParams)),
    ### Initializer MLP ###
#        PD('init_layers', 'Number of layers in the initializer MLP', xrange(1,10),
#           1),
#        PD('init_dropout_rate', '(decimal): Global dropout_rate variable for init_layer',
#           decimal(0.0, 0.9), 
#           0.2),
#        PD('init_h_activation', '', None, 'tanh'),
#        PD('init_h_dropout_rate', '', 
#           decimal(0.0, 0.9), 
#           equalto('init_dropout_rate')),
#        PD('init_c_activation', '', None, 'tanh'),
#        PD('init_c_dropout_rate', '', 
#           decimal(0.0, 0.9),
#           equalto('init_dropout_rate')),
#        PD('init_1_n', 'Number of units in hidden layer 1. The paper sets it to D',
#           integer(1, 10000), 
#           equalto('D')),
#        PD('init_1_dropout_rate', '(decimal): dropout rate for the layer', 
#           decimal(0.0, 0.9), 
#           0.),
#        PD('init_1_activation', 
#           'Activation function of the first layer. In the paper, the final' 
#           'layer has tanh and all penultinate layers have relu activation', 
#           None,
#           'tanh'),
    ### Loss / Cost Layer ###
#        PD('sum_logloss',
#           'Whether to normalize log-loss per sample as in standard log perplexity ' 
#           'calculation or whether to just sum up log-losses as in the paper. Defaults' 
#           'to True in conformance with the paper.',
#           boolean,
#           True
#          ),
#        PD('MeanSumAlphaEquals1',
#          '(boolean): When calculating the alpha penalty, the paper uses the term: '
#           'square{1 - sum_over_t{alpha_t_i}}). This assumes that the mean sum_over_t should be 1. '
#           "However, that's not true, since the mean of sum_over_t term should be C/L. This "
#           "variable if set to True, causes the term to change to square{C/L - sum_over_t{alpha_t_i}}). "
#           "The default value is True in conformance with the paper.",
#          boolean,
#          False),
#        PD('pLambda', 'Lambda value for alpha penalty',
#           decimal(0),
#           0.0001)  
        )
    def __init__(self, initVals=None):
        dlc.HyperParams.__init__(self, self.proto, initVals)
        
D_RNN = CALSTMParams(
        ## Overrides of default values.
        ## FYI: By convention, all boolean params' default value is True, such
        ##      that the overrides would be easy to spot.
        {
            'att_weighted_gather': True,
            'sum_logloss': True,
            'MeanSumAlphaEquals1': True,
            'output_follow_paper': True
        })
D_RNN.att_layers = tfc.MLPParams({
        # Number of units in all layers of the attention model = D in the paper"s source-code.
        'layers_units': (D_RNN.D,),
        'activation_fn': tf.nn.tanh, # = tanh in the paper's source code
        'tb': D_RNN.tb, # using the global tensorboard params
        'dropout': tfc.DropoutParams({'keep_prob': 0.9})
        })
#D_RNN.output_layers = tfc.MLPParams({
#        ## One layer with num_units = m is added if output_follow_paper == True
#        ## Last layer must have num_units = K because it outputs logits.
#        ## paper has all layers with num_units = m. I've noteiced that they build rectangular MLPs, i.e. not triangular.
#        'layers_units': (D_RNN.m, D_RNN.K),
#        'activation_fn': tf.nn.relu, # paper has it set to relu
#        'tb': D_RNN.tb,
#        'dropout': tfc.DropoutParams({'keep_prob': 0.9})
#        })
D_RNN.decoder_lstm = tfc.RNNParams({
        'B': D_RNN.B,
        'i': D_RNN.m+D_RNN.D, ## size of embedding vector + z_t
         ##'num_units': D_RNN.n ## paper uses a value of 1000
        'layers_units': (1000, 512),
        'dropout': tfc.DropoutParams({'keep_prob': 0.9})
        })

print 'D_RNN = ', D_RNN
