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
           557 #get_vocab_size(data_folder)
           ),
        PD('n',
           "The variable n in the paper. The number of units in the decoder_lstm cell(s). "
           "The paper uses a value of 1000.",
           integer(),
           1000),
        PD('m',
           '(integer): dimensionality of the embedded input vector (Ey / Ex).'
           "Note: For a stacked CALSTM, the upper layers will be fed output of the previous CALSTM, "
           "threfore thir input dimensionality will not be equal to the embedding dimensionality, rather "
           " it will be equal to output_size of the previous CALSTM. That's why this value needs to be "
           "appropriately adjusted for upper CALSTM layers.", 
           integer(1),
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
    ### Decoder LSTM Params ###
        PD('decoder_lstm', 'Decoder LSTM parameters. At this time only one layer is supported.',
           instanceof(tfc.RNNParams)),

        )
    def __init__(self, initVals=None):
        dlc.HyperParams.__init__(self, self.proto, initVals)

    def __copy__(self):
        ## Shallow copy
        return self.__class__(self)
    
    def copy(self, override_vals=None):
        ## Shallow copy
        return self.__class__(self).updated(override_vals)
        
def update_layer_params(D_RNN):
    D_RNN.att_layers = tfc.MLPParams({
            # Number of units in all layers of the attention model = D in the paper"s source-code.
            'layers_units': (D_RNN.D,),
            'activation_fn': tf.nn.tanh, # = tanh in the paper's source code
            'tb': D_RNN.tb, # using the global tensorboard params
            'dropout': tfc.DropoutParams({'keep_prob': 0.9})
            })
    D_RNN.decoder_lstm = tfc.RNNParams({
            'B': D_RNN.B,
            'i': D_RNN.m+D_RNN.D, ## size of input vector + z_t
             ##'num_units': D_RNN.n ## paper uses a value of 1000
            'layers_units': (1000,),
            'dropout': tfc.DropoutParams({'keep_prob': 0.9})
            })

CALSTM_1 = CALSTMParams(
        ## Overrides of default values.
        ## FYI: By convention, all boolean params' default value is True, such
        ##      that the overrides would be easy to spot.
        {
            'att_weighted_gather': True,
            'sum_logloss': True,
            'MeanSumAlphaEquals1': True,
            'output_follow_paper': True
        })
    
    
update_layer_params(CALSTM_1)
CALSTM_2 = CALSTM_1.copy({'m':CALSTM_1.decoder_lstm.layers_units[-1]})
update_layer_params(CALSTM_2)