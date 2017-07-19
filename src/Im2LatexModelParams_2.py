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

Created on Fri Jul 14 13:49:04 2017
Tested on python 2.7

@author: Sumeet S Singh
"""
import dl_commons as dlc
import tf_commons as tfc
from dl_commons import PD, instanceof, integer, decimal, boolean, equalto, issequenceof
import tensorflow as tf
from CALSTMParams import CALSTMParams, CALSTM_1


class Im2LatexModelParams(dlc.HyperParams):
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
        PD('m',
           '(integer): dimensionality of the embedded input vector (Ey / Ex)'
           "Note: For a stacked CALSTM, the upper layers will be fed output of the previous CALSTM, "
           "threfore thir input dimensionality will not be equal to the embedding dimensionality, rather "
           "it will be equal to output_size of the previous CALSTM. That's why this value needs to be "
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
        PD('keep_prob', '(decimal): Value between 0.1 and 1.0 indicating the keep_probability of dropout layers.'
           'A value of 1 implies no dropout.',
           decimal(0.1, 1), 
           1.0),
        PD('tb', "Tensorboard Params.",
           instanceof(tfc.TensorboardParams),
           tfc.TensorboardParams()),
        PD('dropout', 
           'Dropout parameters if any - global at the level of Im2LatexModel.',
           instanceof(tfc.DropoutParams, True)),
        PD('dtype',
           'dtype for the entire model.',
           (tf.float32,),
           tf.float32),
        PD('use_ctc_loss', 
           "Whether to train using ctc_loss or cross-entropy/log-loss/log-likelihood. In either case "
           "ctc_loss will be logged.",
           boolean,
           False),
    ### Embedding Layer ###
        PD('embeddings_initializer', 'Initializer for embedding weights', None, 'glorot_uniform'),
        PD('embeddings_initializer_tf', 'Initializer for embedding weights', dlc.iscallable(), 
           tf.contrib.layers.xavier_initializer()),
    ### Decoder LSTM Params ###
        PD('n',
           '(integer): Number of hidden-units of the LSTM cell',
           integer(100,10000),
           1000),
        PD('D_RNN',
           'sequence of Im2LatexDecoderRNNParams, one for each AttentionLSTM layer in the stack. The paper '
           "has code for more than one layer, but mentions that it is not well-tested. I take that to mean "
           "that the published results are based on one layer alone.",
           issequenceof(CALSTMParams)),
    ### Output MLP
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

    ### Initializer MLP ###
        PD('init_model', 
           'MLP stack of the init_state model. In addition to the stack specified here, an additional FC '
           "layer will be forked off at the top for each 'c' and 'h' state in the RNN Im2LatexDecoderRNN state."
           "Hence, this is a 'multi-headed' MLP because it has multiple top-layers.",
           instanceof(tfc.MLPParams)
           ),
        PD('init_layers', 'Number of layers in the initializer MLP', xrange(1,10),
           1),
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
    def __init__(self, initVals=None):
        dlc.HyperParams.__init__(self, self.proto, initVals)
    def __copy__(self):
        ## Shallow copy
        return self.__class__(self)
    
    def copy(self, override_vals=None):
        ## Shallow copy
        return self.__class__(self).updated(override_vals)

HYPER = Im2LatexModelParams(
        ## Overrides of default values.
        ## FYI: By convention, all boolean params' default value is True
        {
            'att_weighted_gather': True,
            'sum_logloss': True,
            'MeanSumAlphaEquals1': True,
            'output_follow_paper': True,
#            'D_RNN': (CALSTM_1, CALSTM_2)
            'D_RNN': (CALSTM_1,)
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
HYPER.init_model = tfc.MLPParams({
                   'layers_units': (HYPER.D,), ## The paper's source sets it to D
                   'dropout': HYPER.dropout,
                   'tb': HYPER.tb,
                   'activation_fn': tf.nn.tanh
                   })


print 'HYPER = ', HYPER

