#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
    Hyper-parameters for the model.

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

Created on Fri Jul 21 11:05:51 2017
Tested on python 2.7

@author: Sumeet S Singh
"""

import logging
import numpy as np
import os
import tensorflow as tf
import dl_commons as dlc
import tf_commons as tfc
from dl_commons import (PD, instanceof, integer, integerOrNone, decimal, boolean, equalto, issequenceof,
                        iscallable, iscallableOrNone, LambdaVal, instanceofOrNone)
from tf_commons import (ConvStackParams, ConvLayerParams, MaxpoolParams)

def setLogLevel(logger, level):
    logging_levels = (logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG)
    logger.setLevel(logging_levels[level - 1])

def makeFormatter():
    return logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

def makeLogger(logging_level=3, name='default'):
    logger = logging.Logger(name)
    ch = logging.StreamHandler()
    ch.setFormatter(makeFormatter())
    logger.addHandler(ch)
    setLogLevel(logger, logging_level)
    return logger

def pad_image_shape(shape, padding):
    return (shape[0] + 2*padding, shape[1] + 2*padding) + shape[2:]

def get_image_shape(raw_data_dir_, num_channels, valid_padding):
    standardized_size = np.load(os.path.join(raw_data_dir_, 'padded_image_dim.pkl'))
    return pad_image_shape( (standardized_size['height'], standardized_size['width'], num_channels), valid_padding )

class GlobalParams(dlc.HyperParams):
    """ Common Properties to trickle down. """
    proto = (
        PD('build_image_context', '(boolean): Whether to include convnet as part of the model',
            (0,1,2) ## 0=> do not build convnet, 1=> use vgg16 app from Keras, 2=> build my own
            ),
        PD('image_shape_unframed',
           'Shape of input images. Should be a python sequence.'
           'This is superceded by image_shape which includes an extra padding frame around it',
           issequenceof(int),
           LambdaVal(lambda _,p: (120,1075,3) if (p.build_image_context != 2) else (120,1075,1))
           ## = get_image_shape(raw_data_dir, num_channels, 0)
           ),
        PD('MaxSeqLen',
           "Max sequence length including the end-of-sequence marker token. Is used to "
            "limit the number of decoding steps.",
           integer(151,200),
           151 #get_max_seq_len(data_folder)
           ),
        PD('B',
           '(integer): Size of mini-batch for training, validation and testing graphs/towers. '
           'NOTE: Batch-size for the data-reader is different and set under property "data_reader_B"',
           integer(1),
           96
           ),
        PD('K',
           'Vocabulary size including zero',
           xrange(500,1000),
           LambdaVal(lambda _, d: 557+1 if d.use_ctc_loss else 557) #get_vocab_size(data_folder) + 1 for Blank-Token
           ),
        PD('CTCBlankTokenID', 'ID of the space/blank token',
           integerOrNone(),
           ## By CTC requirement, blank token should be == K-1
           LambdaVal(lambda _, p: (p.K-1) if p.use_ctc_loss else None)
           ),
        PD('SpaceTokenID', 'Space Token ID',
           integer(),
           556
           ),
        PD('NullTokenID',
           'ID of the EOS token == Null Token',
           (0,),
           0
           ),
        PD('StartTokenID',
           'ID of the begin-sequence token. Equal to either 1 or Whitespace.',
           (1, 556), ## 1 or space
           1
           ),
        PD('n',
           "The variable n in the paper. The number of units in the decoder_lstm cell(s). "
           "The paper uses a value of 1000.",
           integer(),
           1000),
        PD('m',
           '(integer): dimensionality of the embedded input vector (Ey / Ex).'
           "Note: For a stacked CALSTM, the upper layers will be fed output of the previous CALSTM, "
           "therefore their input dimensionality will not be equal to the embedding dimensionality, rather "
           " it will be equal to output_size of the previous CALSTM. That's why this value needs to be "
           "appropriately adjusted for upper CALSTM layers.",
           integer(1),
           64
           ),
        PD('H', 'Height of feature-map produced by conv-net. Specific to the dataset image size.', None,
            LambdaVal(lambda _,p: 8 if (p.build_image_context == 2) else 3)
           ),
        PD('W', 'Width of feature-map produced by conv-net. Specific to the dataset image size.', None,
            LambdaVal(lambda _,p: 68 if (p.build_image_context == 2) else 33)
            ),
        PD('L',
           '(integer): number of pixels in an image feature-map = HxW (see paper or model description)',
           integer(1),
           LambdaVal(lambda _, d: d['H'] * d['W'])
           ),
        PD('D',
           '(integer): number of features coming out of the conv-net. Depth/channels of the last conv-net layer.'
           'See paper or model description.',
           integer(1),
           512),
        PD('tb', "Tensorboard Params.",
           instanceof(tfc.TensorboardParams),
           tfc.TensorboardParams()
           ),
        PD('dropout',
           'Dropout parameters if any - global. Absence of this property '
           'signals no dropouts. If this is non-None, then weights regularizer should be None.',
           instanceofOrNone(tfc.DropoutParams)
           ),
        PD('dtype',
           'tensorflow float type for the entire model.',
           (tf.float32, tf.float64),
           tf.float32
           ),
        PD('dtype_np',
           'dtype for the entire model.',
           (np.float32, np.float64),
           np.float32
           ),
        PD('int_type',
           'tensorflow int type for the entire model.',
           (tf.int32, tf.int64),
           tf.int32
           ),
        PD('int_type_np',
           'inttype for the entire model.',
           (np.int32, np.int64),
           np.int32
           ),
        PD('weights_initializer',
              'Tensorflow weights initializer function',
              iscallable(),
              tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32) ## = glorot_uniform
              # tf.contrib.layers.variance_scaling_initializer()
              ),
        PD('biases_initializer',
              'Tensorflow biases initializer function, e.g. tf.zeros_initializer(). ',
              iscallable(),
              tf.zeros_initializer()
              ),
        PD('rLambda',
            'Lambda value (scale) for regularizer.',
            decimal(),
            0.0005
            ),
        PD('weights_regularizer',
              'L1 / L2 norm regularization. If this is non-None then dropout should be None.',
              iscallableOrNone(),
              tf.contrib.layers.l2_regularizer(scale=1.0, scope='L2_Regularizer')
              # tf.contrib.layers.l1_regularizer(scale, scope=None)
              ),
        PD('use_ctc_loss',
            "Whether to train using ctc_loss or cross-entropy/log-loss/log-likelihood. In either case "
            "ctc_loss will be logged.",
            boolean,
            False),
        PD('biases_regularizer',
              'L1 / L2 norm regularization',
              iscallable(noneokay=True), None),
        PD('use_peephole',
            '(boolean): whether to employ peephole connections in the decoder LSTM',
            (True, False),
            True),
        PD('logger', 'Python logger object for logging.',
            instanceof(logging.Logger)
            ),
        )
    def __init__(self, initVals=None):
        dlc.HyperParams.__init__(self, self.proto, initVals)
    def __copy__(self):
        ## Shallow copy
        return self.__class__(self)
    def copy(self, override_vals={}):
        ## Shallow copy
        return self.__class__(self).updated(override_vals)

class CALSTMParams(dlc.HyperParams):
    @staticmethod
    def makeProto(GLOBAL=GlobalParams()):
        """
        Trickle down parameters from GlobalParams down the params-tree. Strictly, this should be done
        inside __init__ because prototype is supposed to be static. However, I decided to do this inside
        proto so that all parameter updates are located in one place.
        """
        GLOBAL = GLOBAL.copy().updated({'dropout': None}) ## No dropout within CALSTM
        return GlobalParams.proto + (
        ### Attention Model Params ###
            PD('att_layers', 'MLP parameters for attention model', instanceof(tfc.MLPParams),
               tfc.MLPParams(GLOBAL).updated({
                    # Number of units in all layers of the attention model = D in the paper"s source-code.
                    'layers_units': (equalto('D', GLOBAL), ),
                    'activation_fn': tf.nn.tanh # = tanh in the paper's source code
                    ## 'dropout': None # Remove dropout in the attention model
                       }).freeze()
                ),
            PD('att_share_weights', 'Whether the attention model should share weights across the "L" image locations or not.'
               'Choosing "True" conforms to the paper resulting in a (D+n,att_1_n) weight matrix. Choosing False will result in a MLP with (L*D+n,att_1_n) weight matrix. ',
               boolean,
               True),
            PD('att_weighted_gather', 'The paper"s source uses an affine transform with trainable weights, to narrow the output of the attention'
               "model from (B,L,dim) to (B,L,1). Its like an embedding matrix."
               "I have an alternative implementation that simply averages the matrix (B,L,dim) to (B,L,1)."
               "Default value however, is True in conformance with the paper's implementation.",
               (True, False),
               True),
        ### Decoder LSTM Params ###
            PD('decoder_lstm', 'Decoder LSTM parameters. At this time only one layer is supported.',
               instanceof(tfc.RNNParams),
               tfc.RNNParams(GLOBAL).updated({
                    'B': equalto('B', GLOBAL),
                    'i': None, ## size of input vector + z_t. Set dynamically.
                     ## paper uses a value of n=1000
                    ## 'layers_units': (equalto('n', GLOBAL),),
                    'layers_units': (equalto('n', GLOBAL), equalto('n', GLOBAL),),
                    ## 'dropout': None # No dropout by default
                    })
                )
        )
    def __init__(self, initVals):
       ## "Note: For a stacked CALSTM, the upper layers will be fed output of the previous CALSTM, "
       ## "therefore their input dimensionality will not be equal to the embedding dimensionality, rather "
       ## " it will be equal to output_size of the previous CALSTM. That's why the value of m needs to be "
       ## "appropriately adjusted for upper CALSTM layers."
        assert initVals['m'] is not None
#        proto = self.proto if GLOBALS is None else self.makeProto(GLOBALS)
        GLOBALS = GlobalParams(initVals)
        dlc.HyperParams.__init__(self, self.makeProto(GLOBALS), initVals)
        self._trickledown(GLOBALS)
    def _trickledown(self, GLOBAL):
        self.decoder_lstm.i = LambdaVal(lambda _, __: self.m + GLOBAL.D)
    def __copy__(self):
        ## Shallow copy
        return self.__class__(self)
    def copy(self, override_vals={}):
        ## Shallow copy
        return self.__class__(self).updated(override_vals)

class Im2LatexModelParams(dlc.HyperParams):

    @staticmethod
    def makeProto(GLOBAL=GlobalParams()):
        return GlobalParams.proto + (
        ### Training Parameters ####
            PD('assert_whole_batch', '(boolean): Disallow batch size that is not integral factor '
                'of the bin-size',
                boolean,
                True
                ),
            PD('squash_input_seq', '(boolean): Remove whitespace from target sequences',
                boolean,
                False
                ),
            PD('input_queue_capacity', 'Capacity of input queue.',
                integer(1),
                LambdaVal(lambda _, d: d.num_gpus * 3)
                ),
            PD('DecodingSlack',
                "Since we ignore blanks/spaces in loss and accuracy measurement, the network is free "
                "to insert blanks into the decoded/predicted sequence. Therefore the predicted sequence "
                "can be arbitrarily long. However, we need to limit the max decoded sequence length. We "
                "do so by determining the extra slack to give to the network - the more slack we give it "
                "presumably that much easier the learning will be. This parameter includes that slack. In "
                "other words, MaxDecodeLen = MaxSeqLen + DecodingSlack",
                integer(0),
                20
                ),
            PD('MaxDecodeLen',
                "See the description for MaxSeqLen and DecodingSlack",
                integer(151),
                LambdaVal(lambda _, p: p.MaxSeqLen + p.DecodingSlack)
                ),
            PD('no_ctc_merge_repeated',
               "(boolean): Negated value of ctc_merge_repeated argubeamsearch_length_penatlyment for ctc operations",
               boolean,
               True),
            PD('ctc_beam_width', 'Beam Width to use for ctc_beamsearch_decoder, which is different from the seq2seq.BeamSearchDecoder',
                integer(1)
                ),
            PD('seq2seq_beam_width', 'Beam Width to use for seq2seq.BeamSearchDecoder, which is different from the ctc_beamsearch_decoder',
                integer(1)
                ),
            PD('beamsearch_length_penalty',
                'length_penalty_weight used by beamsearch decoder. Same as alpha value of length-penalty described in https://arxiv.org/pdf/1609.08144.pdf'
                'In the paper they used a value of alpha in the range [0.6,0.7]. A value of 0 turns length-penalty off.',
                decimal(0.,1.),
                0.6
                ),
            PD('swap_memory',
               'swap_memory option to tf.scan',
                boolean,
                False
                ),
            PD('tf_session_allow_growth',
               'tf ConfigProto.gpu_option_allow_growth. Setting this will allow the gpu memory to be allocated incrementally instead of all at once.',
                boolean,
                False
                ),
            PD('adam_alpha', '(float or None): alpha value (step, learning_rate) of adam optimizer.',
               instanceof(float),
               0.0001 # default in tf.train.AdamOptimizer is 0.001
              ),
            PD('optimizer',
               'tensorflow optimizer function (e.g. AdamOptimizer).',
               instanceof(tf.train.Optimizer),
               ## Value set in self._trickledown
               ## tf.train.AdamOptimizer(learning_rate=hyper.adam_alpha)
               ),
            PD('no_towers', 'Should be always set to False. Indicates code-switch to build without towers which will not work',
               (False,),
               False
               ),
            PD('num_gpus', 'Number of GPUs employed in parallel',
               integer(1)
               ),
            PD('data_reader_B', 'batch_size for the data_reader', 
               integer(1),
               LambdaVal(lambda _, d: d.B * d.num_gpus)
               ),
        ### Embedding Layer ###
            PD('embeddings_initializer', 'Initializer for embedding weights', 
                iscallable(),
               ## tf.contrib.layers.xavier_initializer()
               equalto('weights_initializer')
               ),
        PD('embeddings_regularizer',
              'L1 / L2 norm regularization',
              iscallableOrNone(),
              equalto('weights_regularizer')
              ),
        ### ConvNet Params ###
            PD('CONVNET', 'ConvStackParams for the convent',
                instanceofOrNone(ConvStackParams),
                ## Value is set dynamically inside make_hyper
                ),
            PD('image_frame_width',
                'Width of an extra padding frame around the (possibly already padded) image. This extra padding is used '
                'in order to ensure that there is enough whites-space around the edges of the image, so as to enable VALID padding '
                'in the first conv-net layer without losing any information. The effect of doing this is to simulate SAME padding '
                'but using custom padding values (background color in this case) instead of zeroes (which is what SAME padding would do). '
                'This value should be equal to (kernel_size-1)/2 using kernel_size of the first convolution layer.'
                ,
                integer(),
                LambdaVal(lambda _, p: 0 if (p.build_image_context != 2) else (p.CONVNET.layers[0].kernel_shape[0]-1)/2 )
                ## Dynamically set to = (kernel_size-1)/2 given kernel_size of first conv-net layer
                ),
            PD('image_shape',
                'Shape of input images. Should be a python sequence.'
                '= image_shape_unpadded + image_frame_width around it',
                issequenceof(int),
                LambdaVal(lambda _, p: pad_image_shape(p.image_shape_unframed, p.image_frame_width))
                ## = get_image_shape(raw_data_dir, num_channels, image_frame_width)
                ),
        ### Decoder CALSTM Params ###
            PD('CALSTM_STACK',
               'sequence of CALSTMParams, one for each CALSTM layer in the stack. The paper '
               "has code for more than one layer, but mentions that it is not well-tested. I take that to mean "
               "that the published results are based on one layer alone.",
               issequenceof(CALSTMParams)
               ),
        ### Output MLP
            PD('output_follow_paper',
               '(boolean): Output deep layer uses some funky logic in the paper instead of a straight MLP'
               'Setting this value to True (default) will follow the paper"s logic. Otherwise'
               "a straight MLP will be used.",
               boolean,
               True
               ),
            PD('output_layers',
               "(MLPParams): Parameters for the output MLP. The last layer outputs the logits and therefore "
               "must have num_units = K. If output_follow_paper==True, an additional initial layer is created "
               "with num_units = m and activtion tanh. Note: In the paper all layers have num_units=m",
               instanceof(tfc.MLPParams),
                   tfc.MLPParams(GLOBAL).updated({
                        ## One layer with num_units = m is added if output_follow_paper == True
                        ## Last layer must have num_units = K because it outputs logits.
                        ## paper has all layers with num_units = m. I've noticed that they build rectangular MLPs, i.e. not triangular.
                        'layers_units': (equalto('m',GLOBAL), equalto('K',GLOBAL)),
                        'activation_fn': tf.nn.relu, # paper has it set to relu
                        'op_name': 'yLogits_MLP'
                        }).freeze()
                ),
        ### Initializer MLP ###
            PD('init_model',
               'MLP stack of the init_state model. In addition to the stack specified here, an additional FC '
               "layer will be forked off at the top for each 'c' and 'h' state in the RNN Im2LatexDecoderRNN state."
               "Hence, this is a 'multi-headed' MLP because it has multiple top-layers.",
               instanceof(tfc.MLPParams),
                   tfc.MLPParams(GLOBAL).updated({
                       'layers_units': (equalto('D', GLOBAL),), ## The paper's source sets all hidden units to D
                       ## paper sets hidden activations=relu and final=tanh
                       'activation_fn': tf.nn.relu
                       }).freeze()
               ),
            PD('init_model_final_layers', '',
               instanceof(tfc.FCLayerParams),
                   tfc.FCLayerParams(GLOBAL).updated({
                       ## num_units to be set dynamically
                       ## paper sets hidden activations=relu and final=tanh
                       'activation_fn': tf.nn.tanh
                       }).freeze()
               ),
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
              True),
            PD('pLambda', 'Lambda value for alpha penalty',
               decimal(0),
               0.0005), # default in the paper is 00001
            PD('k', 'Number of top-scoring beams to consider for best-of-k metrics.',
               integer(1),
               5)
        )
    def __init__(self, initVals):
        dlc.HyperParams.__init__(self, self.makeProto(GlobalParams(initVals).freeze()), initVals)
        self._trickledown()
    def _trickledown(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.adam_alpha)
    def __copy__(self):
        ## Shallow copy
        return self.__class__(self)
    def copy(self, override_vals={}):
        ## Shallow copy
        return self.__class__(self).updated(override_vals)

def make_hyper(initVals={}, freeze=True):
    initVals = dlc.Properties(initVals)
    ## initVals.image_frame_width = 1
    globals = GlobalParams(initVals)
    assert (globals.rLambda == 0) or (globals.dropout is None), 'Both dropouts and weights_regularizer are non-None'

    CALSTM_1 = CALSTMParams(initVals.copy().updated({'m':globals.m})).freeze()
    CALSTM_2 = CALSTM_1.copy({'m':CALSTM_1.decoder_lstm.layers_units[-1]})

    if globals.build_image_context != 2:
        CONVNET = None
    else:
        ## Conv and Maxpool architecture lifted from VGG16
        CONVNET = ConvStackParams({
            'op_name': 'Convnet',
            'tb': globals.tb,
            'activation_fn': tf.nn.relu,
            'weights_initializer': globals.weights_initializer,
            'biases_initializer': globals.biases_initializer,
            'weights_regularizer': globals.weights_regularizer,
            'biases_regularizer': globals.biases_regularizer,
            'padding': 'SAME',
            'layers': (
                ConvLayerParams({'output_channels':64, 'kernel_shape':(3,3), 'stride':(1,1), 'padding':'VALID'}),
                ConvLayerParams({'output_channels':64, 'kernel_shape':(3,3), 'stride':(1,1)}),
                MaxpoolParams({'kernel_shape':(2,2), 'stride':(2,2)}),

                ConvLayerParams({'output_channels':128, 'kernel_shape':(3,3), 'stride':(1,1)}),
                ConvLayerParams({'output_channels':128, 'kernel_shape':(3,3), 'stride':(1,1)}),
                MaxpoolParams({'kernel_shape':(2,2), 'stride':(2,2)}),

                ConvLayerParams({'output_channels':256, 'kernel_shape':(3,3), 'stride':(1,1)}),
                ConvLayerParams({'output_channels':256, 'kernel_shape':(3,3), 'stride':(1,1)}),
                ConvLayerParams({'output_channels':256, 'kernel_shape':(3,3), 'stride':(1,1)}),
                MaxpoolParams({'kernel_shape':(2,2), 'stride':(2,2)}),

                ConvLayerParams({'output_channels':512, 'kernel_shape':(3,3), 'stride':(1,1)}),
                ConvLayerParams({'output_channels':512, 'kernel_shape':(3,3), 'stride':(1,1)}),
                ConvLayerParams({'output_channels':512, 'kernel_shape':(3,3), 'stride':(1,1)}),
                MaxpoolParams({'kernel_shape':(2,2), 'stride':(2,2)}),

                ConvLayerParams({'output_channels':512, 'kernel_shape':(3,3), 'stride':(1,1)}),
                ConvLayerParams({'output_channels':512, 'kernel_shape':(3,3), 'stride':(1,1)}),
                ConvLayerParams({'output_channels':512, 'kernel_shape':(3,3), 'stride':(1,1)}),
                # MaxpoolParams({'kernel_shape':(2,2), 'stride':(2,2)}),
            )
        })
    
    HYPER = Im2LatexModelParams(initVals).updated({
        'CALSTM_STACK':(CALSTM_1,CALSTM_2),
        'CONVNET': CONVNET
        })
        
    if freeze:
        HYPER.freeze()

#    print 'Hyper Params = '
#    print HYPER
    return HYPER
