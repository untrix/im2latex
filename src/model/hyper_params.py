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
import data_commons as dtc
import dl_commons as dlc
from dl_commons import (instanceof, integer, integerOrNone, decimal, boolean, equalto, issequenceof, issequenceofOrNone,
                        PD, iscallable, iscallableOrNone, LambdaVal, instanceofOrNone, Properties)
from tf_commons import (ConvStackParams, ConvLayerParams, MaxpoolParams, FCLayerParams, MLPParams,
                        DropoutParams, TensorboardParams, RNNParams)

def pad_image_shape(shape, padding):
    return (shape[0] + 2*padding, shape[1] + 2*padding) + shape[2:]

# def get_image_shape(raw_data_dir_, num_channels):
#     standardized_size = np.load(os.path.join(raw_data_dir_, 'padded_image_dim.pkl'))
#     return (standardized_size['height'], standardized_size['width'], num_channels)
#
# def get_vocab_size(raw_data_dir):
#     id2word = np.load(os.path.join(raw_data_dir, 'dict_id2word.pkl'))
#     return max(id2word.keys()) + 1
#

class GlobalParams(dlc.HyperParams):
    """ Common Properties to trickle down. """
    proto = (
        ## Data-set Properties ##
        PD(
            'raw_data_dir',
            'Filesystem path of raw_data_folder from where the pre-processed data is stored.',
            dlc.instanceof(str)
        ),
        PD(
            'image_shape_unframed',
            'Shape of input images. Should be a python sequence.'
            'This is superseded by image_shape which optionally includes an extra padding frame around the input image'
            'Value is loaded from the dataset and is not configurable.',
            issequenceof(int),
            # Set dynamically based on dataset.
        ),
        PD(
            'MaxSeqLen',
            "Max sequence length including the end-of-sequence marker token. Is used to "
            "limit the number of decoding steps. Value is loaded from the dataset and is not configurable.",
            integer(151, 200),
            # Set dynamically based on dataset.
        ),
        PD(
            'K',
            'Vocabulary size including zero. Value is loaded from the dataset and is not configurable.',
            (358, 557, 339),
            # Set dynamically based on dataset.
            # LambdaVal(lambda _, d: 557+1 if d.use_ctc_loss else 557) #get_vocab_size(data_folder) + 1 for Blank-Token
        ),
        PD(
            'CTCBlankTokenID',
            'ID of the space/blank token. By tf.nn.ctc requirement, blank token should be == K-1,'
            'Value is loaded from the dataset and is not configurable.',
            integerOrNone(),
            # Set dynamically based on dataset.
        ),
        PD(
            'SpaceTokenID',
            'Space Token ID if present in the dataset.',
            integerOrNone(),
            # Set dynamically based on dataset.
        ),
        PD(
            'NullTokenID',
            'ID of the EOS token == Null Token. Must be zero. Its value is loaded from the dataset and is not configurable.',
            (0,),
            # Set dynamically based on dataset.
        ),
        PD(
            'StartTokenID',
            'ID of the begin-sequence token. The value is loaded from the dataset and is not configurable.',
            (1,),
            # Set dynamically based on dataset.
        ),
        ###############################
        PD(
            'build_image_context',
            """
            (enum): Type of decoder conv-net model to use:
            0 => Do not build decoder conv-net. Use pre-generated image features instead.
            1 => Use VGG16 Conv-Net model (imported from Keras).
            2 => Use a custom conv-net (defined in make_hyper)
            """,
            (0, 1, 2)
        ),
        PD(
            'build_scanning_RNN', '(boolean): Whether to build a regular RNN or a scanning RNN',
            boolean,
        ),
        PD(
            'B',
            '(integer): Size of mini-batch for training, validation and testing graphs/towers. '
            'NOTE: Batch-size for the data-reader is different and set under property "data_reader_B"',
            integer(1),
        ),
        PD(
            'n',
            "The variable n in the paper. The number of units in the decoder_lstm cell(s). "
            "The paper uses a value of 1000.",
            (1000, 1500),
            1500
        ),
        PD(
            'm',
            '(integer): dimensionality of the embedded input vector (Ex).'
            "Note: For a stacked CALSTM, the upper layers will be fed output of the previous CALSTM, "
            "therefore their input dimensionality will not be equal to the embedding dimensionality, rather "
            " it will be equal to output_size of the previous CALSTM. That's why this value needs to be "
            "appropriately adjusted for upper CALSTM layers.",
            (64, 3),
            LambdaVal(lambda _, p: 3 if p.build_scanning_RNN else 64)
        ),
        PD(
            'REGROUP_IMAGE',
            """
            Specifies how the image feature vectors should be grouped together 
            along Height and Width axes. For e.g. if the original dimension of the context feature map was (3,33,512) 
            - i.e. original H=3, original W=33 and D=512- and if REGROUP_IMAGE was (3,3) then the new 
            context-map would have shape (1, 11, 512*3*3) resulting in H=1, W=33, D=4608 and L=33.
            A None value implies no regrouping.
            """,
            issequenceofOrNone(int),
        ),
        PD('image_size',
           'Older image-size was "small". Newer one is "big"',
           ('small', 'big'),
           'big'
           ),
        PD(
            'H0', 'Height of feature-map produced by conv-net. Specific to the dataset image size.',
            integer(1),
            LambdaVal(lambda _, p: 4 if (p.image_size == 'big') else 3)
            # LambdaVal(lambda _, p: 8 if (p.build_image_context == 2) else (4 if p.dataset == 3 else 3))
        ),
        PD(
            'W0', 'Width of feature-map produced by conv-net. Specific to the dataset image size.',
            integer(1),
            LambdaVal(lambda _, p: 34 if (p.image_size == 'big') else 33)
            # LambdaVal(lambda _, p: 68 if (p.build_image_context == 2) else (34 if p.dataset == 3 else 33))
        ),
        PD(
            'L0',
            '(integer): number of pixels in an image feature-map coming out of conv-net = H0xW0 (see paper or model description)',
            integer(1),
            LambdaVal(lambda _, p: p.H0 * p.W0)
        ),
        PD(
            'D0',
            '(integer): number of features coming out of the conv-net. Depth/channels of the last conv-net layer.'
            'See paper or model description.',
            integer(1),
            512),
        PD(
            'H', 'Height of feature-map produced fed to the decoder.',
            integer(1),
            LambdaVal(lambda _, p: p.H0 if (p.REGROUP_IMAGE is None) else p.H0 / p.REGROUP_IMAGE[0])
        ),
        PD(
            'W', 'Width of feature-map fed to the decoder.',
            integer(1),
            LambdaVal(lambda _, p: p.W0 if (p.REGROUP_IMAGE is None) else p.W0 / p.REGROUP_IMAGE[1])
        ),
        PD(
            'L',
            '(integer): number of pixels in an image feature-map fed to the decoder = HxW (see paper or model description)',
            integer(1),
            LambdaVal(lambda _, p: p.H * p.W)
        ),
        PD(
            'D',
            '(integer): number of image-features fed to the decoder. Depth/channels of the last conv-net layer.'
            'See paper or model description.',
            integer(1),
            LambdaVal(
                lambda _, p: p.D0 if (p.REGROUP_IMAGE is None) else p.D0 * p.REGROUP_IMAGE[0] * p.REGROUP_IMAGE[1])),
        PD(
            'tb', "Tensorboard Params.",
            instanceof(TensorboardParams),
        ),
        PD(
            'dropout',
            'Dropout parameters if any - global. Absence of this property '
            'signals no dropouts. If this is non-None, then weights regularizer should be None.',
            instanceofOrNone(DropoutParams)
        ),
        PD(
            'dtype',
            'tensorflow float type for the entire model.',
            (tf.float32, tf.float64),
            tf.float32
        ),
        PD(
            'dtype_np',
            'dtype for the entire model.',
            (np.float32, np.float64),
            np.float32
        ),
        PD(
            'int_type',
            'tensorflow int type for the entire model.',
            (tf.int32, tf.int64),
            tf.int32
        ),
        PD(
            'int_type_np',
            'numpy inttype for the entire model.',
            (np.int32, np.int64),
            np.int32
        ),
        PD(
            'weights_initializer',
            'Tensorflow weights initializer function',
            iscallable(),
            tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32)  ## = glorot_uniform
            # tf.contrib.layers.variance_scaling_initializer()
        ),
        PD(
            'biases_initializer',
            'Tensorflow biases initializer function, e.g. tf.zeros_initializer(). ',
            iscallable(),
            tf.zeros_initializer()
        ),
        PD(
            'rLambda',
            'Lambda value (scale) for regularizer.',
            decimal(),
        ),
        PD(
            'weights_regularizer',
            'L1 / L2 norm regularization. If this is non-None then dropout should be None.',
            iscallableOrNone(),
            # tf.contrib.layers.l2_regularizer(scale=1.0, scope='L2_Regularizer')
            # tf.contrib.layers.l1_regularizer(scale=1.0, scope="L1_Regularizer")
        ),
        PD(
            'use_ctc_loss',
            "Whether to train using ctc_loss or cross-entropy/log-loss/log-likelihood. In either case "
            "ctc_loss will be logged. Also, use_ctc_loss must be turned on if building scanning-RNN.",
            boolean,
            LambdaVal(lambda _, p: p.build_scanning_RNN)
        ),
        PD(
            'biases_regularizer',
            'L1 / L2 norm regularization',
            iscallable(noneokay=True),
            None),
        PD(
            'use_peephole',
            '(boolean): whether to employ peephole connections in the decoder LSTM',
            (True, False),
            True),
        PD(
            'logger', 'Python logger object for logging.',
            instanceof(logging.Logger)
        ),
    )

    def __init__(self, initVals=None):
        dlc.HyperParams.__init__(self, self.proto, initVals)
        self._trickledown()

    def _trickledown(self):

        data_props = np.load(os.path.join(self.raw_data_dir, 'data_props.pkl'))
        num_channels = 1 if (self.build_image_context == 2) else 3
        self.image_shape_unframed = (data_props['padded_image_dim']['height'], data_props['padded_image_dim']['width'], num_channels)

        self.MaxSeqLen = data_props['MaxSeqLen']
        self.SpaceTokenID = data_props['SpaceTokenID']
        self.NullTokenID = data_props['NullTokenID']
        self.StartTokenID = data_props['StartTokenID']
        if self.SpaceTokenID is not None:
            if False: # self.use_ctc_loss:
                self.K = data_props['K'] + 1
                self.CTCBlankTokenID = self.K - 1
            else:
                self.K = data_props['K']
                self.CTCBlankTokenID = None
        else:
            self.K = data_props['K'] + 1
            self.CTCBlankTokenID = self.K - 1

    def __copy__(self):
        ## Shallow copy
        return self.__class__(self)
    def copy(self, override_vals={}):
        ## Shallow copy
        return self.__class__(self).updated(override_vals)

class CALSTMParams(dlc.HyperParams):
    proto = GlobalParams.proto + (
        ### Attention Model Params ###
        PD(
            'att_layers', 'MLP or Convnet parameters for attention model',
            dlc.either(instanceof(MLPParams), instanceof(ConvStackParams)),
            ## Set dynamically in self._trickledown()
        ),
        PD(
            'att_model',
            'Whether the attention model should share weights across the "L" image locations - i.e. perform a 1x1 convolution - or use a straight MLP.'
            'Choosing "MLP_shared" conforms to the paper resulting in a (D+n,att_1_n) kernel (similar to 1x1 convolution but implemented as an MLPStack).'
            'Choosing "1x1_conv" results in a 1x1 convolution-stack implementation instead of shared-MLP as above. The model should be identical to that obtained'
            'by setting att_model=MLP_shared (plus equivalent num_units and same activation functions)'
            'Choosing "MLP_full" will result in a standard MLP whose input is the entire flattened image feature vector concatenated with h(t-1), resulting '
            'in a vector size (L*D + n) and a weight matrix of size (L*D+n,att_1_n). That is, the kernel will receive input from all the "L" '
            "locations of image feature-map (in addition to h(t-1)) as against only one location in the MLP_shared or 1x1_conv cases.",
            ('MLP_shared', '1x1_conv', 'MLP_full'),
        ),
        PD(
            'att_modulator',
            'A neural-network whose scalar output modulates the soft context signal "z_t" in comparison to the embedded-label signal fed into the decoder LSTM.'
            'Serves as an "equalizer knob" that either amplifies or attenuates the context input signal versus the embedded-label input signal to the LSTM.'
            'In the paper this scalar-value is called "beta" and in their code it is called the "selector". They implemented it as the output of a sigmoid and hence a value between 0 and 1.'
            "We'll use a generic MLP stack with either a sigmoid activation in the last layer or a relu (in wchich case the modulator can take values greater than 1). "
            "Input to this MLP is the h(t-1) - the previous LSTM output.",
            instanceof(MLPParams),
        ),
        PD(
            'build_att_modulator',
            '(boolean): Turns on/off the att_modulator function.',
            boolean
        ),
        PD(
            'feed_clock_to_att',
            'Turns on/off feeding scanning clock into the attention model. Only applicable for scanning-RNN.',
            boolean,
            LambdaVal(lambda _, p: True if p.build_scanning_RNN else None)
        ),
        PD(
            'no_clock_to_lstm',
            'Turns on/off feeding scanning clock into the decoder-LSTM. Only applicable for scanning-RNN.',
            boolean,
            LambdaVal(lambda _, p: True if p.build_scanning_RNN else None)
        ),
        ### Decoder LSTM Params ###
        PD(
            'decoder_lstm',
            'Decoder LSTM parameters. Multiple LSTM layers are supported.',
            instanceof(RNNParams),
            ## Set dynamically in self._trickledown()
        )
    )

    def __init__(self, initVals):
       ## "Note: For a stacked CALSTM, the upper layers will be fed output of the previous CALSTM, "
       ## "therefore their input dimensionality will not be equal to the embedding dimensionality, rather "
       ## " it will be equal to output_size of the previous CALSTM. That's why the value of m needs to be "
       ## "appropriately adjusted for upper CALSTM layers."
        assert initVals['m'] is not None
        dlc.HyperParams.__init__(self, self.proto, initVals, seal=False)
        ## No dropout within CALSTM
        self.dropout = None
        self._trickledown()

    def _trickledown(self):
        """
        Trickle down changes onto parameters in sub-tree(s) if needed.
        (For same level dependencies use LambdaFunctions instead.)
        Call at the end of __init__ and end of update.
        """

        #### Attention Model ####
        if self.att_model == '1x1_conv':
            self.att_layers = ConvStackParams({
                'op_name': '1x1Conv',
                'tb': self.tb,
                # 'activation_fn': tf.nn.relu,
                # 'weights_initializer': self.weights_initializer,
                # 'biases_initializer': self.biases_initializer,
                # 'weights_regularizer': self.weights_regularizer,
                # 'biases_regularizer': self.biases_regularizer,
                # 'padding': 'VALID',
                'layers': (
                    ## TODO: Try with relu activation
                    ConvLayerParams(self).updated({'output_channels':self.D, 'kernel_shape':(1,1), 'stride':(1,1), 'padding':'VALID', 'activation_fn': tf.nn.tanh}).freeze(),
                    ConvLayerParams(self).updated({'output_channels':self.D, 'kernel_shape':(1,1), 'stride':(1,1), 'padding':'VALID', 'activation_fn': tf.nn.tanh}).freeze(),
                    ConvLayerParams(self).updated({'output_channels':1,      'kernel_shape':(1,1), 'stride':(1,1), 'padding':'VALID', 'activation_fn': None}).freeze()
                    )
                }).freeze()
            assert self.att_layers.layers[-1].output_channels == 1, 'num output_channels of the final layer of the att_convnet should equal 1'
            assert self.att_layers.layers[-1].activation_fn == None ## Softmax activation will be added after squeezing dims

        elif self.att_model == 'MLP_shared':
            self.att_layers = MLPParams(self).updated({
                'op_name': 'MLP_shared',
                # Number of units in all layers of the attention model = D in the Show and Tell paper"s source-code.
                'layers': (
                    FCLayerParams(self).updated({'num_units': self.D, 'activation_fn': tf.nn.tanh, 'dropout': self.dropout}).freeze(),
                    FCLayerParams(self).updated({'num_units': self.D, 'activation_fn': tf.nn.tanh, 'dropout': self.dropout}).freeze(),
                    FCLayerParams(self).updated({'num_units': 1,      'activation_fn': None, 'dropout': None}).freeze(),
                    )
                }).freeze()
            assert self.att_layers.layers[-1].num_units == 1, 'num_units of the final layer of the att_kernel should equal 1'
            assert self.att_layers.layers[-1].activation_fn == None ## Softmax activation will be added after squeezing dims
            assert self.att_layers.layers[-1].dropout == None ## No droput in the softmax layer

        elif self.att_model == 'MLP_full':
            self.att_layers = MLPParams(self).updated({
                'op_name': 'MLP_full',
                'layers': (
                    FCLayerParams(self).updated({'num_units': max(self.L, 256), 'activation_fn': tf.nn.tanh, 'dropout': self.dropout}).freeze(),
                    FCLayerParams(self).updated({'num_units': max(self.L, 128), 'activation_fn': tf.nn.tanh, 'dropout': self.dropout}).freeze(),
                    FCLayerParams(self).updated({'num_units': self.L, 'activation_fn': None,       'dropout': None}).freeze(),
                    )
                }).freeze()
            assert self.att_layers.layers[-1].num_units == self.L, 'num_units of the final layer of the att_MLP should equal L(%d)'%self.L
            assert self.att_layers.layers[-1].activation_fn == None ## softmax activation will be added in code for consistency
            assert self.att_layers.layers[-1].dropout == None ## No droput before/after softmax activation

        #### Attention Modulator ####
        if self.build_att_modulator:
            self.att_modulator = MLPParams(self).updated({
                'op_name': 'beta_MLP',
                ## paper's code uses 1 layer only. The last-layer must have only one neuron os that the output is scalar.
                'layers': (
                    FCLayerParams(self).updated({'num_units': 1, 'activation_fn': tf.nn.sigmoid, 'dropout': None}).freeze(),
                    )
                }).freeze()

        #### LSTM-Stack ####
        self.decoder_lstm = RNNParams(self).updated({
            'B': self.B,
            'i': self.D if (self.build_scanning_RNN and self.no_clock_to_lstm) else self.m + self.D,  # size of Ex_t + size of z_t.
                # show-and-tell paper uses a value of n=1000
            'layers_units': (self.n, self.n),
            ## 'dropout': None # No dropout in LSTM
            }).freeze()

        self.seal()

    def __copy__(self):
        ## Shallow copy
        return self.__class__(self)
    def copy(self, override_vals={}):
        ## Shallow copy
        return self.__class__(self).updated(override_vals)

class Im2LatexModelParams(dlc.HyperParams):
    proto = GlobalParams.proto + (
        ### Training Parameters ####
        PD(
            'assert_whole_batch', '(boolean): Disallow batch size that is not integral factor '
                                  'of the bin-size',
            boolean,
        ),
        PD(
            'squash_input_seq', '(boolean): Remove whitespace from target sequences',
            boolean,
        ),
        PD(
            'input_queue_capacity', 'Capacity of input queue.',
            integer(1),
            LambdaVal(lambda _, d: d.num_towers * 3)
        ),
        PD(
            'DecodingSlack',
            "Since we ignore blanks/spaces in loss and accuracy measurement, the network is free "
            "to insert blanks into the decoded/predicted sequence. Therefore the predicted sequence "
            "can be arbitrarily long. However, we need to limit the max decoded sequence length. We "
            "do so by determining the extra slack to give to the network - the more slack we give it "
            "presumably that much easier the learning will be. This parameter includes that slack. In "
            "other words, MaxDecodeLen = MaxSeqLen + DecodingSlack",
            integer(0),
            20
        ),
        PD(
            'MaxDecodeLen',
            "See the description for MaxSeqLen and DecodingSlack",
            integer(151),
            LambdaVal(lambda _, p: p.MaxSeqLen + p.DecodingSlack)
        ),
        PD(
            'SFactor',
            'Applicable to Scanning LSTM only: Multiplier to derive MaxS from MaxSeqLen',
            decimal(1.0),
            LambdaVal(lambda _, p: 1.5 if p.build_scanning_RNN else None)
        ),
        PD(
            'MaxS', 'Applicable to Scanning LSTM only: Max value of S for the given data-set',
            integer(1),
            LambdaVal(lambda _, p: int(p.MaxSeqLen * p.SFactor) if p.build_scanning_RNN else None)
        ),
        PD(
            'no_ctc_merge_repeated',
            "(boolean): Negated value of ctc_merge_repeated beamsearch_length_penatly for ctc operations",
            boolean,
            True),
        PD(
            'ctc_beam_width',
            'Beam Width to use for ctc_beamsearch_decoder, which is different from the seq2seq.BeamSearchDecoder',
            integer(1)
        ),
        PD(
            'seq2seq_beam_width',
            'Beam Width to use for seq2seq.BeamSearchDecoder, which is different from the ctc_beamsearch_decoder',
            integer(1)
        ),
        PD(
            'beamsearch_length_penalty',
            'length_penalty_weight used by beamsearch decoder. Same as alpha value of length-penalty described in https://arxiv.org/pdf/1609.08144.pdf'
            'In the paper they used a value of alpha in the range [0.6,0.7]. A value of 0 turns length-penalty off.',
            decimal(0., 1.),
            # 0.6
        ),
        PD(
            'swap_memory',
            'swap_memory option to tf.scan',
            boolean,
            False
        ),
        PD(
            'tf_session_allow_growth',
            'tf ConfigProto.gpu_option_allow_growth. Setting this will allow the gpu memory to be allocated incrementally instead of all at once.',
            boolean,
            # False
        ),
        PD(
            'adam_alpha', '(float or None): alpha value (step, learning_rate) of adam optimizer.',
            instanceof(float),
            # 0.0001 # default in tf.train.AdamOptimizer is 0.001
        ),
        PD(
            'adam_beta1',
            'beta1 value of adam-optimizer. If undefined here, the default in tf.train.AdamOptimizer is is 0.9.',
            decimal(0., 1.),
        ),
        PD(
            'adam_beta2',
            'beta2 value of adam-optimizer. If undefined here, the default in tf.train.AdamOptimizer is is 0.999.',
            decimal(0., 1.),
            # 0.9
        ),
        PD(
            'optimizer',
            'tensorflow optimizer function (e.g. AdamOptimizer).',
            ('adam',),
            'adam'
        ),
        PD(
            'no_towers',
            'Should be always set to False. Indicates code-switch to build without towers which will not work',
            (False,),
            False
        ),
        PD(
            'num_gpus', 'Number of GPUs employed in parallel',
            integer(1)
        ),
        PD(
            'towers_per_gpu',
            """
            Number of towers per GPU running concurrently. Multiple towers per gpu are 
            needed in order to circumvent OOM errors.""",
            integer(1)
        ),
        PD(
            'num_towers',
            """
            Number of towers per GPU running concurrently. Multiple towers per gpu are 
            needed in order to circumvent OOM errors.""",
            integer(1),
            LambdaVal(lambda _, p: p.num_gpus * p.towers_per_gpu)
        ),
        PD(
            'data_reader_B', 'batch_size for the data_reader',
            integer(1),
            LambdaVal(lambda _, d: d.B * d.num_towers)
        ),
        ### Embedding Layer ###
        PD(
            'embeddings_initializer', 'Initializer for embedding weights',
            iscallable(),
            ## tf.contrib.layers.xavier_initializer()
            equalto('weights_initializer')
        ),
        PD(
            'embeddings_regularizer',
            'L1 / L2 norm regularization',
            iscallableOrNone(),
            equalto('weights_regularizer')
        ),
        ### ConvNet Params ###
        PD(
            'CONVNET', 'ConvStackParams for the convent',
            instanceofOrNone(ConvStackParams),
            ## Value is set dynamically inside make_hyper
        ),
        PD(
            'image_frame_width',
            'Width of an extra padding frame around the (possibly already padded) image. This extra padding is used '
            'in order to ensure that there is enough whites-space around the edges of the image, so as to enable VALID padding '
            'in the first conv-net layer without losing any information. The effect of doing this is to simulate SAME padding '
            'but using custom padding values (background color in this case) instead of zeroes (which is what SAME padding would do). '
            'This value should be equal to (kernel_size)//2 using kernel_size of the first convolution layer.'
            ,
            integer(),
            LambdaVal(lambda _, p: 0 if (p.build_image_context != 2) else (p.CONVNET.layers[0].kernel_shape[0]) // 2)
            ## Dynamically set to = (kernel_size-1)/2 given kernel_size of first conv-net layer
        ),
        PD(
            'image_shape',
            'Shape of input images. Should be a python sequence.'
            '= image_shape_unpadded + image_frame_width around it',
            issequenceof(int),
            LambdaVal(lambda _, p: pad_image_shape(p.image_shape_unframed, p.image_frame_width))
            ## = get_image_shape(raw_data_folder, num_channels, image_frame_width)
        ),
        ### Decoder CALSTM Params ###
        PD(
            'CALSTM_STACK',
            'sequence of CALSTMParams, one for each CALSTM layer in the stack. The paper '
            "has code for more than one layer, but mentions that it is not well-tested. I take that to mean "
            "that the published results are based on one layer alone.",
            issequenceof(CALSTMParams)
        ),
        ### Output MLP
        PD(
            'output_reuse_embeddings',
            '(boolean): Output layer in the paper has a special first layer which considers embedding weights as part of the first-layer weight matrix.'
            'Setting this value to True (default) will follow the paper"s logic. Otherwise'
            "a straight MLP will be used wherein all inputs (including Ey(t-1)) are first concatenated and fed into an MLP."
            "Including the softmax layer, the paper uses a minimum of 2 layers.",
            boolean,
            # True
        ),
        PD(
            'outputMLP_skip_connections',
            '(boolean): Applicable only when output_reuse_embeddings==False. Setting this value to False will cause'
            'image context (z_t) and sequence input (Ex_t) to not be fed into the output MLP. If True (Default), the'
            'output MLP receives a concatenation of Ex_t, h_t and z_t as input. If set to False, only h_t is input.',
            boolean,
            True
        ),
        PD(
            'output_first_layer', "Some params of first layer of output MLP if output_reuse_embeddings==True",
            instanceof(Properties)
            ## Value set dynamically inside self._trickledown() iff output_reuse_embeddings==True
        ),
        PD(
            'output_layers',
            "(MLPParams): Parameters for the output MLP. The last layer outputs the logits and therefore "
            "must have num_units = K. If output_reuse_embeddings==True, an additional initial layer is created "
            "with num_units = m and activtion tanh. Therefore the min number of layers is 2 in that case. "
            "Note: In the paper all layers have num_units=m except the last(softmax) layer.",
            instanceof(MLPParams),
            ## Value set dynamically inside self._trickledown()
        ),
        ### Initializer MLP ###
        PD(
            'build_init_model',
            """ 
            Boolean parameter specifying whether or not to build the LSTM init_state model. If set to False zero-state
            will be used for init-state, otherwise a init-state model will be created based on other init_model_*
            params.
            """,
            boolean
        ),
        PD(
            'init_model_input_transform',
            """
            Transform to apply to the image-context input to the init model. Only applies if build_init_model == True.
            'mean' implies take a mean across the 'L' image-locations and produce an input of size (batchsize, D).
            'full' implies take in all the 'L' features and produce an input tensor of shape (batchsize, L*D).
                Note that with this option the # of parameters in the first layer will go up by a factor of L i.e.
                around 100x.
            """,
            ('mean', 'full')
        ),
        PD(
            'init_model_hidden',
            'MLP stack for hidden layers of the init_state model. In addition to the stack specified here, an additional FC '
            "layer will be forked off at the top for each 'c' and 'h' state in the RNN Im2LatexDecoderRNN state."
            "Hence, this is a 'multi-headed' MLP because it has multiple top-layers."
            "By default their implementation has num_hidden_layers==0 (i.e. n_layers_init==1).",
            instanceof(MLPParams),
            ## Value set dynamically inside self._trickledown()
        ),
        PD(
            'init_model_final_layers', '',
            instanceof(FCLayerParams),
            ## Value set dynamically inside self._trickledown()
        ),
        ### Loss / Cost Layer ###decoder_lstm
        PD(
            'sum_logloss',
            'Whether to normalize log-loss per sample as in standard log perplexity '
            'calculation or whether to just sum up log-losses as in the paper. Defaults'
            'to True in conformance with the paper.',
            boolean,
            # True
        ),
        PD(
            'MeanSumAlphaEquals1',
            '(boolean): When calculating the alpha penalty, the paper uses the term: '
            'square{1 - sum_over_t{alpha_t_i}}). This assumes that the mean sum_over_t should be 1. '
            "However, that's not true, since the mean of sum_over_t term should be C/L. This "
            "variable if set to True, causes the term to change to square{C/L - sum_over_t{alpha_t_i}}). "
            "The default value is True in conformance with the paper.",
            boolean,
            # True
        ),
        PD(
            'pLambda',
            'Lambda value for alpha penalty, Setting this to zero turns off alpha_penalty.',
            (0.0,  0.0005, 0.005, 0.0001, 0.05),
            # LambdaVal(lambda _, p: 0.005 if p.build_scanning_RNN else 0.000)
        ),  # default in the show-and-tell paper is .00001?
        PD(
            'target_aae',
            """
            Target mean_norm_AAE value to shoot for. Varies with data-set. Value discovered by experimentation.
            """,
            (0., 51.42, 51.79),
            # LambdaVal(lambda _, p: None if (p.pLambda == 0) else 51.42)
        ),
        PD(
            'target_ase',
            """
            Target mean_norm_ASE value to shoot for. Varies with data-set. Value discovered by experimentation.
            """,
            (0.0, 5.27, 5.35, 10.0),
            # LambdaVal(lambda _, p: None if (p.pLambda == 0) else (10.0 if p.build_scanning_RNN else 5.27))
            LambdaVal(lambda _, p: None if (p.pLambda == 0) else 5.27)
        ),
        PD(
            'k', 'Number of top-scoring beams to consider for best-of-k metrics.',
            integer(1),
            # Value specified in run.py
        )
    )

    def __init__(self, initVals):
        dlc.HyperParams.__init__(self, self.proto, initVals, seal=False)
        self._trickledown()

    def _trickledown(self):
        """
        Trickle changes down to dependant parameters in sub-tree(s).
        (For same level dependencies use LambdaFunctions instead.)
        Call at the end of __init__ and end of update.
        """
        ######## Output Model ########
        if self.output_reuse_embeddings:
            assert not self.build_scanning_RNN, 'Scanning RNN cannot reuse-embeddings because there are no embeddings'
            self.output_first_layer = FCLayerParams(self).updated({
                'num_units': self.m,
                'activation_fn': tf.nn.tanh,  # Shouldn't this be None?
                # dropout imported from outer scope
                }).freeze()

            self.output_layers = MLPParams(self).updated({
                # One layer with num_units = m is added if output_reuse_embeddings == True
                'op_name': 'yLogits_MLP',
                # dropout imported from outer scope
                'layers': (
                    ## paper has activation set to relu for all but the softmax layer
                    ## paper has all hidden layers with num_units = m.
                    # TODO: num_units should probably be self.K otherwise the model is a reverse pyramid
                    FCLayerParams(self).updated({'num_units': 64, 'activation_fn':tf.nn.relu}).freeze(),
                    ## Last layer must have num_units = K and activation_fn=None because it outputs logits.
                    FCLayerParams(self).updated({'num_units': self.K, 'activation_fn': None, 'dropout': None}).freeze(),
                    )
                }).freeze()
        else:
            self.output_layers = MLPParams(self).updated({
                'op_name': 'yLogits_MLP',
                'layers': (
                    # paper has activation set to relu for all but the softmax layer
                    # paper has all hidden layers with num_units = m.
                    FCLayerParams(self).updated({'num_units': 358, 'activation_fn':tf.nn.tanh}).freeze(),
                    FCLayerParams(self).updated({'num_units': 358, 'activation_fn':tf.nn.tanh}).freeze(),
                    # Last layer must have num_units = K and activation_fn=None because it outputs logits.
                    FCLayerParams(self).updated({'num_units': self.K, 'activation_fn': None, 'dropout': None}).freeze(),
                    )
                }).freeze()

        assert self.output_layers.layers[-2].num_units >= self.K
        assert self.output_layers.layers[-1].activation_fn == None, 'The last layer must have linear activation because softmax is added later (since we need logits for efficient cross-entropy calculation)'
        if (not self.output_reuse_embeddings):
            assert len(self.output_layers.layers) >= 2, "Need one hidden layer at least to match the paper's complexity."

        ######## Init Model ########
        if self.build_init_model:
            # Note: There are no hidden init layers by default in the Show&Tell paper
            self.init_model_hidden = MLPParams(self).updated({
                'layers': (
                    # Show&Tell paper sets hidden activations=relu
                    # The Show&Tell paper's source sets all hidden units to D
                    FCLayerParams(self).updated({'num_units': min(self.D, 100), 'activation_fn': tf.nn.tanh}).freeze(),
                )
                }).freeze()

            self.init_model_final_layers = FCLayerParams(self).updated({
                # Show&Tell paper sets final=tanh
                'activation_fn': tf.nn.tanh,
                'dropout': None
                }).freeze()

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
    initVals.update(globals)

    # assert (globals.rLambda == 0) or (globals.dropout is None), 'Both dropouts and weights_regularizer are non-None'

    CALSTM_1 = CALSTMParams(initVals).freeze()
    # CALSTM_2 = CALSTM_1.copy({'m':CALSTM_1.decoder_lstm.layers_units[-1]}).freeze()
    # CALSTM_2 = CALSTMParams(initVals.copy().updated({'m':CALSTM_1.decoder_lstm.layers_units[-1]})).freeze()

    if globals.build_image_context != 2:
        CONVNET = None
    else:
        convnet_common = {
            'weights_initializer': globals.weights_initializer,
            'biases_initializer': globals.biases_initializer,
            'weights_regularizer': globals.weights_regularizer,
            'biases_regularizer': globals.biases_regularizer,
            'activation_fn': tf.nn.tanh,  # vgg16 has tf.nn.relu
            'padding': 'SAME',
        }

        # CONVNET = ConvStackParams({
        #     'op_name': 'Convnet',
        #     'tb': globals.tb,
        #     'layers': (
        #         ConvLayerParams(convnet_common).updated({'output_channels': 256, 'kernel_shape':(3,3), 'stride':(1,1), 'padding':'VALID'}).freeze(),
        #         # ConvLayerParams(convnet_common).updated({'output_channels': 512, 'kernel_shape':(3,3), 'stride':(1,1)}).freeze(),
        #         MaxpoolParams(convnet_common).updated({'kernel_shape':(2,2), 'stride':(2,2)}).freeze(),
        #
        #         ConvLayerParams(convnet_common).updated({'output_channels':128, 'kernel_shape':(3,3), 'stride':(1,1)}).freeze(),
        #         MaxpoolParams(convnet_common).updated({'kernel_shape':(2,2), 'stride':(2,2)}).freeze(),
        #
        #         ConvLayerParams(convnet_common).updated({'output_channels':128, 'kernel_shape':(3,3), 'stride':(1,1)}).freeze(),
        #         MaxpoolParams(convnet_common).updated({'kernel_shape':(2,2), 'stride':(2,2)}).freeze(),
        #
        #         ConvLayerParams(convnet_common).updated({'output_channels':128, 'kernel_shape':(3,3), 'stride':(1,1)}).freeze(),
        #         MaxpoolParams(convnet_common).updated({'kernel_shape': (2, 2), 'stride': (2, 2)}).freeze(),
        #
        #         ConvLayerParams(convnet_common).updated({'output_channels':128, 'kernel_shape':(3,3), 'stride':(1,1)}).freeze(),
        #         MaxpoolParams(convnet_common).updated({'kernel_shape':(2,2), 'stride':(2,2)}).freeze(),
        #     )
        # }).freeze()

        CONVNET = ConvStackParams({
            'op_name': 'Convnet',
            'tb': globals.tb,
            'layers': (
                ConvLayerParams(convnet_common).updated({'output_channels': 64, 'kernel_shape':(3,3), 'stride':(1,1), 'padding':'VALID'}).freeze(),
                # ConvLayerParams(convnet_common).updated({'output_channels': 64, 'kernel_shape':(3,3), 'stride':(1,1)}).freeze(),
                MaxpoolParams(convnet_common).updated({'kernel_shape':(2,2), 'stride':(2,2)}).freeze(),

                ConvLayerParams(convnet_common).updated({'output_channels':128, 'kernel_shape':(3,3), 'stride':(1,1)}).freeze(),
                MaxpoolParams(convnet_common).updated({'kernel_shape':(2,2), 'stride':(2,2)}).freeze(),

                ConvLayerParams(convnet_common).updated({'output_channels':256, 'kernel_shape':(3,3), 'stride':(1,1)}).freeze(),
                MaxpoolParams(convnet_common).updated({'kernel_shape':(2,2), 'stride':(2,2)}).freeze(),

                ConvLayerParams(convnet_common).updated({'output_channels':512, 'kernel_shape':(3,3), 'stride':(1,1)}).freeze(),
                MaxpoolParams(convnet_common).updated({'kernel_shape': (2, 2), 'stride': (2, 2)}).freeze(),

                ConvLayerParams(convnet_common).updated({'output_channels':512, 'kernel_shape':(3,3), 'stride':(1,1)}).freeze(),
                MaxpoolParams(convnet_common).updated({'kernel_shape':(2,2), 'stride':(2,2)}).freeze(),
            )
        }).freeze()

    # else: ## VGG16 architecture
    #     convnet_common = {
    #         'weights_initializer': globals.weights_initializer,
    #         'biases_initializer': globals.biases_initializer,
    #         'weights_regularizer': globals.weights_regularizer,
    #         'biases_regularizer': globals.biases_regularizer,
    #         'activation_fn': tf.nn.relu,
    #         'padding': 'SAME',
    #     }
    #     ## Conv and Maxpool architecture lifted from VGG16
    #     CONVNET = ConvStackParams({
    #         'op_name': 'Convnet',
    #         'tb': globals.tb,
    #         'layers': (
    #             ConvLayerParams(convnet_common).updated({'output_channels':64, 'kernel_shape':(3,3), 'stride':(1,1), 'padding':'VALID'}).freeze(),
    #             ConvLayerParams(convnet_common).updated({'output_channels':64, 'kernel_shape':(3,3), 'stride':(1,1)}).freeze(),
    #             MaxpoolParams(convnet_common).updated({'kernel_shape':(2,2), 'stride':(2,2)}).freeze(),
    #
    #             ConvLayerParams(convnet_common).updated({'output_channels':128, 'kernel_shape':(3,3), 'stride':(1,1)}).freeze(),
    #             ConvLayerParams(convnet_common).updated({'output_channels':128, 'kernel_shape':(3,3), 'stride':(1,1)}).freeze(),
    #             MaxpoolParams(convnet_common).updated({'kernel_shape':(2,2), 'stride':(2,2)}).freeze(),
    #
    #             ConvLayerParams(convnet_common).updated({'output_channels':256, 'kernel_shape':(3,3), 'stride':(1,1)}).freeze(),
    #             ConvLayerParams(convnet_common).updated({'output_channels':256, 'kernel_shape':(3,3), 'stride':(1,1)}).freeze(),
    #             ConvLayerParams(convnet_common).updated({'output_channels':256, 'kernel_shape':(3,3), 'stride':(1,1)}).freeze(),
    #             MaxpoolParams(convnet_common).updated({'kernel_shape':(2,2), 'stride':(2,2)}).freeze(),
    #
    #             ConvLayerParams(convnet_common).updated({'output_channels':512, 'kernel_shape':(3,3), 'stride':(1,1)}).freeze(),
    #             ConvLayerParams(convnet_common).updated({'output_channels':512, 'kernel_shape':(3,3), 'stride':(1,1)}).freeze(),
    #             ConvLayerParams(convnet_common).updated({'output_channels':512, 'kernel_shape':(3,3), 'stride':(1,1)}).freeze(),
    #             MaxpoolParams(convnet_common).updated({'kernel_shape':(2,2), 'stride':(2,2)}).freeze(),
    #
    #             ConvLayerParams(convnet_common).updated({'output_channels':512, 'kernel_shape':(3,3), 'stride':(1,1)}).freeze(),
    #             ConvLayerParams(convnet_common).updated({'output_channels':512, 'kernel_shape':(3,3), 'stride':(1,1)}).freeze(),
    #             ConvLayerParams(convnet_common).updated({'output_channels':512, 'kernel_shape':(3,3), 'stride':(1,1)}).freeze(),
    #             MaxpoolParams(convnet_common).updated({'kernel_shape':(2,2), 'stride':(2,2)}).freeze(),
    #         )
    #     }).freeze()

    HYPER = Im2LatexModelParams(initVals).updated({
        'CALSTM_STACK':(CALSTM_1,),
        'CONVNET': CONVNET
        })

    if freeze:
        HYPER.freeze()

    return HYPER
