#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 19:33:38 2017

@author: sumeet
"""

HYPER_PD = (
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
        #PD('embeddings_initializer_tf', 'Initializer for embedding weights', None, 
        #   tf.contrib.layers.xavier_initializer),
    ### Decoder LSTM Params ###
        PD('n',
           '(integer): Number of hidden-units of the LSTM cell',
           integer(100,10000),
           1000),
        PD('decoder_lstm_peephole',
           '(boolean): whether to employ peephole connections in the decoder LSTM',
           (True, False),
           False),
        PD('decoder_out_layers',
           'Number of layers in the decoder output MLP. defaults to 1 as in the papers source',
           xrange(1,10), 1),
        PD('output_activation', 'Activtion function for deep output layer', None,
           'tanh'),
        PD('output_follow_paper',
           'Output deep layer uses some funky logic in the paper instead of a straight MLP'
           'Setting this value to True (default) will follow the paper"s logic. Otherwise'
           "a straight MLP will be used.", boolean, 
           True),
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
