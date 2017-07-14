#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 11:37:52 2017

@author: sumeet
"""
import tensorflow as tf
import tf_commons as tfc
import dl_commons as dlc
from Im2LatexDecoderRNNParams import D_RNN
from Im2LatexRNN import Im2LatexDecoderRNN
from Im2LatexModel import Im2LatexModel, HYPER
from keras import backend as K

def test_rnn():
    B = HYPER.B
    Kv = HYPER.K
    L = HYPER.L

    ## TODO: Introduce Stochastic Learning

    m = Im2LatexModel()
    y_s = tf.placeholder(tf.int32, shape=(HYPER.B, None))
    print 'y_s[:,0] shape: ', K.int_shape(y_s[:,0])
    print 'embedding_lookup shape: ', K.int_shape(m._embedding_lookup(y_s[:,0]))
    ## im = tf.placeholder(dtype=tf.float32, shape=(HYPER.B,) + HYPER.image_shape, name='image_batch')
    ## a = m._build_image_context(im)
    a = tf.placeholder(dtype=tf.float32, shape=(D_RNN.B, D_RNN.L, D_RNN.D), name='image_a')
    rnn = Im2LatexDecoderRNN(D_RNN, a, 10)
    print 'rnn ', rnn.state_size, rnn.output_size
    #init_c, init_h = m._build_init_layer(a)

    decoder = tf.contrib.seq2seq.BeamSearchDecoder(rnn,
                                                   m._embedding_lookup,
                                                   y_s[:,0],
                                                   0,
                                                   rnn.zero_state(D_RNN.B*rnn.BeamWidth, tf.float32),
                                                   beam_width=rnn.BeamWidth)
    
    print 'decoder._start_tokens: ', K.int_shape(decoder._start_tokens)
    print 'decoder._start_inputs: ', K.int_shape(decoder._start_inputs)
    final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                                           maximum_iterations=D_RNN.Max_Seq_Len + 10,
                                                                                           swap_memory=True)
    print 'final_outputs: ', K.int_shape(final_outputs.predicted_ids)
    print 'final_state: ', (final_state)
    print 'final_sequence_lengths', (final_sequence_lengths)
    
    return final_outputs.predicted_ids

def visualize_rnn():
    graph = tf.Graph()
    with graph.as_default():

        a = tf.placeholder(dtype=tf.float32, shape=(D_RNN.B, D_RNN.L, D_RNN.D), name='image_a')
        init_Ex = tf.placeholder(tf.float32, shape=(D_RNN.B, D_RNN.m), name='init_Ex')
        rnn = Im2LatexDecoderRNN(D_RNN, a, 1)
        init_state = rnn.zero_state(D_RNN.B*rnn.BeamWidth, tf.float32)
        print 'type of init_state = ', type(init_state)
        output, state = rnn(init_Ex, init_state)
        
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:

            print 'Flushing graph to disk'
            tf_sw = tf.summary.FileWriter(tfc.makeTBDir(D_RNN.tb), graph=graph)
            tf_sw.flush()

test_rnn()
visualize_rnn()
