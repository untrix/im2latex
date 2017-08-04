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

Created on Tue Jul 25 13:41:32 2017

@author: Sumeet S Singh

Tested on python 2.7
"""
import os
import time
import argparse as arg
import numpy as np
import tensorflow as tf
import tf_commons as tfc
from Im2LatexModel import Im2LatexModel
from keras import backend as K
import hyper_params
from data_reader import BatchContextIterator
import dl_commons as dlc

def num_trainable_vars():
    total_n = 0
    for var in tf.trainable_variables():
        n = tfc.sizeofVar(var)
        total_n += n
    return total_n

def printVars():
    total_n = 0
    total_vggnet = 0
    total_init = 0
    total_calstm = 0
    total_output = 0
    total_embedding = 0

    print 'Trainable Variables'
    for var in tf.trainable_variables():
        n = tfc.sizeofVar(var)
        total_n += n
        if 'VGGNet/' in var.name :
            total_vggnet += n
        elif 'CALSTM' in var.name:
            total_calstm += n
        elif 'Im2LatexRNN/Output_Layer/' in var.name:
            total_output += n
        elif 'Initializer_MLP/' in var.name:
            total_init += n
        elif 'Im2LatexRNN/Embedding/Embedding_Matrix' in var.name:
            total_embedding += n
        else:
            assert False
        print var.name, K.int_shape(var), 'num_params = ', n

    print '\nTotal number of trainable params = ', total_n
    print 'Convnet: %d (%d%%)'%(total_vggnet, total_vggnet*100./total_n)
    print 'Initializer: %d (%d%%)'%(total_init, total_init*100./total_n)
    print 'CALSTM: %d (%d%%)'%(total_calstm, total_calstm*100./total_n)
    print 'Output Layer: %d (%d%%)'%(total_output, total_output*100./total_n)
    print 'Embedding Matrix: %d (%d%%)'%(total_embedding, total_embedding*100./total_n)

def train(num_steps, print_steps, num_epochs,
          raw_data_folder=None,
          vgg16_folder=None,
          globalParams=None,
          keep_prob=0.9):
    """
    Start training the model. All kwargs with default values==None must be supplied.
    """
    graph = tf.Graph()
    with graph.as_default():
        globalParams.update({'build_image_context':False,
                             'sum_logloss': False, ## setting to true equalizes ctc_loss and log_loss if y_s == squashed_seq
                             'dropout':tfc.DropoutParams({'keep_prob': tf.placeholder(tf.float32,
                                                                                      name="KeepProb")}),
                             'pLambda': 0.0005,
                             'MeanSumAlphaEquals1': False
                            })
        print('#################### Default Param Overrides: ####################\n%s'%(globalParams,))
        print('##################################################################\n%s'%(globalParams,))
        hyper = hyper_params.make_hyper(globalParams)
        print '#########################  Hyper-params: #########################\n%s'%(hyper,)
        print('##################################################################\n%s'%(globalParams,))
        batch_iterator = BatchContextIterator(raw_data_folder,
                                              vgg16_folder,
                                              hyper,
                                              num_steps,
                                              num_epochs)

        ## Training Graph
        with tf.name_scope('TrainingGraph'):
            model = Im2LatexModel(hyper, reuse=False)
            train_ops = model.build_train_graph()
            with tf.variable_scope('InputQueue'):
                enqueue_op = train_ops.inp_q.enqueue(batch_iterator.get_pyfunc(), name='enqueue')
            trainable_vars_n = num_trainable_vars()

        ## Validation Graph
        with tf.name_scope('DecodingGraph'):
            beamwidth=1
            hyper_predict = hyper_params.make_hyper(globalParams.copy().updated({'dropout':None}))
            model_predict = Im2LatexModel(hyper_predict, beamwidth, reuse=True)
            valid_ops = model_predict.beamsearch()
            with tf.variable_scope('InputQueue'):
                enqueue_op2 = valid_ops.inp_q.enqueue(batch_iterator.get_pyfunc(), name='enqueue')
            assert(num_trainable_vars() == trainable_vars_n)

        qr = tf.train.QueueRunner(train_ops.inp_q, [enqueue_op, enqueue_op2])
        coord = tf.train.Coordinator()

        printVars()

        config=tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as session:
            print 'Flushing graph to disk'
            tf_sw = tf.summary.FileWriter(tfc.makeTBDir(hyper.tb), graph=graph)
            tf_sw.flush()
            tf.global_variables_initializer().run()

            enqueue_threads = qr.create_threads(session, coord=coord, start=True)

            start_time = time.time()
            train_time = []
            valid_time = []
            step = 0
            try:
                while not coord.should_stop():
                    step_start_time = time.time()
                    feed_dict={hyper.dropout.keep_prob: keep_prob}
                    _, ll, ctc, cost, gstep, penalty, sai, sai2, msl = session.run((train_ops.train, 
                                    train_ops.log_likelihood,
                                    train_ops.ctc_loss,
                                    train_ops.cost,
                                    train_ops.global_step,
                                    train_ops.alpha_penalty,
                                    train_ops.mean_sum_alpha_i, train_ops.mean_sum_alpha_i2,
                                    train_ops.mean_seq_len), feed_dict=feed_dict)
                    step += 1
                    train_time.append(time.time()-step_start_time)
                    if step % print_steps == 0:
                        valid_start_time = time.time()
                        ## session.run((valid_ops.outputs, valid_ops.seq_lens))
                        valid_time.append(time.time() - valid_start_time)
                        print 'Time for %d steps, elapsed = %f, mean training = %f, mean validation = %f'%(gstep,
                                                                           time.time()-start_time,
                                                                           np.mean(train_time),
                                                                           np.mean(valid_time))
                        print 'Step %d, Log Perplexity %f, ctc_loss %f, penalty %f, cost %f, sum_alpha %f, sum_alpha1 %f, mean_seq_len %f'%(step,
                                                                                              ll[()],
                                                                                              ctc[()],
                                                                                              penalty[()],
                                                                                              cost[()],
                                                                                              sai[()],
                                                                                              sai2[()],
                                                                                              msl[()]
                                                                                              )
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            except Exception as e:
                coord.request_stop(e)
            finally:
                print 'Elapsed time for %d steps = %f'%(step, time.time()-start_time)
                coord.request_stop()
                coord.join(enqueue_threads)

def main():
    _data_folder = '../data/generated2'

    parser = arg.ArgumentParser(description='train model')
    parser.add_argument("--num-steps", "-n", dest="num_steps", type=int,
                        help="Number of training steps to run. Defaults to -1 if unspecified, i.e. run to completion",
                        default=-1)
    parser.add_argument("--num-epochs", "-e", dest="num_epochs", type=int,
                        help="Number of training steps to run. Defaults to 10 if unspecified.",
                        default=10)
    parser.add_argument("--batch-size", "-b", dest="batch_size", type=int,
                        help="Batchsize. If unspecified, defaults to the default value in hyper_params",
                        default=None)
    parser.add_argument("--print-steps", "-s", dest="print_steps", type=int,
                        help="Number of training steps after which to log results. Defaults to 50 if unspecified",
                        default=50)
    parser.add_argument("--data-folder", "-d", dest="data_folder", type=str,
                        help="Data folder. If unspecified, defaults to " + _data_folder,
                        default=_data_folder)
    parser.add_argument("--raw-data-folder", dest="raw_data_folder", type=str,
                        help="Raw data folder. If unspecified, defaults to data_folder/training",
                        default=None)
    parser.add_argument("--vgg16-folder", dest="vgg16_folder", type=str,
                        help="vgg16 data folder. If unspecified, defaults to raw_data_folder/vgg16_features",
                        default=None)
    parser.add_argument("--image-folder", dest="image_folder", type=str,
                        help="image folder. If unspecified, defaults to data_folder/formula_images",
                        default=None)
    parser.add_argument("--partial-batch", "-p",  dest="partial_batch", action='store_true',
                        help="Sets assert_whole_batch hyper param to False. Default hyper_param value will be used if unspecified")
    parser.add_argument("--queue-capacity", "-q", dest="queue_capacity", type=int,
                        help="Capacity of input queue. Defaults to hyperparam defaults if unspecified.",
                        default=None)

    args = parser.parse_args()
    data_folder = args.data_folder
    if args.image_folder:
        image_folder = args.image_folder
    else:
        image_folder = os.path.join(data_folder,'formula_images')

    if args.raw_data_folder:
        raw_data_folder = args.raw_data_folder
    else:
        raw_data_folder = os.path.join(data_folder, 'training')

    if args.vgg16_folder:
        vgg16_folder = args.vgg16_folder
    else:
        vgg16_folder = os.path.join(raw_data_folder, 'vgg16_features')

    globalParams = dlc.Properties()
    
    if args.batch_size is not None:
        globalParams.B = args.batch_size
    if args.partial_batch:
        globalParams.assert_whole_batch = False
    if args.queue_capacity is not None:
        globalParams.input_queue_capacity = args.queue_capacity

    train(args.num_steps, args.print_steps, args.num_epochs,
          raw_data_folder=raw_data_folder,
          vgg16_folder=vgg16_folder,
          globalParams=globalParams)

main()
