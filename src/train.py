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
import logging
import argparse as arg
import numpy as np
import tensorflow as tf
import tf_commons as tfc
from Im2LatexModel import Im2LatexModel
from keras import backend as K
import hyper_params
from data_reader import create_context_iterators
import dl_commons as dlc
import nltk
from nltk.util import ngrams

def num_trainable_vars():
    total_n = 0
    for var in tf.trainable_variables():
        n = tfc.sizeofVar(var)
        total_n += n
    return total_n

def printVars(logger):
    total_n = 0
    total_vggnet = 0
    total_init = 0
    total_calstm = 0
    total_output = 0
    total_embedding = 0

    logger.warn( 'Trainable Variables')
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
        logger.warn('%s %s num_params = %d'%(var.name, K.int_shape(var),n) )

    logger.warn( '\nTotal number of trainable params = %d'%total_n)
    logger.warn( 'Convnet: %d (%d%%)'%(total_vggnet, total_vggnet*100./total_n))
    logger.warn( 'Initializer: %d (%d%%)'%(total_init, total_init*100./total_n))
    logger.warn( 'CALSTM: %d (%d%%)'%(total_calstm, total_calstm*100./total_n))
    logger.warn( 'Output Layer: %d (%d%%)'%(total_output, total_output*100./total_n))
    logger.warn( 'Embedding Matrix: %d (%d%%)'%(total_embedding, total_embedding*100./total_n))

def train(num_steps, print_steps, num_epochs,
          raw_data_folder=None,
          vgg16_folder=None,
          globalParams=None,
          keep_prob=None):
    """
    Start training the model. All kwargs with default values==None must be supplied.
    """
    logger = globalParams.logger
    graph = tf.Graph()
    with graph.as_default():
        globalParams.update({'build_image_context':False,
                             'sum_logloss': False, ## setting to true equalizes ctc_loss and log_loss if y_s == squashed_seq
                             'dropout': None if keep_prob >= 1.0 else tfc.DropoutParams({'keep_prob': keep_prob}),
                             'pLambda': 0.0005,
                             'MeanSumAlphaEquals1': False
                            })
        logger.warn('\n#################### Default Param Overrides: ####################\n%s'%(globalParams,))
        logger.warn('##################################################################\n')
        hyper = hyper_params.make_hyper(globalParams)
        logger.warn( '\n#########################  Hyper-params: #########################\n%s'%(hyper,))
        logger.warn('##################################################################\n')
        batch_iterator, batch_iterator2 = create_context_iterators(raw_data_folder,
                                              vgg16_folder,
                                              hyper,
                                              globalParams)

        ##### Training Graph
        with tf.name_scope('Training'):
            model = Im2LatexModel(hyper, reuse=False)
            train_ops = model.build_train_graph()
            with tf.variable_scope('InputQueue'):
                enqueue_op = train_ops.inp_q.enqueue(batch_iterator.get_pyfunc(), name='enqueue')
                close_queue1 = train_ops.inp_q.close(cancel_pending_enqueues=True)
            trainable_vars_n = num_trainable_vars() # 8544670
            assert 8544670 == trainable_vars_n

        ##### Validation Graph
        with tf.name_scope('Validation'):
            beamwidth=globalParams.beam_width
            hyper_predict = hyper_params.make_hyper(globalParams.copy().updated({'dropout':None}))
            model_predict = Im2LatexModel(hyper_predict, beamwidth, reuse=True)
            valid_ops = model_predict.test()
            with tf.variable_scope('InputQueue'):
                enqueue_op2 = valid_ops.inp_q.enqueue(batch_iterator2.get_pyfunc(), name='enqueue')
                close_queue2 = valid_ops.inp_q.close(cancel_pending_enqueues=True)
            assert(num_trainable_vars() == trainable_vars_n)

        qr1 = tf.train.QueueRunner(train_ops.inp_q, [enqueue_op], cancel_op=[close_queue1])
        qr2 = tf.train.QueueRunner(valid_ops.inp_q, [enqueue_op2], cancel_op=[close_queue2])
        coord = tf.train.Coordinator()

        printVars(logger)

        config=tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as session:
            logger.warn( 'Flushing graph to disk')
            tf_sw = tf.summary.FileWriter(tfc.makeTBDir(hyper.tb), graph=graph)
            tf_sw.flush()
            tf.global_variables_initializer().run()

            enqueue_threads = qr1.create_threads(session, coord=coord, start=True)
            enqueue_threads.extend(qr2.create_threads(session, coord=coord, start=True))

            start_time = time.time()
            train_time = []; ctc_losses = []; logs = []
            step = 0
            try:
                while not coord.should_stop():
                    step_start_time = time.time()
                    # if hyper.dropout is not None:
                    #     feed_dict={hyper.dropout.keep_prob: keep_prob}
                    # else:
                    #     feed_dict=None
                    _, ctc_loss, log = session.run(
                        (
                            train_ops.train, 
                            train_ops.ctc_loss,
                            train_ops.tb_logs
                        ))
                    ctc_losses.append(ctc_loss[()])
                    logs.append(log)
                    train_time.append(time.time()-step_start_time)
                    step += 1

                    do_log, num_valid_batches = get_logging_steps(step, globalParams, batch_iterator, batch_iterator2)
                    if do_log:
                        logger.warn( 'Step %d'%(step))
                        train_time_per100 = np.mean(train_time) * 100. / hyper.B
                        valid_res = validation_cycle(session, valid_ops, batch_iterator2, hyper, globalParams, step, tf_sw, num_valid_batches)
                        logger.warn('Time for %d steps, elapsed = %f, training-time-per-100 = %f, validation-time-per-100 = %f'%(
                            step,
                            time.time()-start_time,
                            train_time_per100,
                            valid_res.valid_time_per100))
                        ## emit training graph metrics of the minimum and maximum loss batches
                        i_min = np.argmin(ctc_losses)
                        i_max = np.argmax(ctc_losses)
                        if i_min < i_max:
                            tf_sw.add_summary(logs[i_min], global_step=(step+i_min+1 - print_steps))
                            tf_sw.add_summary(logs[i_max], global_step= (step+i_max+1 - print_steps))
                        elif i_min > i_max:
                            tf_sw.add_summary(logs[i_max], global_step= (step+i_max+1 - print_steps))
                            tf_sw.add_summary(logs[i_min], global_step=(step+i_min+1 - print_steps))
                        else:
                            if step == 1:
                                tf_sw.add_summary(logs[i_min], global_step=1)
                            else:
                                tf_sw.add_summary(logs[i_min], global_step=(step+i_min+1 - print_steps))

                        log_time = session.run(train_ops.log_time, feed_dict={train_ops.ph_train_time: train_time_per100})
                        tf_sw.add_summary(log_time, global_step=step)
                        tf_sw.flush()
                        ## reset metrics
                        train_time = []; ctc_losses = []; logs = []

            except tf.errors.OutOfRangeError, StopIteration:
                logger.warn('Done training -- epoch limit reached')
            except Exception as e:
                logger.warn( '***************** Exiting with exception: *****************\n%s'%e)
                coord.request_stop(e)
            finally:
                logger.warn( 'Elapsed time for %d steps = %f'%(step, time.time()-start_time))
                coord.request_stop()
                # close_queue1.run()
                # close_queue2.run()
                coord.join(enqueue_threads)

def get_logging_steps(step, args, train_it, valid_it):
    valid_epochs = args.valid_epochs if (args.valid_epochs is not None) else 1
    valid_steps = int(valid_epochs * train_it.epoch_size)
    do_log = (step % args.print_steps == 0) or (step % valid_steps == 0) or (step == train_it.max_steps)
    num_valid_batches = valid_it.epoch_size if (step % valid_steps == 0) or (step == train_it.max_steps) else 1

    return do_log, num_valid_batches


def validation_cycle(session, valid_ops, batch_iterator, hyper, args, step, tf_sw, max_steps):
    logger = hyper.logger
    valid_start_time = time.time()
    batch_size = batch_iterator.batch_size
    epoch_size = batch_iterator.epoch_size
    ## Print a batch randomly
    print_batch_num = np.random.randint(0, epoch_size) if args.print_batch else -1
    eds = []; best_eds = []
    accuracies = []; best_accuracies = []
    lens = []
    n = 0
    hyper.logger.warn('validation cycle starting for %d steps'%max_steps)
    while n < max_steps:
        n += 1
        ids = l = s = b = bis = bids = bl = best_accuracy = accuracy = ed = best_ed = None

        if (not args.print_batch) or (n != print_batch_num):
            ed, best_ed, accuracy, best_accuracy = session.run((
                                valid_ops.top1_mean_ed,
                                valid_ops.bok_mean_ed,
                                valid_ops.top1_accuracy,
                                valid_ops.bok_accuracy
                                ))
            eds.append(ed)
            best_eds.append(best_ed)
            accuracies.append(accuracy)
            best_accuracies.append(best_accuracy)

        else:
            ids, l, s, bis, bids, bl, ed, best_ed, accuracy, best_accuracy = session.run((
                                valid_ops.topK_ids, 
                                valid_ops.topK_lens,
                                valid_ops.topK_scores,
                                valid_ops.all_id_scores,
                                valid_ops.all_ids,
                                valid_ops.all_seq_lens,
                                valid_ops.top1_mean_ed,
                                valid_ops.bok_mean_ed,
                                valid_ops.top1_accuracy,
                                valid_ops.bok_accuracy
                                ))
            eds.append(ed)
            best_eds.append(best_ed)
            accuracies.append(accuracy)
            best_accuracies.append(best_accuracy)

            i = np.random.randint(0, batch_iterator.batch_size)
            logger.warn( '############ RANDOM VALIDATION BATCH %d SAMPLE %d ############'%(n, i))
            beam = 0
            logger.warn( 'top k lens=%s'%l[i])
            logger.warn( 'top k sequence scores=%s'%s[i])
            logger.warn( 'top 1 token sequence=%s'%ids[i,0])
            # print 'top token scores=%s'%bis[i, 0]
            # print 'all beam sequences=%s'%bids[i]
            # print 'all beam token scores = %s'%bis[i]
            # print 'all beam lens = %s'%bl[i]
            # for k in range(l[i].shape[0]):
            #     assert l[i, k] == bl[i, k]
            # for k in range(l[i].shape[0]):
            #     try:
            #         sum_scores = np.sum(bis[i, k])
            #         reported_score = s[i,k]
            #         assert sum_scores -  reported_score < 0.00001
            #     except:
            #         hyper.logger.critical('\nBEAM SCORES DO NOT MATCH: %f vs %f\n'%(sum_scores,reported_score))
            assert np.argmax(np.sum(bis[i], axis=-1)) == beam
            # assert np.all(ids[i,0] == bids[i,beam])
            logger.warn( '###################################\n')
            assert np.all(bl <= (hyper.Max_Seq_Len + 10))

    metrics = dlc.Properties({
        'len': lens,
        'edit_distance': eds,
        'BoK_distance': best_eds,
        'accuracy': accuracies,
        'bok_accuracy': best_accuracies,
        'valid_time_per100': (time.time() - valid_start_time) * 100. / (max_steps * hyper.B)
    })

    logs_aggregate = session.run(valid_ops.logs_aggregate,
                                feed_dict={
                                    valid_ops.ph_seq_lens: lens,
                                    valid_ops.ph_edit_distance: metrics.edit_distance,
                                    valid_ops.ph_BoK_distance: metrics.BoK_distance,
                                    valid_ops.ph_accuracy: metrics.accuracy,
                                    valid_ops.ph_BoK_accuracy: metrics.bok_accuracy,
                                    valid_ops.ph_valid_time : metrics.valid_time_per100
                                })
    tf_sw.add_summary(logs_aggregate, step)
    tf_sw.flush()
    hyper.logger.warn('validation cycle finished')
    return metrics

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
    parser.add_argument("--beam-width", "-w", dest="beam_width", type=int,
                        help="Beamwidth. If unspecified, defaults to 100",
                        default=100)
    parser.add_argument("--print-steps", "-s", dest="print_steps", type=int,
                        help="Number of training steps after which to log results. Defaults to 50 if unspecified",
                        default=50)
    parser.add_argument("--keep-prob", "-k", dest="keep_prob", type=float,
                        help="Dropout 'keep' probability. Defaults to 0.9",
                        default=0.9)
    parser.add_argument("--adap_alpha", "-a", dest="alpha", type=float,
                        help="Alpha (step / learning-rate) value of adam optimizer.",
                        default=None)
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
    parser.add_argument("--logging-level", "-l", dest="logging_level", type=int,
                        help="Logging verbosity level from 1 to 5 in increasing order of verbosity.",
                        default=3)
    parser.add_argument("--valid-frac", "-f", dest="valid_frac", type=float,
                        help="Fraction of samples to use for validation. Defaults to 0.01",
                        default=0.05)
    parser.add_argument("--validation-epochs", "-v", dest="valid_epochs", type=float,
                        help="Number (and fraction) of epochs after which to run a full validation cycle. Defaults to 1.",
                        default=1.)
    parser.add_argument("--print-batch",  dest="print_batch", action='store_true',
                        help="(Boolean): Only for debugging. Prints more stuff once in a while. Defaults to False.",
                        default=False)

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

    logger = logging.getLogger()
    logging_level = (logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG)
    logger.setLevel(logging_level[args.logging_level - 1])
    logger.addHandler(logging.StreamHandler())

    globalParams = dlc.Properties({
                                    'print_steps': args.print_steps,
                                    'num_steps': args.num_steps,
                                    'num_epochs': args.num_epochs,
                                    'logger': logger,
                                    'beam_width':args.beam_width,
                                    'valid_frac': args.valid_frac,
                                    'valid_epochs': args.valid_epochs,
                                    'print_batch': args.print_batch
                                    })
    
    if args.batch_size is not None:
        globalParams.B = args.batch_size
    if args.partial_batch:
        globalParams.assert_whole_batch = False
    if args.queue_capacity is not None:
        globalParams.input_queue_capacity = args.queue_capacity
    if args.alpha is not None:
        globalParams.adam_alpha = args.alpha

    train(args.num_steps, args.print_steps, args.num_epochs,
          raw_data_folder=raw_data_folder,
          vgg16_folder=vgg16_folder,
          globalParams=globalParams,
          keep_prob=args.keep_prob
         )

main()
