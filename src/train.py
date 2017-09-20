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
import numpy as np
import tensorflow as tf
import tf_commons as tfc
from Im2LatexModel import Im2LatexModel
from keras import backend as K
import hyper_params
from data_reader import create_context_iterators, create_imagenet_iterators, create_BW_image_iterators
import dl_commons as dlc
import data_commons as dtc
# import nltk
# from nltk.util import ngrams

logger = None

def num_trainable_vars():
    total_n = 0
    for var in tf.trainable_variables():
        n = tfc.sizeofVar(var)
        total_n += n
    return total_n

def printVars(logger):
    total_n = 0
    total_vggnet = 0
    total_convnet = 0
    total_init = 0
    total_calstm = 0
    total_output = 0
    total_embedding = 0

    logger.info( 'Trainable Variables')
    for var in tf.trainable_variables():
        n = tfc.sizeofVar(var)
        total_n += n
        if 'VGGNet/' in var.name :
            total_vggnet += n
        elif 'Convnet/' in var.name:
            total_convnet += n
        elif 'CALSTM' in var.name:
            total_calstm += n
        elif 'I2L_RNN/Output_Layer/' in var.name:
            total_output += n
        elif 'Initializer_MLP/' in var.name:
            total_init += n
        elif 'I2L_RNN/Embedding/Embedding_Matrix' in var.name:
            total_embedding += n
        else:
            assert False, 'unrecognized variable %s'%var
        logger.info('%s %s num_params = %d'%(var.name, K.int_shape(var),n) )

    logger.info( 'Total number of trainable params = %d'%total_n)
    logger.info( 'Vggnet: %d (%d%%)'%(total_vggnet, total_vggnet*100./total_n))
    logger.info( 'Convnet: %d (%d%%)'%(total_convnet, total_vggnet*100./total_n))
    logger.info( 'Initializer: %d (%d%%)'%(total_init, total_init*100./total_n))
    logger.info( 'CALSTM: %d (%d%%)'%(total_calstm, total_calstm*100./total_n))
    logger.info( 'Output Layer: %d (%d%%)'%(total_output, total_output*100./total_n))
    logger.info( 'Embedding Matrix: %d (%d%%)'%(total_embedding, total_embedding*100./total_n))

def main(raw_data_folder,
          vgg16_folder,
          args,
          hyper):
    """
    Start training the model.
    """
    dtc.initialize(args.generated_data_dir, hyper)
    global logger
    logger = hyper.logger

    graph = tf.Graph()
    with graph.as_default():
        if hyper.build_image_context == 1:
            train_it, valid_it, tr_acc_it = create_imagenet_iterators(raw_data_folder,
                                                hyper,
                                                args)
        elif hyper.build_image_context == 2:
            train_it, valid_it, tr_acc_it = create_BW_image_iterators(raw_data_folder,
                                                hyper,
                                                args)
        else:
            train_it, valid_it, tr_acc_it = create_context_iterators(raw_data_folder,
                                                vgg16_folder,
                                                hyper,
                                                args)

        ##### Training Graphs
        with tf.name_scope('Training'):
            train_ops = []
            hyper.optimizer = tf.train.AdamOptimizer(learning_rate=hyper.adam_alpha)
            tf_train_step = tf.get_variable('global_step', dtype=hyper.int_type, trainable=False, initializer=0)
            with tf.variable_scope('InputQueue'):
                train_q = tf.FIFOQueue(self.C.input_queue_capacity,
                                    (self.C.int_type, self.C.int_type,
                                    self.C.int_type, self.C.int_type,
                                    self.C.dtype))
                enqueue_op = train_q.enqueue(train_it.get_pyfunc(), name='enqueue')
                close_queue1 = train_q.close(cancel_pending_enqueues=True)
            with tf.device('/gpu:1'):
                model = Im2LatexModel(hyper, train_q, reuse=False, )
                train_ops.append( model.build_training_graph())
            with tf.device('/gpu:0'):
                model = Im2LatexModel(hyper, train_q, reuse=True)
                train_ops.append(model.build_training_graph())
            trainable_vars_n = num_trainable_vars() # 8544670 or 8547670
            hyper.logger.info('Num trainable variables = %d', trainable_vars_n)
            ## assert trainable_vars_n == 8547670 if hyper.use_peephole else 8544670
            ## assert trainable_vars_n == 23261206 if hyper.build_image_context
        qr1 = tf.train.QueueRunner(train_q, [enqueue_op], cancel_op=[close_queue1])

        ##### Validation Graph
        with tf.name_scope('Validation'):
            hyper_predict = hyper_params.make_hyper(args.copy().updated({'dropout':None}))
            with tf.variable_scope('InputQueue'):
                valid_q = tf.FIFOQueue(self.C.input_queue_capacity,
                                    (self.C.int_type, self.C.int_type,
                                    self.C.int_type, self.C.int_type,
                                    self.C.dtype))
                enqueue_op2 = valid_q.enqueue(valid_it.get_pyfunc(), name='enqueue')
                close_queue2 = valid_q.close(cancel_pending_enqueues=True)
            with tf.device('/gpu:0'):
                model_predict = Im2LatexModel(hyper_predict, valid_q, hyper.seq2seq_beam_width, reuse=True)
                valid_ops_0 = model_predict.test()
            with tf.device('/gpu:1'):
                model_predict = Im2LatexModel(hyper_predict, valid_q, hyper.seq2seq_beam_width, reuse=True)
                valid_ops_1 = model_predict.test()
            hyper.logger.info('Num trainable variables = %d', num_trainable_vars())
            assert(num_trainable_vars() == trainable_vars_n)
        qr2 = tf.train.QueueRunner(valid_q, [enqueue_op2], cancel_op=[close_queue2])

        # ##### Training Accuracy Graph
        # if (args.make_training_accuracy_graph):
        #     with tf.name_scope('TrainingAccuracy'):
        #         hyper_predict2 = hyper_params.make_hyper(args.copy().updated({'dropout':None}))
        #         with tf.device('/gpu:1'):
        #             model_predict2 = Im2LatexModel(hyper_predict, hyper.seq2seq_beam_width, reuse=True)
        #             tr_acc_ops = model_predict2.test()
        #         with tf.variable_scope('QueueOps'):
        #             enqueue_op3 = tr_acc_ops.inp_q.enqueue(tr_acc_it.get_pyfunc(), name='enqueue')
        #             close_queue3 = tr_acc_ops.inp_q.close(cancel_pending_enqueues=True)
        #         assert(num_trainable_vars() == trainable_vars_n)
        #     qr3 = tf.train.QueueRunner(tr_acc_ops.inp_q, [enqueue_op3], cancel_op=[close_queue3])
        # else:
        tr_acc_ops = None

        coord = tf.train.Coordinator()
        # print train_ops

        printVars(logger)

        config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config.gpu_options.allow_growth = hyper.tf_session_allow_growth

        with tf.Session(config=config) as session:
            logger.info('Flushing graph to disk')
            tf_sw = tf.summary.FileWriter(args.logdir, graph=graph)
            # tf_params = tf.constant(value=hyper.to_table(), dtype=tf.string, name='hyper_params')
            # tf_text = tf.summary.text('hyper_params_logger', tf_params)
            # log_params = session.run(tf_text)
            # tf_sw.add_summary(log_params, global_step=None)
            tf_sw.flush()

            enqueue_threads = qr1.create_threads(session, coord=coord, start=True)
            enqueue_threads.extend(qr2.create_threads(session, coord=coord, start=True))
            if (args.make_training_accuracy_graph):
                enqueue_threads.extend(qr3.create_threads(session, coord=coord, start=True))

            saver = tf.train.Saver(max_to_keep=5, pad_step_number=True, save_relative_paths=True)
            if args.restore_from_checkpoint:
                saver.recover_last_checkpoints(args.logdir + '/checkpoints_list')
                saver.restore(session, saver.last_checkpoints[-1])
                step = tf_train_step.eval()
            else:
                tf.global_variables_initializer().run()
                step = 0

            start_time = time.time()
            ## Set metrics
            train_time = []; ctc_losses = []; logs = []
            try:
                while not coord.should_stop():
                    step_start_time = time.time()
                    step += 1
                    doLog = do_log(step, args, train_it, valid_it)
                    if not doLog:
                        _, ctc_loss, log = session.run(
                            (
                                train_ops.train, 
                                train_ops.ctc_loss,
                                train_ops.tb_logs
                            ))
                        predicted_ids = y_s = None
                    else:
                        _, ctc_loss, log, y_s, predicted_ids = session.run(
                            (
                                train_ops.train, 
                                train_ops.ctc_loss,
                                train_ops.tb_logs,
                                train_ops.y_s,
                                train_ops.predicted_ids
                            ))

                    ## Accumulate metrics
                    ctc_losses.append(ctc_loss[()])
                    logs.append(log)
                    train_time.append(time.time()-step_start_time)

                    if doLog:
                        logger.info( 'Step %d',step)
                        train_time_per100 = np.mean(train_time) * 100. / hyper.B
                        accuracy_res = measure_accuracy(
                            session,
                            dlc.Properties({'valid_ops':valid_ops, 'tr_acc_ops':tr_acc_ops}), 
                            dlc.Properties({'train_it':train_it, 'valid_it':valid_it, 'tr_acc_it':tr_acc_it}),
                            hyper,
                            args,
                            step, 
                            tf_sw)
                        if accuracy_res:
                            logger.info('Time for %d steps, elapsed = %f, training-time-per-100 = %f, validation-time-per-100 = %f'%(
                                step,
                                time.time()-start_time,
                                train_time_per100,
                                accuracy_res.valid_time_per100))
                        else:
                            logger.info('Time for %d steps, elapsed = %f, training-time-per-100 = %f'%(
                                step,
                                time.time()-start_time,
                                train_time_per100))
                            
                        ## emit training graph metrics of the minimum and maximum loss batches
                        i_min = np.argmin(ctc_losses)
                        i_max = np.argmax(ctc_losses)
                        if i_min < i_max:
                            tf_sw.add_summary(logs[i_min], global_step=(step+i_min+1 - args.print_steps))
                            tf_sw.add_summary(logs[i_max], global_step= (step+i_max+1 - args.print_steps))
                        elif i_min > i_max:
                            tf_sw.add_summary(logs[i_max], global_step= (step+i_max+1 - args.print_steps))
                            tf_sw.add_summary(logs[i_min], global_step=(step+i_min+1 - args.print_steps))
                        else:
                            if step == 1:
                                tf_sw.add_summary(logs[i_min], global_step=1)
                            else:
                                tf_sw.add_summary(logs[i_min], global_step=(step+i_min+1 - args.print_steps))

                        log_time = session.run(train_ops.log_time, feed_dict={train_ops.ph_train_time: train_time_per100})
                        tf_sw.add_summary(log_time, global_step=step)
                        tf_sw.flush()
                        
                        logger.info( '############ RANDOM TRAINING BATCH ############')
                        logger.info('[target_ids, predicted_ids]=\n%s', ids2str(y_s, predicted_ids))
                        logger.info( '###################################\n')

                        if do_validate(step, args, train_it, valid_it):
                            saver.save(session, args.logdir + '/snapshot', global_step=step, latest_filename='checkpoints_list')

                        ## Reset metrics
                        train_time = []; ctc_losses = []; logs = []

            except tf.errors.OutOfRangeError, StopIteration:
                logger.info('Done training -- epoch limit reached')
            except Exception as e:
                logger.info( '***************** Exiting with exception: *****************\n%s'%e)
                coord.request_stop(e)
            finally:
                logger.info( 'Elapsed time for %d steps = %f'%(step, time.time()-start_time))
                coord.request_stop()
                # close_queue1.run()
                # close_queue2.run()
                coord.join(enqueue_threads)

def ids2str(target_ids, predicted_ids):
    """
    Args:
        target_ids: Numpy array of shape (B,T)
        predicted_ids: Numpy array of same shape as target_ids
    """
    target_str = np.expand_dims(dtc.seq2str(target_ids, 'Target:'), axis=1)
    predicted_str = np.expand_dims(dtc.seq2str(predicted_ids, 'Prediction:'),axis=1)
    return np.concatenate((predicted_str, target_str), axis=1)

def do_validate(step, args, train_it, valid_it):
    valid_frac = args.valid_epochs if (args.valid_epochs is not None) else 1
    valid_steps = int(valid_frac * train_it.epoch_size)
    do_validate = (step % valid_steps == 0) or (step == train_it.max_steps)
    num_valid_batches = valid_it.epoch_size if do_validate else 0

    logger.debug('do_validate returning %s at step %d', do_validate, step)

    return do_validate, num_valid_batches

def do_log(step, args, train_it, valid_it):
    validate, _ = do_validate(step, args, train_it, valid_it)
    do_log = (step % args.print_steps == 0) or (step == train_it.max_steps) or validate
    logger.debug('do_log returning %s at step %d', do_log, step)
    return do_log

def format_ids(predicted_ids, target_ids):
    np.apply_along_axis

def measure_accuracy(session, ops, batch_its, hyper, args, step, tf_sw):
    logger = hyper.logger
    validate, num_steps = do_validate(step, args, batch_its.train_it, batch_its.valid_it)
    valid_start_time = time.time()
    if not validate:
        if args.make_training_accuracy_graph: # Run one training batch thought training_accuracy_graph
            logs_top1, logs_topK, bok_ids = session.run((
                                ops.tr_acc_ops.logs_tr_acc_top1,
                                ops.tr_acc_ops.logs_tr_acc_topK,
                                ops.tr_acc_ops.bok_ids
                                ))
            tf_sw.add_summary(logs_top1, step)
            tf_sw.add_summary(logs_topK, step)
            tf_sw.flush()
            metrics = dlc.Properties({
                'valid_time_per100': ((time.time() - valid_start_time) * 100. / (hyper.B * hyper.seq2seq_beam_width))
                })
            logger.info( '############ RANDOM TRAINING BATCH ############')
            logger.info('bok ids=\n%s', bok_ids)
            logger.info( '###################################\n')        
            return metrics
        else:
            return None
    else: ## run a full validation cycle
        valid_ops = ops.valid_ops
        batch_it = batch_its.valid_it
        batch_size = batch_it.batch_size
        epoch_size = batch_it.epoch_size
        ## Print a batch randomly
        print_batch_num = np.random.randint(0, epoch_size) if args.print_batch else -1
        eds = []; best_eds = []
        accuracies = []; best_accuracies = []
        lens = []
        hits = []
        n = 0
        hyper.logger.info('validation cycle starting for %d steps', num_steps)
        while n < num_steps:
            n += 1
            if (n != print_batch_num):
                l, ed, accuracy, num_hits = session.run((
                                    valid_ops.top1_lens,
                                    valid_ops.top1_mean_ed,
                                    valid_ops.top1_accuracy,
                                    valid_ops.top1_num_hits
                                    ))
            else:
                l, ed, accuracy, num_hits, top1_ids, y_s = session.run((
                                    valid_ops.top1_lens,
                                    valid_ops.top1_mean_ed,
                                    valid_ops.top1_accuracy,
                                    valid_ops.top1_num_hits,
                                    valid_ops.top1_ids,
                                    valid_ops.y_s
                                    ))
                logger.info( '############ RANDOM VALIDATION BATCH %d ############', n)
                beam = 0
                logger.info('prediction mean_ed=%f', ed)
                logger.info('prediction accuracy=%f', accuracy)
                logger.info('prediction hits=%d', num_hits)
                logger.info('[target_ids, predicted_ids]=\n%s', ids2str(y_s, top1_ids))
                logger.info( '###################################\n')

            lens.append(l)
            eds.append(ed)
            accuracies.append(accuracy)
            hits.append(num_hits)

        metrics = dlc.Properties({
            'valid_time_per100': (time.time() - valid_start_time) * 100. / (num_steps * hyper.B * hyper.seq2seq_beam_width)
            })
        logs_agg_top1 = session.run(valid_ops.logs_agg_top1,
                                    feed_dict={
                                        valid_ops.ph_top1_seq_lens: lens,
                                        valid_ops.ph_edit_distance: eds,
                                        valid_ops.ph_num_hits: hits,
                                        valid_ops.ph_accuracy: accuracies,
                                        valid_ops.ph_valid_time : metrics.valid_time_per100
                                    })
        tf_sw.add_summary(logs_agg_top1, step)
        tf_sw.flush()
        hyper.logger.info('validation cycle finished')
        return metrics

