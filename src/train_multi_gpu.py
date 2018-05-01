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
import time
import numpy as np
import tensorflow as tf
import tf_commons as tfc
from Im2LatexModel import Im2LatexModel, sync_testing_towers, sync_training_towers
from keras import backend as K
import hyper_params
from data_reader import create_context_iterators, create_imagenet_iterators, create_BW_image_iterators
import dl_commons as dlc
import data_commons as dtc
import signal

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
    total_lstm_0 = 0
    total_lstm_1 = 0
    total_att1 = 0
    total_calstm2 = 0
    total_lstm2_0 = 0
    total_lstm2_1 = 0
    total_att2 = 0
    total_output = 0
    total_embedding = 0

    logger.info('Trainable Variables')
    for var in tf.trainable_variables():
        n = tfc.sizeofVar(var)
        total_n += n
        if 'VGGNet/' in var.name :
            total_vggnet += n
        elif 'Convnet/' in var.name:
            total_convnet += n
        elif 'CALSTM_1' in var.name:
            total_calstm += n
            if 'multi_rnn_cell/cell_0' in var.name:
                total_lstm_0 += n
            elif 'multi_rnn_cell/cell_1' in var.name:
                total_lstm_1 += n
            elif 'AttentionModel' in var.name:
                total_att1 += n
        elif 'CALSTM_2' in var.name:
            total_calstm2 += n
            if 'multi_rnn_cell/cell_0' in var.name:
                total_lstm2_0 += n
            elif 'multi_rnn_cell/cell_1' in var.name:
                total_lstm2_1 += n
            elif 'AttentionModel' in var.name:
                total_att2 += n
        elif 'I2L_RNN/Output_Layer/' in var.name:
            total_output += n
        elif 'Initializer_MLP/' in var.name:
            total_init += n
        elif 'I2L_RNN/Embedding/Embedding_Matrix' in var.name:
            total_embedding += n
        else:
            assert False, 'unrecognized variable %s'%var
        logger.info('%s %s num_params = %d'%(var.name, K.int_shape(var), n) )

    logger.info( 'Total number of trainable params = %d'%total_n)
    if total_n > 0:
        logger.info( 'Vggnet: %d (%2.2f%%)'%(total_vggnet, total_vggnet*100./total_n))
        logger.info( 'Convnet: %d (%2.2f%%)'%(total_convnet, total_convnet*100./total_n))
        logger.info( 'Initializer: %d (%2.2f%%)'%(total_init, total_init*100./total_n))
        logger.info( 'CALSTM_1: %d (%2.2f%%)'%(total_calstm, total_calstm*100./total_n))
        logger.info( 'LSTM1_0: %d (%2.2f%%)'%(total_lstm_0, total_lstm_0*100./total_n))
        logger.info( 'LSTM1_1: %d (%2.2f%%)'%(total_lstm_1, total_lstm_1*100./total_n))
        logger.info( 'AttentionModel1: %d (%2.2f%%)'%(total_att1, total_att1*100./total_n))
        if total_calstm2 > 0:
            logger.info( 'CALSTM_2: %d (%2.2f%%)'%(total_calstm2, total_calstm2*100./total_n))
            logger.info( 'LSTM2_0: %d (%2.2f%%)'%(total_lstm2_0, total_lstm2_0*100./total_n))
            logger.info( 'LSTM2_1: %d (%2.2f%%)'%(total_lstm2_1, total_lstm2_1*100./total_n))
            logger.info( 'AttentionModel2: %d (%2.2f%%)'%(total_att2, total_att2*100./total_n))
        logger.info( 'Output Layer: %d (%2.2f%%)'%(total_output, total_output*100./total_n))
        logger.info( 'Embedding Matrix: %d (%2.2f%%)'%(total_embedding, total_embedding*100./total_n))

def make_standardized_step(hyper):
    B = hyper.data_reader_B
    def standardized_step(step):
        return (step * B) // 64
    return standardized_step
standardized_step = None


class Accumulator(dlc.Properties):
    def __init__(self):
        dlc.Properties.__init__(self)
        self.reset()
    def reset(self):
        for k in self.keys():
            del self[k]
    def append(self, d):
        for k, v in d.iteritems():
            if k not in self:
                self[k] = []
            self[k].append(v)
    def extend(self, d):
        for k, v in d.iteritems():
            if k not in self:
                self[k] = []
            self[k].extend(v)


class TFOpNames(dlc.Properties):
    def __init__(self, keys, val):
        dlc.Properties.__init__(self)
        object.__setattr__(self, '_keys_tuple', tuple(keys))
        self._reset_vals(val)
        self.seal()

    def keys(self):
        return object.__getattribute__(self, '_keys_tuple')

    def _reset_vals(self, val):
        for key in self.keys():
            self[key] = val

    def _set_vals(self, vals):
        assert len(vals) == len(self.keys())
        keys = self.keys()
        for i in range(len(vals)):
            self[keys[i]] = vals[i]

    def _op_tuple(self, op_dict):
        l = []
        for name in self.keys():
            l.append(op_dict[name])
        return tuple(l)

    def run_ops(self, session, op_dict, feed_dict={}):
        self._set_vals(session.run(self._op_tuple(op_dict), feed_dict=feed_dict))


class TFRun(TFOpNames):
    def __init__(self, session, ops, op_names, reset_val):
        TFOpNames.__init__(self, op_names, reset_val)
        object.__setattr__(self, '_ops_dict', ops)
        object.__setattr__(self, '_session', session)

    @property
    def ops(self):
        return object.__getattribute__(self, '_ops_dict')

    @property
    def session(self):
        return object.__getattribute__(self, '_session')

    def run_ops(self):
        TFOpNames.run_ops(self, self.session, self.ops)


def main(raw_data_folder,
          vgg16_folder,
          args,
          hyper):
    """
    Start training the model.
    """
    dtc.initialize(args.raw_data_dir, hyper)
    global logger
    logger = hyper.logger
    global standardized_step
    standardized_step = make_standardized_step(hyper)

    graph = tf.Graph()
    with graph.as_default():
        if hyper.build_image_context == 1:
            train_it, eval_it = create_imagenet_iterators(raw_data_folder,
                                                          hyper,
                                                          args)
        elif hyper.build_image_context == 2:
            train_it, eval_it = create_BW_image_iterators(raw_data_folder,
                                                          hyper,
                                                          args)
        else:
            train_it, eval_it = create_context_iterators(raw_data_folder,
                                                         vgg16_folder,
                                                         hyper,
                                                         args)

        qrs = []
        ##### Training Graphs
        train_tower_ops = []; train_ops = None
        trainable_vars_n = 0
        toplevel_var_scope = tf.get_variable_scope()
        reuse_vars = False
        with tf.name_scope('Training'):
            tf_train_step = tf.get_variable('global_step', dtype=hyper.int_type, trainable=False, initializer=0)
            if hyper.optimizer == 'adam':
                opt = tf.train.AdamOptimizer(learning_rate=hyper.adam_alpha, beta1=hyper.adam_beta1, beta2=hyper.adam_beta2)
            else:
                raise Exception('Unsupported optimizer - %s - configured.' % (hyper.optimizer,))

            if args.doTrain:
                with tf.variable_scope('InputQueue'):
                    train_q = tf.FIFOQueue(hyper.input_queue_capacity, train_it.out_tup_types)
                    tf_enqueue_train_queue = train_q.enqueue_many(train_it.get_pyfunc_with_split(hyper.num_towers))
                    tf_close_train_queue = train_q.close(cancel_pending_enqueues=True)
                for i in range(hyper.num_gpus):
                    for j in range(hyper.towers_per_gpu):
                        with tf.name_scope('gpu_%d'%i + ('/tower_%d'%j if hyper.towers_per_gpu > 1 else '')):
                            with tf.device('/gpu:%d'%i):
                                model = Im2LatexModel(hyper, train_q, opt=opt, reuse=reuse_vars)
                                train_tower_ops.append(model.build_training_tower())
                                if not reuse_vars:
                                    trainable_vars_n = num_trainable_vars()  # 8544670 or 8547670
                                    hyper.logger.info('Num trainable variables = %d', trainable_vars_n)
                                    reuse_vars = True
                                    ## assert trainable_vars_n == 8547670 if hyper.use_peephole else 8544670
                                    ## assert trainable_vars_n == 23261206 if hyper.build_image_context
                                else:
                                    assert num_trainable_vars() == trainable_vars_n, 'trainable_vars %d != expected %d'%(num_trainable_vars(), trainable_vars_n)
                train_ops = sync_training_towers(hyper, train_tower_ops, tf_train_step, optimizer=opt)
        if args.doTrain:
            qr1 = tf.train.QueueRunner(train_q, [tf_enqueue_train_queue], cancel_op=[tf_close_train_queue])
            qrs.append(qr1)

        ##### Validation/Testing Graph
        eval_tower_ops = []; eval_ops = None
        if eval_it:  # and (args.doTrain or args.doValidate):
            with tf.name_scope('Validation' if not args.doTest else 'Testing'):
                hyper_predict = hyper_params.make_hyper(args.copy().updated({'dropout':None}))
                with tf.variable_scope('InputQueue'):
                    eval_q = tf.FIFOQueue(hyper.input_queue_capacity, eval_it.out_tup_types)
                    enqueue_op2 = eval_q.enqueue_many(eval_it.get_pyfunc_with_split(hyper.num_towers))
                    close_queue2 = eval_q.close(cancel_pending_enqueues=True)
                for i in range(args.num_gpus):
                    for j in range(args.towers_per_gpu):
                        with tf.name_scope('gpu_%d' % i + ('/tower_%d' % j if hyper.towers_per_gpu > 1 else '')):
                            with tf.device('/gpu:%d'%i):
                                if hyper.build_scanning_RNN:
                                    model_predict = Im2LatexModel(hyper_predict,
                                                                  eval_q,
                                                                  reuse=reuse_vars)
                                    eval_tower_ops.append(model_predict.build_training_tower())
                                else:
                                    model_predict = Im2LatexModel(hyper_predict,
                                                                  eval_q,
                                                                  seq2seq_beam_width=hyper.seq2seq_beam_width,
                                                                  reuse=reuse_vars)
                                    eval_tower_ops.append(model_predict.build_testing_tower())
                                if not reuse_vars:
                                    trainable_vars_n = num_trainable_vars()
                                    reuse_vars = True
                                else:
                                    assert num_trainable_vars() == trainable_vars_n, 'trainable_vars %d != expected %d' % (
                                        num_trainable_vars(), trainable_vars_n)

                hyper.logger.info('Num trainable variables = %d', num_trainable_vars())
                assert num_trainable_vars() == trainable_vars_n, 'num_trainable_vars(%d) != %d'%(num_trainable_vars(), trainable_vars_n)
                if hyper.build_scanning_RNN:
                    eval_ops = sync_training_towers(hyper, eval_tower_ops, global_step=None, run_tag='validation' if not args.doTest else 'testing')
                else:
                    eval_ops = sync_testing_towers(hyper, eval_tower_ops, run_tag='validation' if not args.doTest else 'testing')
            qr2 = tf.train.QueueRunner(eval_q, [enqueue_op2], cancel_op=[close_queue2])
            qrs.append(qr2)

        # ##### Training Accuracy Graph
        # if (args.make_training_accuracy_graph):
        #     with tf.name_scope('TrainingAccuracy'):
        #         hyper_predict2 = hyper_params.make_hyper(args.copy().updated({'dropout':None}))
        #         with tf.device('/gpu:1'):
        #             model_predict2 = Im2LatexModel(hyper_predict, hyper.seq2seq_beam_width, reuse=True)
        #             tr_acc_ops = model_predict2.test()
        #         with tf.variable_scope('QueueOps'):
        #             enqueue_op3 = tr_acc_ops.inp_q.enqueue_many(tr_acc_it.get_pyfunc_with_split(hyper.num_towers))
        #             close_queue3 = tr_acc_ops.inp_q.close(cancel_pending_enqueues=True)
        #         assert(num_trainable_vars() == trainable_vars_n)
        #     qr3 = tf.train.QueueRunner(tr_acc_ops.inp_q, [enqueue_op3], cancel_op=[close_queue3])
        # else:
        tr_acc_ops = None

        coord = tf.train.Coordinator()
        training_logic = TrainingLogic(args, coord, train_it, eval_it)

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

            enqueue_threads = []
            for qr in qrs:
                enqueue_threads.extend(qr.create_threads(session, coord=coord, start=True))
            # enqueue_threads = qr1.create_threads(session, coord=coord, start=True)
            # enqueue_threads.extend(qr2.create_threads(session, coord=coord, start=True))
            # if args.make_training_accuracy_graph:
            #     enqueue_threads.extend(qr3.create_threads(session, coord=coord, start=True))
            logger.info('Created enqueue threads')

            saver = tf.train.Saver(max_to_keep=args.num_snapshots, pad_step_number=True, save_relative_paths=True)
            if args.restore_from_checkpoint:
                latest_checkpoint = tf.train.latest_checkpoint(args.logdir, latest_filename='checkpoints_list')
                logger.info('Restoring session from checkpoint %s', latest_checkpoint)
                saver.restore(session, latest_checkpoint)
                step = tf_train_step.eval()
                logger.info('Restored session from checkpoint %s at step %d', latest_checkpoint, step)
            else:
                tf.global_variables_initializer().run()
                step = 0
                logger.info('Starting a new session')

            # Ensure that everything was initialized
            assert len(tf.report_uninitialized_variables().eval()) == 0

            try:
                start_time = time.time()
                ############################# Training (with Validation) Cycle ##############################
                ops_accum = (
                    'train',
                    'predicted_ids_list',
                    'predicted_lens',
                    'y_ctc_list',
                    'ctc_len',
                    'ctc_ed',
                    'log_likelihood',
                    'ctc_loss',
                    'alpha_penalty',
                    'cost',
                    'mean_norm_ase',
                    'mean_norm_aae',
                    'beta_mean',
                    'beta_std_dev',
                    'pred_len_ratio',
                    'num_hits',
                    'reg_loss',
                    'scan_len',
                    'bin_len',
                    'global_step'
                )

                ops_log = (
                    'y_s_list',
                    'predicted_ids_list',
                    'alpha',
                    'beta',
                    'image_name_list',
                    'x_s_list',
                    'tb_step_logs'
                )
                if args.doTrain:
                    logger.info('Starting training')
                    accum = Accumulator()
                    while not training_logic.should_stop():
                        step_start_time = time.time()
                        step += 1
                        doLog = training_logic.do_log(step)
                        if not doLog:
                            batch_ops = TFOpNames(ops_accum, None)
                        else:
                            batch_ops = TFOpNames(ops_accum + ops_log, None)

                        batch_ops.run_ops(session, train_ops)
                        assert step == batch_ops.global_step, 'Step(%d) and global-step(%d) fell out of sync ! :((' % (step, batch_ops.global_step)

                        # Accumulate Metrics
                        accum.append(batch_ops)
                        train_time = time.time()-step_start_time
                        bleu = sentence_bleu_scores(hyper, batch_ops.predicted_ids_list, batch_ops.predicted_lens, batch_ops.y_ctc_list, batch_ops.ctc_len)
                        accum.extend({'bleu_scores': bleu})
                        accum.extend({
                            'sq_predicted_ids': squashed_seq_list(hyper, batch_ops.predicted_ids_list, batch_ops.predicted_lens),
                            'sq_y_ctc':         trimmed_seq_list(hyper, batch_ops.y_ctc_list, batch_ops.ctc_len)
                        })
                        accum.append({'train_time': train_time})

                        if doLog:
                            logger.info('Step %d', step)
                            train_time_per100 = np.mean(train_time) * 100. / (hyper.data_reader_B)

                            with dtc.Storer(args, 'training', step) as storer:
                                storer.write('predicted_ids', batch_ops.predicted_ids_list, np.int16)
                                storer.write('y', batch_ops.y_s_list, np.int16)
                                storer.write('alpha', batch_ops.alpha, np.float32, batch_axis=1)
                                storer.write('beta', batch_ops.beta, np.float32, batch_axis=1)
                                storer.write('image_name', batch_ops.image_name_list, dtype=np.unicode_)
                                storer.write('ed', batch_ops.ctc_ed, np.float32)
                                storer.write('bleu', bleu, np.float32)
                                storer.write('bin_len', batch_ops.bin_len, np.float32)
                                storer.write('scan_len', batch_ops.scan_len, np.float32)
                                storer.write('x', batch_ops.x_s_list, np.float32)

                            # per-step metrics
                            tf_sw.add_summary(batch_ops.tb_step_logs, global_step=standardized_step(step))
                            tf_sw.flush()

                            # aggregate metrics
                            agg_bleu2 = dlc.corpus_bleu_score(accum.sq_predicted_ids, accum.sq_y_ctc)
                            tb_agg_logs = session.run(train_ops.tb_agg_logs, feed_dict={
                                train_ops.ph_train_time: train_time_per100,
                                train_ops.ph_bleu_scores: accum.bleu_scores,
                                train_ops.ph_bleu_score2: agg_bleu2,
                                train_ops.ph_ctc_eds: accum.ctc_ed,
                                train_ops.ph_loglosses: accum.log_likelihood,
                                train_ops.ph_ctc_losses: accum.ctc_loss,
                                train_ops.ph_alpha_penalties: accum.alpha_penalty,
                                train_ops.ph_costs: accum.cost,
                                train_ops.ph_mean_norm_ases: accum.mean_norm_ase,
                                train_ops.ph_mean_norm_aaes: accum.mean_norm_aae,
                                train_ops.ph_beta_mean: accum.beta_mean,
                                train_ops.ph_beta_std_dev: accum.beta_std_dev,
                                train_ops.ph_pred_len_ratios: accum.pred_len_ratio,
                                train_ops.ph_num_hits: accum.num_hits,
                                train_ops.ph_reg_losses: accum.reg_loss,
                                train_ops.ph_scan_lens: accum.scan_len
                            })

                            tf_sw.add_summary(tb_agg_logs, global_step=standardized_step(step))
                            tf_sw.flush()

                            doValidate, num_validation_batches, do_save = training_logic.do_validate(step, (agg_bleu2 if (args.valid_epochs <= 0) else None))
                            if do_save:
                                saver.save(session, args.logdir + '/snapshot', global_step=step,
                                           latest_filename='checkpoints_list')
                            if doValidate:
                                if hyper.build_scanning_RNN:
                                    accuracy_res = evaluate_scanning_RNN(args, hyper, session, eval_ops, ops_accum,
                                                                         ops_log, step, num_validation_batches, tf_sw,
                                                                         training_logic)
                                else:
                                    accuracy_res = evaluate(
                                        session,
                                        dlc.Properties({'eval_ops':eval_ops}),
                                        dlc.Properties({'train_it':train_it, 'eval_it':eval_it}),
                                        hyper,
                                        args,
                                        step,
                                        num_validation_batches,
                                        tf_sw,
                                        training_logic)
                                logger.info('Time for %d steps, elapsed = %f, training-time-per-100 = %f, validation-time-per-100 = %f'%(
                                    step,
                                    time.time()-start_time,
                                    train_time_per100,
                                    accuracy_res.eval_time_per100))
                            else:
                                logger.info('Time for %d steps, elapsed = %f, training-time-per-100 = %f'%(
                                    step,
                                    time.time()-start_time,
                                    train_time_per100))

                            # Reset Metrics
                            accum.reset()

                ############################# Validation/Testing Only ##############################
                elif args.doValidate or args.doTest:
                    logger.info('Starting %s Cycle'%('Validation' if args.doValidate else 'Testing',))
                    if hyper.build_scanning_RNN:
                        evaluate_scanning_RNN(args, hyper, session, eval_ops, ops_accum,
                                              ops_log, step,
                                              eval_it.epoch_size,
                                              tf_sw,
                                              training_logic)
                    else:
                        evaluate(session,
                                 dlc.Properties({'eval_ops': eval_ops}),
                                 dlc.Properties({'train_it': train_it, 'eval_it': eval_it}),
                                 hyper,
                                 args,
                                 step,
                                 eval_it.epoch_size,
                                 tf_sw,
                                 training_logic)


            except tf.errors.OutOfRangeError, StopIteration:
                logger.info('Done training -- epoch limit reached')
            except Exception as e:
                logger.info( '***************** Exiting with exception: *****************\n%s'%e)
                coord.request_stop(e)
            finally:
                logger.info('Elapsed time for %d steps = %f'%(step, time.time()-start_time))
                coord.request_stop()
                coord.join(enqueue_threads)


def sentence_bleu_scores(hyper, pred_ar_list, pred_lens, target_ar_list, target_lens):
    """
    :param hyper:
    :param pred_ar_list: list of np-array of predicted word/id-sequences. shape = [(B,T), ...]
    :param pred_lens: np-array of integers representing lengths of the predicted sequences. shape = (num_towers*B,)
    :param target_ar_list: list of np-array of target word/id-sequences. shape = [(B,T), ...]
    :param target_lens: np-array of integers representing lengths of the target sequences. shape = (num_towers*B,)
    :return:
    """
    assert len(pred_ar_list) == len(target_ar_list)
    num = reduce(lambda b, ar: b+len(ar), pred_ar_list, 0)
    assert num == len(pred_lens)
    logger.debug('Computing sentence_bleu_scores for %d sentences', num)
    bleus = []
    n = 0
    for i in range(len(pred_ar_list)):
        _n = len(pred_ar_list[i])
        assert len(target_ar_list[i]) == _n
        p_lens = pred_lens[n:n+_n]
        t_lens = target_lens[n:n+_n]
        bleus.extend(dlc.sentence_bleu_scores(pred_ar_list[i], p_lens,
                                                     target_ar_list[i], t_lens,
                                                     space_token=hyper.SpaceTokenID,
                                                     blank_token=hyper.NullTokenID,
                                                     eos_token=hyper.NullTokenID))
        n += _n
    return np.asarray(bleus)


def squash_and_concat(seq_batch_list, seq_len_batch, remove_val1=None, remove_val2=None, eos_token=None):
    """
    Gathers all word/id sequences into one list of size num_towers*B. Optionally Squashes and trims the sequences.
    :param seq_batch_list: list of np-array of word/id-sequences. shape = [(B,T), ...]
    :param seq_len_batch: np-array of integers representing lengths of the above sequences. shape = (num_towers*B,)
    :param remove_val1:
    :param remove_val2:
    :param eos_token:
    :return: A list of word/id lists
    """
    num = reduce(lambda b, ar: b+len(ar), seq_batch_list, 0)
    assert num == len(seq_len_batch)
    seq_list = []
    n = 0
    for i, seq_batch in enumerate(seq_batch_list):
        _n = len(seq_batch)
        seq_lens = seq_len_batch[n:n+_n]
        _seq_list = dlc.squashed_seq_list(seq_batch, seq_lens,
                                          remove_val1=remove_val1,
                                          remove_val2=remove_val2,
                                          eos_token=eos_token)
        seq_list.extend(_seq_list)
        n += _n

    return seq_list


def squashed_seq_list(hyper, seq_batch_list, seq_len_batch):
    """
    Squashes and trims word/id sequences and puts them all into one list of size num_towers*B
    :param hyper: HyperParams
    :param seq_batch_list: list of np-array of word/id-sequences. shape = [(B,T), ...]
    :param seq_len_batch: np-array of integers representing lengths of the above sequences. shape = (num_towers*B,)
    :return: a list of word/id lists
    """
    return squash_and_concat(seq_batch_list, seq_len_batch,
                             remove_val1=hyper.SpaceTokenID,
                             remove_val2=hyper.CTCBlankTokenID,
                             eos_token=hyper.NullTokenID)


def trimmed_seq_list(hyper, seq_batch_list, seq_len_batch):
    """
    Trims word/id sequences and puts them all into one list of size num_towers*B
    :param hyper: HyperParams
    :param seq_batch_list: list of np-array of word/id-sequences. shape = [(B,T), ...]
    :param seq_len_batch: np-array of integers representing lengths of the above sequences. shape = (num_towers*B,)
    :return: a list of word/id lists
    """
    return squash_and_concat(seq_batch_list, seq_len_batch, eos_token=hyper.NullTokenID)


def ids2str_list(target_ids, predicted_ids, hyper):
    """
    Same as id2str, except this works on multiple batches. The arguments are lists of numpy arrays
    instead of straight numpy arrays as in the case of id2str.2017-12-11 12:16:58,468 - INFO - Elapsed time for 6 steps = 142.974565

    """
    l = []
    for i in range(len(target_ids)):
        l.append(ids2str(target_ids[i], predicted_ids[i], hyper))
    return l


def ids2str(target_ids, predicted_ids, hyper):
    """
    Args:
        target_ids: Numpy array of shape (B,T)
        predicted_ids: Numpy array of same shape as target_ids
    """
    separator = None #"\t" if not hyper.use_ctc_loss else None
    target_str = np.expand_dims(dtc.seq2str(target_ids, 'Target:', separator), axis=1)
    predicted_str = np.expand_dims(dtc.seq2str(predicted_ids, 'Prediction:', separator),axis=1)
    return np.concatenate((predicted_str, target_str), axis=1)


def ids2str3D_list(ids_list, hyper):
    strs = []
    for i in range(len(ids_list)):
        strs.append(ids2str3D(ids_list[i], hyper))
    return strs


def ids2str3D(ids, hyper):
    """
    Args:
        target_ids: Numpy array of shape (B,T)
        predicted_ids: Numpy array of same shape as target_ids
    """
    separator = None #"\t" if not hyper.use_ctc_loss else None
    strs = []
    for i in range(len(ids)):
        strs.append(dtc.seq2str(ids[i], 'Sample %d:'%i, separator))
    return strs


class TrainingLogic(object):
    full_validation_steps = [3000]
    fv_score = 0.9
    stop_training = False
    validate_next = False
    validate_then_stop = False

    def __init__(self, args, coord, train_it, eval_it):
        self._coord = coord
        self._args = args
        self._train_it = train_it
        self._eval_it = eval_it
        self._setup_signal_handler()

    def _setup_signal_handler(self):
        def validate_then_stop(signum, frame):
            if self.validate_then_stop:
                logger.critical('Received signal %d. Unsetting validate_then_stop flag.', signum)
                self.validate_then_stop = False
            else:
                logger.critical('Received signal %d. Will run validation cycle then stop.', signum)
                self.validate_then_stop = True

        def validate_next(signum, frame):
            if self.validate_next:
                logger.critical('Received signal %d. Unsetting validate_next flag.', signum)
                self.validate_next = False
            else:
                logger.critical('Received signal %d. Will run validation cycle at next step.', signum)
                self.validate_next = True

        def stop_training(signum, frame):
            if self.stop_training:
                logger.critical('Received signal %d. Unsetting stop_training flag.', signum)
                self.stop_training = False
            else:
                logger.critical('Received signal %d. Stopping training.', signum)
                self.stop_training = True

        signal.signal(signal.SIGTERM, validate_then_stop)
        signal.signal(signal.SIGINT, validate_next)
        signal.signal(signal.SIGUSR1, stop_training)

    def _set_flags_after_validation(self):
        if self.validate_then_stop:
            self.validate_then_stop = False
            self.stop_training = True
        if self.validate_next:
            self.validate_next = False

    def should_stop(self):
        return self._coord.should_stop() or self.stop_training

    def do_validate(self, step, score=None):
        if self._eval_it is None:
            doValidate = do_save = False
            num_eval_batches = 0
        elif self._args.doValidate:  # Validation-only run
            doValidate = do_save = True
            num_eval_batches = self._eval_it.epoch_size
        elif self.validate_next or self.validate_then_stop:
            doValidate = do_save = True
            num_eval_batches = self._eval_it.epoch_size
        # elif self._args.valid_epochs <= 0:  # smart validation
        #     assert score is not None, 'score must be supplied if eval_epochs <= 0'
        elif (self._args.valid_epochs <= 0) and (score is not None):  # smart validation
            if (score > self.fv_score) and ((step - self.full_validation_steps[-1]) >= (0.25 * self._train_it.epoch_size)):
                doValidate = True
                num_eval_batches = self._eval_it.epoch_size
                do_save = True
                self.full_validation_steps.append(step)
                self.fv_score = score
                logger.info('TrainingLogic: fv_score set to %f at step %d'%(score, step))
            elif (score > 0.9) and ((step - self.full_validation_steps[-1]) >= (1 * self._train_it.epoch_size)):
                doValidate = do_save = True
                num_eval_batches = self._eval_it.epoch_size
                self.full_validation_steps.append(step)
                logger.info('TrainingLogic: Running full validation at score %f at step %d' % (score, step))
            elif (step - self.full_validation_steps[-1]) >= (4 * self._train_it.epoch_size):
                doValidate = do_save = True
                num_eval_batches = self._eval_it.epoch_size
                self.full_validation_steps.append(step)
                logger.info('TrainingLogic: Running full validation at score %f at step %d' % (score, step))
            else:
                doValidate = do_save = False
                num_eval_batches = 0
        else:
            epoch_frac = self._args.valid_epochs if (self._args.valid_epochs is not None) else 1
            period = int(epoch_frac * self._train_it.epoch_size)
            doValidate = do_save = (step % period == 0) or (step == self._train_it.max_steps)
            num_eval_batches = self._eval_it.epoch_size if doValidate else 0

        # Save snapshot every 2 epochs so that we may restore runs at the beginning of full epochs because that's
        # when the batch-iterators reshuffle the samples.
        if step % int(2 * self._train_it.epoch_size) == 0:
            logger.info('Saving state at epoch boundary (step=%d)'%step)
            do_save = True

        return doValidate, num_eval_batches, do_save

    def do_log(self, step):
        do_log = (step % self._args.print_steps == 0) or (step == self._train_it.max_steps) or self.validate_then_stop or self.validate_next
        # if self._args.valid_epochs <= 0:  # do_validate synchronizes with do_log
        #     return do_log
        # else:
        validate, _, _ = self.do_validate(step, None)
        return do_log or validate


def format_ids(predicted_ids, target_ids):
    np.apply_along_axis


# Note: This function has not been tested after it was retrofitted to dump all samples if args.save_all_eval==True
def evaluate_scanning_RNN(args, hyper, session, ops, ops_accum, ops_log, tr_step, num_steps, tf_sw, training_logic):
    training_logic._set_flags_after_validation()
    start_time = time.time()
    ############################# Validation Or Test Cycle ##############################
    hyper.logger.info('validation cycle starting at step %d for %d steps', tr_step, num_steps)
    accum = Accumulator()
    print_batch_num = np.random.randint(1, num_steps + 1) if not args.save_all_eval else None
    for batch in range(1, 1+num_steps):
        step_start_time = time.time()
        doLog = (print_batch_num == batch)
        if doLog or args.save_all_eval:
            batch_ops = TFRun(session, ops, ops_accum + ops_log, None)
        else:
            batch_ops = TFRun(session, ops, ops_accum, None)

        batch_ops.run_ops()

        ## Accumulate Metrics
        accum.append(batch_ops)
        batch_time = time.time() - step_start_time
        bleu = sentence_bleu_scores(hyper,
                                    batch_ops.predicted_ids_list,
                                    batch_ops.predicted_lens,
                                    batch_ops.y_ctc_list,
                                    batch_ops.ctc_len)
        accum.extend({'bleu_scores': bleu})
        accum.extend({
            'sq_predicted_ids': squashed_seq_list(hyper, batch_ops.predicted_ids_list, batch_ops.predicted_lens),
            'sq_y_ctc': trimmed_seq_list(hyper, batch_ops.y_ctc_list, batch_ops.ctc_len),
            'predicted_ids': batch_ops.predicted_ids_list,
            'y': batch_ops.y_s_list,
            'alpha': batch_ops.alpha,
            'beta': batch_ops.beta,
            'image_name': batch_ops.image_name_list,
            'ctc_ed': batch_ops.ctc_ed,
            'bin_len': batch_ops.bin_len,
            'scan_len': batch_ops.scan_len,
            'x': batch_ops.x_s_list
        })
        accum.append({'batch_time': batch_time})

        if doLog:
            with dtc.Storer(args, 'test' if args.doTest else 'validation', tr_step) as storer:
                storer.write('predicted_ids', batch_ops.predicted_ids_list, np.int16)
                storer.write('y', batch_ops.y_s_list, np.int16)
                storer.write('alpha', batch_ops.alpha, np.float32, batch_axis=1)
                storer.write('beta', batch_ops.beta, np.float32, batch_axis=1)
                storer.write('image_name', batch_ops.image_name_list, dtype=np.unicode_)
                storer.write('ed', batch_ops.ctc_ed, np.float32)
                storer.write('bleu', bleu, np.float32)
                storer.write('bin_len', batch_ops.bin_len, np.float32)
                storer.write('scan_len', batch_ops.scan_len, np.float32)
                storer.write('x', batch_ops.x_s_list, np.float32)

    if args.save_all_eval:
        with dtc.Storer(args, 'test' if args.doTest else 'validation', tr_step) as storer:
            storer.write('predicted_ids', accum.predicted_ids, np.int16)
            storer.write('y', accum.y, np.int16)
            storer.write('alpha', accum.alpha, np.float32, batch_axis=1)
            storer.write('beta', accum.beta, np.float32, batch_axis=1)
            storer.write('image_name', accum.image_name, dtype=np.unicode_)
            storer.write('ed', accum.ctc_ed, np.float32)
            storer.write('bleu', accum.bleu_scores, np.float32)
            storer.write('bin_len', accum.bin_len, np.float32)
            storer.write('scan_len', accum.scan_len, np.float32)
            storer.write('x', accum.x, np.float32)

    # Calculate aggregated validation metrics
    eval_time_per100 = np.mean(batch_time) * 100. / hyper.data_reader_B
    tb_agg_logs = session.run(ops.tb_agg_logs, feed_dict={
        ops.ph_train_time: eval_time_per100,
        ops.ph_bleu_scores: accum.bleu_scores,
        ops.ph_bleu_score2: dlc.corpus_bleu_score(accum.sq_predicted_ids, accum.sq_y_ctc),
        ops.ph_ctc_eds: accum.ctc_ed,
        ops.ph_loglosses: accum.log_likelihood,
        ops.ph_ctc_losses: accum.ctc_loss,
        ops.ph_alpha_penalties: accum.alpha_penalty,
        ops.ph_costs: accum.cost,
        ops.ph_mean_norm_ases: accum.mean_norm_ase,
        ops.ph_mean_norm_aaes: accum.mean_norm_aae,
        ops.ph_beta_mean: accum.beta_mean,
        ops.ph_beta_std_dev: accum.beta_std_dev,
        ops.ph_pred_len_ratios: accum.pred_len_ratio,
        ops.ph_num_hits: accum.num_hits,
        ops.ph_reg_losses: accum.reg_loss,
        ops.ph_scan_lens: accum.scan_len
    })
    tf_sw.add_summary(tb_agg_logs, global_step=standardized_step(tr_step))
    tf_sw.flush()
    return dlc.Properties({'eval_time_per100': eval_time_per100})


def evaluate(session, ops, batch_its, hyper, args, step, num_steps, tf_sw, training_logic):
    training_logic._set_flags_after_validation()
    eval_start_time = time.time()

    eval_ops = ops.eval_ops
    batch_it = batch_its.eval_it
    batch_size = batch_it.batch_size
    # Print a batch randomly
    print_batch_num = np.random.randint(1, num_steps+1) if not args.save_all_eval else None
    accum = Accumulator()
    n = 0
    hyper.logger.info('evaluation cycle starting at step %d for %d steps', step, num_steps)
    ops_base = ('top1_len_ratio',
                'top1_mean_ed',
                'top1_accuracy',
                'top1_num_hits',
                'top1_ids_list',
                'top1_lens',
                'y_ctc_list',
                'ctc_len'
                ) + (('bok_ids_list', 'bok_seq_lens', 'bok_accuracy', 'bok_mean_ed') if args.log_bok else tuple())
    ops_addl = ('y_s_list',
                'top1_alpha_list',
                'top1_beta_list',
                'image_name_list',
                'top1_ed') + (('bok_ed',) if args.log_bok else tuple())

    while n < num_steps:
        n += 1
        if (n == print_batch_num) or args.save_all_eval:
            batch_ops = TFRun(session, ops.eval_ops, ops_base + ops_addl, None)
            # (l, mean_ed, accuracy, num_hits, top1_ids_list, top1_lens, y_ctc_list, ctc_len, y_s_list, top1_alpha_list,
            #  top1_beta_list, image_name_list, top1_ed) = session.run(
            #     (
            #         eval_ops.top1_len_ratio,
            #         eval_ops.top1_mean_ed,
            #         eval_ops.top1_accuracy,
            #         eval_ops.top1_num_hits,
            #         eval_ops.top1_ids_list,
            #         eval_ops.top1_lens,
            #         eval_ops.y_ctc_list,
            #         eval_ops.ctc_len,
            #         eval_ops.y_s_list,
            #         eval_ops.top1_alpha_list,
            #         eval_ops.top1_beta_list,
            #         eval_ops.image_name_list,
            #         eval_ops.top1_ed
            #     ))
        else:
            batch_ops = TFRun(session, ops.eval_ops, ops_base, None)
            # l, mean_ed, accuracy, num_hits, top1_ids_list, top1_lens, y_ctc_list, ctc_len, bok_ids_list, bok_seq_lens = session.run((
            #                     eval_ops.top1_len_ratio,
            #                     eval_ops.top1_mean_ed,
            #                     eval_ops.top1_accuracy,
            #                     eval_ops.top1_num_hits,
            #                     eval_ops.top1_ids_list,
            #                     eval_ops.top1_lens,
            #                     eval_ops.y_ctc_list,
            #                     eval_ops.ctc_len,
            #                     eval_ops.bok_ids_list,
            #                     eval_ops.bok_seq_lens
            #                     ))
            # y_s_list = top1_alpha_list = top1_beta_list = image_name_list = top1_ed = None

        batch_ops.run_ops()
        if args.save_all_eval:
            accum.extend({'y': batch_ops.y_s_list,
                          'top1_ids': batch_ops.top1_ids_list,
                          'alpha': batch_ops.top1_alpha_list,
                          'beta': batch_ops.top1_beta_list,
                          'image_name': batch_ops.image_name_list
                          })
            accum.append({'ed': batch_ops.top1_ed})
            if args.log_bok:
                accum.extend({'bok_ids': batch_ops.bok_ids_list})
                accum.append({'bok_ed': batch_ops.bok_ed})

        bleu = sentence_bleu_scores(hyper, batch_ops.top1_ids_list, batch_ops.top1_lens, batch_ops.y_ctc_list, batch_ops.ctc_len)
        accum.append({'bleus': bleu})
        accum.extend({'sq_predicted_ids': squashed_seq_list(hyper, batch_ops.top1_ids_list, batch_ops.top1_lens)})
        accum.extend({'trim_target_ids': trimmed_seq_list(hyper, batch_ops.y_ctc_list, batch_ops.ctc_len)})
        accum.append({'len_ratio': batch_ops.top1_len_ratio})
        accum.append({'mean_eds': batch_ops.top1_mean_ed})
        accum.append({'accuracies': batch_ops.top1_accuracy})
        accum.append({'hits': batch_ops.top1_num_hits})
        if args.log_bok:
            bok_bleu = sentence_bleu_scores(hyper, batch_ops.bok_ids_list, batch_ops.bok_seq_lens, batch_ops.y_ctc_list, batch_ops.ctc_len)
            accum.extend({'sq_bok_ids': squashed_seq_list(hyper, batch_ops.bok_ids_list, batch_ops.bok_seq_lens)})
            accum.append({'bok_bleus': bok_bleu})
            accum.append({'bok_mean_eds': batch_ops.bok_mean_ed})
            accum.append({'bok_accuracies': batch_ops.bok_accuracy})

        if n == print_batch_num:
            # logger.info('############ RANDOM VALIDATION BATCH %d ############', n)
            # logger.info('prediction mean_ed=%f', mean_ed)
            # logger.info('prediction accuracy=%f', accuracy)
            # logger.info('prediction hits=%d', num_hits)
            # bleu = sentence_bleu_scores(hyper, top1_ids_list, top1_lens, y_ctc_list, ctc_len)

            with dtc.Storer(args, 'test' if args.doTest else 'validation', step) as storer:
                storer.write('predicted_ids', batch_ops.top1_ids_list, np.int16)
                if args.log_bok:
                    storer.write('bok_ids', batch_ops.bok_ids_list, np.int16)
                    storer.write('bok_bleu', bok_bleu, dtype=np.float32)
                    storer.write('bok_ed', batch_ops.bok_ed, dtype=np.float32)
                storer.write('y', batch_ops.y_s_list, np.int16)
                storer.write('alpha', batch_ops.top1_alpha_list, dtype=np.float32, batch_axis=1)
                storer.write('beta', batch_ops.top1_beta_list, dtype=np.float32, batch_axis=1)
                storer.write('image_name', batch_ops.image_name_list, dtype=np.unicode_)
                storer.write('ed', batch_ops.top1_ed, dtype=np.float32)
                storer.write('bleu', bleu, dtype=np.float32)
            # logger.info('############ END OF RANDOM VALIDATION BATCH ############')

    agg_bleu2 = dlc.corpus_bleu_score(accum.sq_predicted_ids, accum.trim_target_ids)
    if args.log_bok:
        agg_bok_bleu2 = dlc.corpus_bleu_score(accum.sq_bok_ids, accum.trim_target_ids)
    eval_time_per100 = (time.time() - eval_start_time) * 100. / (num_steps * batch_size)

    if args.save_all_eval:
        assert len(accum.y) == len(accum.top1_ids) == len(accum.alpha) == len(accum.beta) == len(accum.image_name)
        assert len(accum.bleus) == len(accum.ed), 'len bleus = %d, len ed = %d' % (len(accum.bleus), len(accum.ed),)
        with dtc.Storer(args, 'test' if args.doTest else 'validation', step) as storer:
            storer.write('predicted_ids', accum.top1_ids, np.int16)
            storer.write('y', accum.y, np.int16)
            storer.write('alpha', accum.alpha, dtype=np.float32, batch_axis=1)
            storer.write('beta', accum.beta, dtype=np.float32, batch_axis=1)
            storer.write('image_name', accum.image_name, dtype=np.unicode_)
            storer.write('ed', accum.ed, dtype=np.float32)
            storer.write('bleu', accum.bleus, dtype=np.float32)
            if args.log_bok:
                storer.write('bok_ids', accum.bok_ids, np.int16)
                storer.write('bok_bleu', accum.bok_bleus, dtype=np.float32)
                storer.write('bok_ed', accum.bok_ed, dtype=np.float32)

    summary_ops = TFOpNames(('logs_agg_top1', 'logs_agg_bok') if args.log_bok else ('logs_agg_top1',), None)
    feed_dict = {
        eval_ops.ph_top1_len_ratio: accum.len_ratio,
        eval_ops.ph_edit_distance: accum.mean_eds,
        eval_ops.ph_num_hits: accum.hits,
        eval_ops.ph_accuracy: accum.accuracies,
        eval_ops.ph_valid_time: eval_time_per100,
        eval_ops.ph_bleus: accum.bleus,
        eval_ops.ph_bleu2: agg_bleu2,
        eval_ops.ph_full_validation: 1 if (num_steps == batch_it.epoch_size) else 0
    }
    if args.log_bok:
        feed_dict.update({
            eval_ops.ph_BoK_distance: accum.bok_mean_eds,
            eval_ops.ph_BoK_accuracy: accum.bok_accuracies,
            eval_ops.ph_BoK_bleu2: agg_bok_bleu2,
            eval_ops.ph_BoK_bleus: accum.bok_bleus
        })
    summary_ops.run_ops(session, eval_ops, feed_dict)
    # logs_agg_top1 = session.run((eval_ops.logs_agg_top1, ) + ((eval_ops.logs_agg_bok,) if args.log_bok else tuple()),
    #                             feed_dict={
    #                                 eval_ops.ph_top1_len_ratio: accum.len_ratio,
    #                                 eval_ops.ph_edit_distance: accum.mean_eds,
    #                                 eval_ops.ph_num_hits: accum.hits,
    #                                 eval_ops.ph_accuracy: accum.accuracies,
    #                                 eval_ops.ph_valid_time: eval_time_per100,
    #                                 eval_ops.ph_bleus: accum.bleus,
    #                                 eval_ops.ph_bleu2: agg_bleu2,
    #                                 eval_ops.ph_BoK_bleu2: agg_bok_bleu2 if args.log_bok else [],
    #                                 eval_ops.ph_BoK_bleus: accum.bok_bleus if args.log_bok else [],
    #                                 eval_ops.ph_full_validation: 1 if (num_steps == batch_it.epoch_size) else 0
    #                             })
    with dtc.Storer(args, 'metrics_test' if args.doTest else 'metrics_validation', step) as storer:
        storer.write('edit_distance', np.mean(accum.mean_eds, keepdims=True), np.float32)
        storer.write('num_hits', np.sum(accum.hits, keepdims=True), dtype=np.uint32)
        storer.write('accuracy', np.mean(accum.accuracies, keepdims=True), np.float32)
        storer.write('bleu', np.mean(accum.bleus, keepdims=True), dtype=np.float32)
        storer.write('bleu2', np.asarray([agg_bleu2]), dtype=np.float32)
        if args.log_bok:
            storer.write('bok_bleu', np.mean(accum.bok_bleus, keepdims=True), dtype=np.float32)
            storer.write('bok_bleu2', np.asarray([agg_bok_bleu2]), dtype=np.float32)
            storer.write('bok_edit_distance', np.mean(accum.bok_mean_eds, keepdims=True), np.float32)
            storer.write('bok_accuracy', np.mean(accum.bok_accuracies, keepdims=True), np.float32)

    tf_sw.add_summary(summary_ops.logs_agg_top1, standardized_step(step))
    if args.log_bok:
        tf_sw.add_summary(summary_ops.logs_agg_bok, standardized_step(step))

    tf_sw.flush()
    hyper.logger.info('validation cycle finished. bleu2 = %f', agg_bleu2)
    return dlc.Properties({'eval_time_per100': eval_time_per100})
