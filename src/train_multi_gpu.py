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
    total_lstm_0 = 0
    total_lstm_1 = 0
    total_calstm2 = 0
    total_lstm2_0 = 0
    total_lstm2_1 = 0
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
        elif 'CALSTM_2' in var.name:
            total_calstm2 += n
            if 'multi_rnn_cell/cell_0' in var.name:
                total_lstm2_0 += n
            elif 'multi_rnn_cell/cell_1' in var.name:
                total_lstm2_1 += n
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
    logger.info( 'Vggnet: %d (%2.2f%%)'%(total_vggnet, total_vggnet*100./total_n))
    logger.info( 'Convnet: %d (%2.2f%%)'%(total_convnet, total_convnet*100./total_n))
    logger.info( 'Initializer: %d (%2.2f%%)'%(total_init, total_init*100./total_n))
    logger.info( 'CALSTM_1: %d (%2.2f%%)'%(total_calstm, total_calstm*100./total_n))
    logger.info( 'LSTM1_0: %d (%2.2f%%)'%(total_lstm_0, total_lstm_0*100./total_n))
    logger.info( 'LSTM1_1: %d (%2.2f%%)'%(total_lstm_1, total_lstm_1*100./total_n))
    logger.info( 'CALSTM_2: %d (%2.2f%%)'%(total_calstm2, total_calstm2*100./total_n))
    logger.info( 'LSTM2_0: %d (%2.2f%%)'%(total_lstm2_0, total_lstm2_0*100./total_n))
    logger.info( 'LSTM2_1: %d (%2.2f%%)'%(total_lstm2_1, total_lstm2_1*100./total_n))
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

    def run_ops(self, session, op_dict):
        self._set_vals(session.run(self._op_tuple(op_dict)))


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
    dtc.initialize(args.generated_data_dir, hyper)
    global logger
    logger = hyper.logger
    global standardized_step
    standardized_step = make_standardized_step(hyper)

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

        qrs = []
        ##### Training Graphs
        train_tower_ops = []; train_ops = None
        with tf.name_scope('Training'):
            tf_train_step = tf.get_variable('global_step', dtype=hyper.int_type, trainable=False, initializer=0)
            if hyper.optimizer == 'adam':
                opt = tf.train.AdamOptimizer(learning_rate=hyper.adam_alpha)
            else:
                raise Exception('Unsupported optimizer - %s - configured.' % (hyper.optimizer,))

        if args.doTrain:
            with tf.name_scope('Training'):
                with tf.variable_scope('InputQueue'):
                    train_q = tf.FIFOQueue(hyper.input_queue_capacity, train_it.out_tup_types)
                    tf_enqueue_train_queue = train_q.enqueue_many(train_it.get_pyfunc_with_split(hyper.num_gpus))
                    tf_close_train_queue = train_q.close(cancel_pending_enqueues=True)
                for i in range(args.num_gpus):
                    with tf.name_scope('gpu_%d'%i):
                        with tf.device('/gpu:%d'%i):
                            model = Im2LatexModel(hyper, train_q, opt=opt, reuse=(False if (i == 0) else True))
                            train_tower_ops.append(model.build_training_tower())
                            if i == 0:
                                trainable_vars_n = num_trainable_vars() # 8544670 or 8547670
                                hyper.logger.info('Num trainable variables = %d', trainable_vars_n)
                                ## assert trainable_vars_n == 8547670 if hyper.use_peephole else 8544670
                                ## assert trainable_vars_n == 23261206 if hyper.build_image_context
                            else:
                                assert num_trainable_vars() == trainable_vars_n, 'trainable_vars %d != expected %d'%(num_trainable_vars(), trainable_vars_n)
                train_ops = sync_training_towers(hyper, train_tower_ops, tf_train_step, optimizer=opt)
            qr1 = tf.train.QueueRunner(train_q, [tf_enqueue_train_queue], cancel_op=[tf_close_train_queue])
            qrs.append(qr1)

        ##### Validation Graph
        valid_tower_ops = []; valid_ops = None
        if valid_it and (args.doTrain or args.doValidate):
            with tf.name_scope('Validation'):
                hyper_predict = hyper_params.make_hyper(args.copy().updated({'dropout':None}))
                with tf.variable_scope('InputQueue'):
                    valid_q = tf.FIFOQueue(hyper.input_queue_capacity, valid_it.out_tup_types)
                    enqueue_op2 = valid_q.enqueue_many(valid_it.get_pyfunc_with_split(hyper.num_gpus))
                    close_queue2 = valid_q.close(cancel_pending_enqueues=True)
                for i in range(args.num_gpus):
                    with tf.name_scope('gpu_%d'%i):
                        with tf.device('/gpu:%d'%i):
                            reuse_vars = False if ((i==0) and (not args.doTrain)) else True
                            if hyper.build_scanning_RNN:
                                model_predict = Im2LatexModel(hyper_predict,
                                                              valid_q,
                                                              reuse=reuse_vars)
                                valid_tower_ops.append(model_predict.build_training_tower())
                            else:
                                model_predict = Im2LatexModel(hyper_predict,
                                                              valid_q,
                                                              seq2seq_beam_width=hyper.seq2seq_beam_width,
                                                              reuse=reuse_vars)
                                valid_tower_ops.append(model_predict.build_testing_tower())
                            if not reuse_vars:
                                trainable_vars_n = num_trainable_vars()
                hyper.logger.info('Num trainable variables = %d', num_trainable_vars())
                assert num_trainable_vars() == trainable_vars_n, 'num_trainable_vars(%d) != %d'%(num_trainable_vars(), trainable_vars_n)
                if hyper.build_scanning_RNN:
                    valid_ops = sync_training_towers(hyper, valid_tower_ops, global_step=None, run_tag='validation')
                else:
                    valid_ops = sync_testing_towers(hyper, valid_tower_ops)
            qr2 = tf.train.QueueRunner(valid_q, [enqueue_op2], cancel_op=[close_queue2])
            qrs.append(qr2)

        # ##### Training Accuracy Graph
        # if (args.make_training_accuracy_graph):
        #     with tf.name_scope('TrainingAccuracy'):
        #         hyper_predict2 = hyper_params.make_hyper(args.copy().updated({'dropout':None}))
        #         with tf.device('/gpu:1'):
        #             model_predict2 = Im2LatexModel(hyper_predict, hyper.seq2seq_beam_width, reuse=True)
        #             tr_acc_ops = model_predict2.test()
        #         with tf.variable_scope('QueueOps'):
        #             enqueue_op3 = tr_acc_ops.inp_q.enqueue_many(tr_acc_it.get_pyfunc_with_split(hyper.num_gpus))
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
                if args.doTrain:
                    logger.info('Starting training')
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
                    )

                    ops_log = (
                        'y_s_list',
                        'predicted_ids_list',
                        'alpha',
                        'beta',
                        'image_name_list',
                        'x_s_list',
                    )
                    accum = Accumulator()
                    while not coord.should_stop():
                        step_start_time = time.time()
                        step += 1
                        doLog = do_log(step, args, train_it, valid_it)
                        if not doLog:
                            batch_ops = TFOpNames(ops_accum, None)
                        else:
                            batch_ops = TFOpNames(ops_accum + ops_log, None)

                        batch_ops.run_ops(session, train_ops)
                        ## Accumulate Metrics
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

                            agg_bleu2 = dlc.corpus_bleu_score(accum.sq_predicted_ids, accum.sq_y_ctc)
                            # Calculate aggregated training metrics
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

                            doValidate, num_validation_batches, do_save = TrainingLogic.do_validate(step, args, train_it, valid_it, (agg_bleu2 if (args.valid_epochs <= 0) else None))
                            if do_save:
                                saver.save(session, args.logdir + '/snapshot', global_step=step,
                                           latest_filename='checkpoints_list')
                            if doValidate:
                                if hyper.build_scanning_RNN:
                                    accuracy_res = validate_scanning_RNN(args, hyper, session, valid_ops, ops_accum, ops_log, step, num_validation_batches, tf_sw)
                                else:
                                    accuracy_res = evaluate(
                                        session,
                                        dlc.Properties({'valid_ops':valid_ops, 'tr_acc_ops':tr_acc_ops}),
                                        dlc.Properties({'train_it':train_it, 'valid_it':valid_it, 'tr_acc_it':tr_acc_it}),
                                        hyper,
                                        args,
                                        step,
                                        num_validation_batches,
                                        tf_sw)
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

                            ## Reset Metrics
                            accum.reset()

                ############################# Validation Only ##############################
                elif args.doValidate:
                    logger.info('Starting Validation Cycle')
                    evaluate(
                            session,
                            dlc.Properties({'valid_ops':valid_ops, 'tr_acc_ops':tr_acc_ops}),
                            dlc.Properties({'train_it':train_it, 'valid_it':valid_it, 'tr_acc_it':tr_acc_it}),
                            hyper,
                            args,
                            step,
                            valid_it.epoch_size,
                            tf_sw)

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
    :param pred_lens: np-array of integers representing lengths of the predicted sequences. shape = (num_gpus*B,)
    :param target_ar_list: list of np-array of target word/id-sequences. shape = [(B,T), ...]
    :param target_lens: np-array of integers representing lengths of the target sequences. shape = (num_gpus*B,)
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
    Gathers all word/id sequences into one list of size num_gpus*B. Optionally Squashes and trims the sequences.
    :param seq_batch_list: list of np-array of word/id-sequences. shape = [(B,T), ...]
    :param seq_len_batch: np-array of integers representing lengths of the above sequences. shape = (num_gpus*B,)
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
    Squashes and trims word/id sequences and puts them all into one list of size num_gpus*B
    :param hyper: HyperParams
    :param seq_batch_list: list of np-array of word/id-sequences. shape = [(B,T), ...]
    :param seq_len_batch: np-array of integers representing lengths of the above sequences. shape = (num_gpus*B,)
    :return: a list of word/id lists
    """
    return squash_and_concat(seq_batch_list, seq_len_batch,
                             remove_val1=hyper.SpaceTokenID,
                             remove_val2=hyper.CTCBlankTokenID,
                             eos_token=hyper.NullTokenID)


def trimmed_seq_list(hyper, seq_batch_list, seq_len_batch):
    """
    Trims word/id sequences and puts them all into one list of size num_gpus*B
    :param hyper: HyperParams
    :param seq_batch_list: list of np-array of word/id-sequences. shape = [(B,T), ...]
    :param seq_len_batch: np-array of integers representing lengths of the above sequences. shape = (num_gpus*B,)
    :return: a list of word/id lists
    """
    return squash_and_concat(seq_batch_list, seq_len_batch, eos_token=hyper.NullTokenID)


def ids2str_list(target_ids, predicted_ids, hyper):
    """
    Same as id2str, except this works on multiple batches. The arguments are lists of numpy arrays
    instead of straight numpy arrays as in the case of id2str.
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
    fv_score = 0.75

    @classmethod
    def do_validate(cls, step, args, train_it, valid_it, score=None):
        if valid_it is None:
            doValidate = do_save = False
            num_valid_batches = 0
        elif args.doValidate:  # Validation-only run
            doValidate = do_save = True
            num_valid_batches = valid_it.epoch_size
        elif (args.valid_epochs <= 0):  # smart validation
            assert score is not None, 'score must be supplied if valid_epochs <= 0'
            if (score - cls.fv_score) >= 0.01:
                doValidate = True
                num_valid_batches = valid_it.epoch_size
                do_save = True
                cls.full_validation_steps.append(step)
                cls.fv_score = score
                logger.info('TrainingLogic: fv_score set to %f at step %d'%(score, step))
            elif (score > (cls.fv_score * 0.95)):
                if (step - cls.full_validation_steps[-1]) >= (1 * train_it.epoch_size):
                    doValidate = do_save = True
                    num_valid_batches = valid_it.epoch_size
                    cls.full_validation_steps.append(step)
                    logger.info('TrainingLogic: Running full validation at score %f at step %d' % (score, step))
                else:
                    doValidate = False
                    do_save = False
                    num_valid_batches = 0
            elif (step - cls.full_validation_steps[-1]) >= (2 * train_it.epoch_size):
                doValidate = do_save = True
                num_valid_batches = valid_it.epoch_size
                cls.full_validation_steps.append(step)
                logger.info('TrainingLogic: Running full validation at score %f at step %d' % (score, step))
            else:
                doValidate = do_save = False
                num_valid_batches = 0
        else:
            epoch_frac = args.valid_epochs if (args.valid_epochs is not None) else 1
            period = int(epoch_frac * train_it.epoch_size)
            doValidate = do_save = (step % period == 0) or (step == train_it.max_steps)
            num_valid_batches = valid_it.epoch_size if doValidate else 0

        # if doValidate:
        #     if args.valid_epochs > 0:
        #         num_valid_batches = valid_it.epoch_size
        #     else:  # smart validation
        #         if len(cls.full_validation_steps) == 0:
        #             num_valid_batches = valid_it.epoch_size
        #             cls.full_validation_steps.append(step)
        #         elif (step - cls.full_validation_steps[-1]) > (0.95 * train_it.epoch_size):
        #             num_valid_batches = valid_it.epoch_size
        #             cls.full_validation_steps.append(step)
        #         else:
        #             num_valid_batches = 5
        # else:
        #     num_valid_batches = 0

        return doValidate, num_valid_batches, do_save


def do_log(step, args, train_it, valid_it):
    # validate, _ = do_validate(step, args, train_it, valid_it)
    do_log = (step % args.print_steps == 0) or (step == train_it.max_steps)
    if (args.valid_epochs <= 0 ):  # do_validate synchronizes with do_log
        # score = np.mean(accum.bleu_scores) if (args.valid_epochs == 'smart') else None
        return do_log
    else:
        validate, _, _ = TrainingLogic.do_validate(step, args, train_it, valid_it, None)
        return do_log or validate


def format_ids(predicted_ids, target_ids):
    np.apply_along_axis


def validate_scanning_RNN(args, hyper, session, ops, ops_accum, ops_log, tr_step, num_steps, tf_sw):
    start_time = time.time()
    ############################# Training (with Validation) Cycle ##############################
    hyper.logger.info('validation cycle starting at step %d for %d steps', tr_step, num_steps)
    accum = Accumulator()
    print_batch_num = np.random.randint(1, num_steps + 1) if args.print_batch else -1
    for batch in range(1, 1+num_steps):
        step_start_time = time.time()
        doLog = (print_batch_num == batch)
        if not doLog:
            batch_ops = TFRun(session, ops, ops_accum, None)
        else:
            batch_ops = TFRun(session, ops, ops_accum + ops_log, None)

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
            'sq_y_ctc': trimmed_seq_list(hyper, batch_ops.y_ctc_list, batch_ops.ctc_len)
        })
        accum.append({'batch_time': batch_time})

        if doLog:
            with dtc.Storer(args, 'validation', tr_step) as storer:
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

    # Calculate aggregated validation metrics
    valid_time_per100 = np.mean(batch_time) * 100. / hyper.data_reader_B
    tb_agg_logs = session.run(ops.tb_agg_logs, feed_dict={
        ops.ph_train_time: valid_time_per100,
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
    return dlc.Properties({'valid_time_per100': valid_time_per100})

def evaluate(session, ops, batch_its, hyper, args, step, num_steps, tf_sw):
    # validate, num_steps = do_validate(step, args, batch_its.train_it, batch_its.valid_it)
    valid_start_time = time.time()

    valid_ops = ops.valid_ops
    batch_it = batch_its.valid_it
    batch_size = batch_it.batch_size
    epoch_size = batch_it.epoch_size
    ## Print a batch randomly
    print_batch_num = np.random.randint(1, num_steps+1) if args.print_batch else -1
    accum = Accumulator()
    n = 0
    hyper.logger.info('validation cycle starting at step %d for %d steps', step, num_steps)
    while n < num_steps:
        n += 1
        if (n != print_batch_num):
            l, ed, accuracy, num_hits, top1_ids_list, top1_lens, y_ctc_list, ctc_len = session.run((
                                valid_ops.top1_len_ratio,
                                valid_ops.top1_mean_ed,
                                valid_ops.top1_accuracy,
                                valid_ops.top1_num_hits,
                                valid_ops.top1_ids_list,
                                valid_ops.top1_lens,
                                valid_ops.y_ctc_list,
                                valid_ops.ctc_len
                                ))
            y_s_list = top1_alpha_list = top1_beta_list = image_name_list = top1_ed = None
        else:
            l, ed, accuracy, num_hits, top1_ids_list, top1_lens, y_ctc_list, ctc_len, y_s_list, top1_alpha_list, top1_beta_list, image_name_list, top1_ed = session.run((
                                valid_ops.top1_len_ratio,
                                valid_ops.top1_mean_ed,
                                valid_ops.top1_accuracy,
                                valid_ops.top1_num_hits,
                                valid_ops.top1_ids_list,
                                valid_ops.top1_lens,
                                valid_ops.y_ctc_list,
                                valid_ops.ctc_len,
                                valid_ops.y_s_list,
                                valid_ops.top1_alpha_list,
                                valid_ops.top1_beta_list,
                                valid_ops.image_name_list,
                                valid_ops.top1_ed
                                ))

        bleu = sentence_bleu_scores(hyper, top1_ids_list, top1_lens, y_ctc_list, ctc_len)
        accum.extend({'bleus': bleu})
        accum.extend({'predicted_ids': squashed_seq_list(hyper, top1_ids_list, top1_lens)})
        accum.extend({'target_ids': trimmed_seq_list(hyper, y_ctc_list, ctc_len)})
        accum.append({'lens': l})
        accum.append({'eds': ed})
        accum.append({'accuracies': accuracy})
        accum.append({'hits': num_hits})

        if n == print_batch_num:
            # logger.info('############ RANDOM VALIDATION BATCH %d ############', n)
            # logger.info('prediction mean_ed=%f', ed)
            # logger.info('prediction accuracy=%f', accuracy)
            # logger.info('prediction hits=%d', num_hits)
            # bleu = sentence_bleu_scores(hyper, top1_ids_list, top1_lens, y_ctc_list, ctc_len)

            with dtc.Storer(args, 'validation', step) as storer:
                storer.write('predicted_ids', top1_ids_list, np.int16)
                storer.write('y', y_s_list, np.int16)
                storer.write('alpha', top1_alpha_list, dtype=np.float32, batch_axis=1)
                storer.write('beta', top1_beta_list, dtype=np.float32, batch_axis=1)
                storer.write('image_name', image_name_list, dtype=np.unicode_)
                storer.write('ed', top1_ed, dtype=np.float32)
                storer.write('bleu', bleu, dtype=np.float32)

            # logger.info( '############ END OF RANDOM VALIDATION BATCH ############')


    valid_time_per100 = (time.time() - valid_start_time) * 100. / (num_steps * batch_size)
    agg_bleu2 = dlc.corpus_bleu_score(accum.predicted_ids, accum.target_ids)
    logs_agg_top1 = session.run(valid_ops.logs_agg_top1,
                                feed_dict={
                                    valid_ops.ph_top1_len_ratio: accum.lens,
                                    valid_ops.ph_edit_distance: accum.eds,
                                    valid_ops.ph_num_hits: accum.hits,
                                    valid_ops.ph_accuracy: accum.accuracies,
                                    valid_ops.ph_valid_time: valid_time_per100,
                                    valid_ops.ph_bleus: accum.bleus,
                                    valid_ops.ph_bleu2: agg_bleu2,
                                    valid_ops.ph_full_validation: 1 if (num_steps == batch_it.epoch_size) else 0
                                })

    tf_sw.add_summary(logs_agg_top1, log_step(step))
    tf_sw.flush()
    hyper.logger.info('validation cycle finished. bleu2 = %f', agg_bleu2)
    return dlc.Properties({'valid_time_per100': valid_time_per100})
