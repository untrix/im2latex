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

Works on python 2.7
"""
import sys
import argparse
import os
import logging
import tensorflow as tf
sys.path.extend(['commons', 'model'])
import dl_commons as dlc
import tf_commons as tfc
import data_commons as dtc
import hyper_params
import train_multi_gpu

def main():
    logger = dtc.makeLogger(set_global=True)

    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument("--num-steps", "-n", dest="num_steps", type=int,
                        help="Number of training steps to run. Defaults to -1 if unspecified, i.e. run to completion",
                        default=-1)
    parser.add_argument("--num-epochs", "-e", dest="num_epochs", type=int,
                        help="Number of training epochs to run. Defaults to 10 if unspecified.",
                        default=10)
    parser.add_argument("--batch-size", "-b", dest="batch_size", type=int,
                        help="Batchsize per gpu. If unspecified, defaults to the default value in hyper_params",
                        default=None)
    parser.add_argument("--seq2seq-beam-width", "-w", dest="seq2seq_beam_width", type=int,
                        help="seq2seq Beamwidth. If unspecified, defaults to 10",
                        default=10)
    parser.add_argument("--ctc-beam-width", dest="ctc_beam_width", type=int,
                        help="CTC Beamwidth. If unspecified, defaults to 10",
                        default=10)
    parser.add_argument("--print-steps", "-s", dest="print_steps", type=int,
                        help="Number of training steps after which to log results. Defaults to 100 if unspecified",
                        default=100)
    parser.add_argument("--keep-prob", "-k", dest="keep_prob", type=float,
                        help="Dropout 'keep' probability. Defaults to 0.5",
                        default=1.0)
    parser.add_argument("--adam_alpha", "-a", dest="alpha", type=float,
                        help="Alpha (step / learning-rate) value of adam optimizer.",
                        default=0.0001)
    parser.add_argument("--r-lambda", "-r", dest="rLambda", type=float,
                        help="Sets value of rLambda - lambda value used for regularization. Defaults to 0.00005.",
                        default=0.00005)
    parser.add_argument("--data-folder", "-d", dest="data_folder", type=str,
                        help="Data folder. If unspecified, defaults to raw_data_folder/..",
                        default=None)
    parser.add_argument("--raw-data-folder", dest="raw_data_folder", type=str,
                        help="Raw data folder. Must be specified.",
                        default=None)
    parser.add_argument("--vgg16-folder", dest="vgg16_folder", type=str,
                        help="vgg16 data folder. If unspecified, defaults to data_folder/vgg16_features",
                        default=None)
    parser.add_argument("--image-folder", dest="image_folder", type=str,
                        help="image folder. If unspecified, defaults to data_folder/formula_images",
                        default=None)
    parser.add_argument("--partial-batch", "-p",  dest="partial_batch", action='store_true',
                        help="Sets assert_whole_batch hyper param to False. Default value for this option is False "
                             " (i.e. assert_whole_batch=True)",
                        default=False)
    parser.add_argument("--queue-capacity", "-q", dest="queue_capacity", type=int,
                        help="Capacity of input queue. Defaults to hyperparam defaults if unspecified.",
                        default=None)
    parser.add_argument("--logging-level", "-l", dest="logging_level", type=int, choices=range(1,6),
                        help="Logging verbosity level from 1 to 5 in increasing order of verbosity.",
                        default=4)
    parser.add_argument("--log-bok", dest="log_bok", action='store_true',
                        help="Whether to log best-of-k results",
                        default=False)
    parser.add_argument("--valid-frac", "-f", dest="valid_frac", type=float,
                        help="Fraction of samples to use for validation. Defaults to 0.05",
                        default=0.05)
    parser.add_argument("--validation-epochs", "-v", dest="valid_epochs", type=float,
                        help="""Number (or fraction) of epochs after which to run a full validation cycle. For this
                             behaviour, the number should be greater than 0. A value less <= 0 on the other hand,
                             implies 'smart' validation - which will result in selectively 
                             capturing snapshots around max_scoring peaks (based on training/bleu2).""",
                        default=1.0)
    parser.add_argument("--save-all-eval",  dest="save_all_eval", action='store_true',
                        help="(Boolean): False => Save only one random validation/testing batch, True = > Save all validation/testing batches",
                        default=False)
    parser.add_argument("--build-image-context", "-i", dest="build_image_context", type=int,
                        help="Sets value of hyper.build_image_context. Default is 2 => build my own convnet.",
                        default=2)
    parser.add_argument("--swap-memory", dest="swap_memory", action='store_true',
                        help="swap_memory option of tf.scan and tf.while_loop. Default to False."
                             " Enabling allows training larger mini-batches at the cost of speed.",
                        default=False)
    parser.add_argument("--restore", dest="restore_logdir", type=str,
                        help="restore from checkpoint. Provide logdir path as argument. Don't specify the --logdir argument.",
                        default=None)
    parser.add_argument("--logdir", dest="logdir", type=str,
                        help="(optional) Sets TensorboardParams.tb_logdir. Can't specify the --restore argument along with this.",
                        default=None)
    parser.add_argument("--logdir-tag", dest="logdir_tag", type=str,
                        help="(optional) Sets TensorboardParams.logdir_tab. Can't specify the --restore argument along with this.",
                        default=None)
    # parser.add_argument("--use-ctc-loss", dest="use_ctc_loss", action='store_true',
    #                     help="Sets the use_ctc_loss hyper parameter. Defaults to False.",
    #                     default=False)
    parser.add_argument("--validate", dest="doValidate", action='store_true',
                        help="Run validation cycle only. --restore option should be provided along with this.",
                        default=False)
    parser.add_argument("--test", dest="doTest", action='store_true',
                        help="Run test cycle only but with training dataset. --restore option should be provided along with this.",
                        default=False)
    parser.add_argument("--squash-input-seq", dest="squash_input_seq", action='store_true',
                        help="(boolean) Set value of squash_input_seq hyper param. Defaults to True.",
                        default=True)
    parser.add_argument("--num-snapshots", dest="num_snapshots", type=int,
                        help="Number of latest snapshots to save. Defaults to 100 if unspecified",
                        default=100)

    args = parser.parse_args()

    raw_data_folder = args.raw_data_folder
    if args.data_folder:
        data_folder = args.data_folder
    else:
        data_folder = os.path.join(raw_data_folder, '..')

    if args.image_folder:
        image_folder = args.image_folder
    else:
        image_folder = os.path.join(data_folder, 'formula_images')


    if args.vgg16_folder:
        vgg16_folder = args.vgg16_folder
    else:
        vgg16_folder = os.path.join(data_folder, 'vgg16_features')

    if args.restore_logdir is not None:
        assert args.logdir is None, 'Only one of --restore-logdir and --logdir can be specified.'
        assert args.logdir_tag is None, "--logdir-tag can't be specified alongside --logdir"
        tb = tfc.TensorboardParams({'tb_logdir': os.path.dirname(args.restore_logdir)}).freeze()
    elif args.logdir is not None:
        tb = tfc.TensorboardParams({'tb_logdir': args.logdir, 'logdir_tag': args.logdir_tag}).freeze()
    else:
        tb = tfc.TensorboardParams({'tb_logdir': './tb_metrics', 'logdir_tag': args.logdir_tag}).freeze()

    if args.doValidate:
        assert args.restore_logdir is not None, 'Please specify --restore option along with --validate'
        assert not args.doTest, '--test and --validate cannot be given together'

    if args.doTest:
        assert args.restore_logdir is not None, 'Please specify --restore option along with --test'
        assert not args.doValidate, '--test and --validate cannot be given together'

    globalParams = dlc.Properties({
                                    'raw_data_dir': raw_data_folder,
                                    'assert_whole_batch': not args.partial_batch,
                                    'logger': logger,
                                    'tb': tb,
                                    'print_steps': args.print_steps,
                                    'num_steps': args.num_steps,
                                    'num_epochs': args.num_epochs,
                                    'num_snapshots': args.num_snapshots,
                                    'data_dir': data_folder,
                                    'generated_data_dir': data_folder,
                                    'image_dir': image_folder,
                                    'ctc_beam_width': args.ctc_beam_width,
                                    'seq2seq_beam_width': args.seq2seq_beam_width,
                                    'k': 5,
                                    'valid_frac': args.valid_frac,
                                    'valid_epochs': args.valid_epochs,
                                    'save_all_eval': args.save_all_eval,
                                    'build_image_context':args.build_image_context,
                                    'sum_logloss': False,  # setting to true equalizes ctc_loss and log_loss if y_s == squashed_seq
                                    'dropout': None if args.keep_prob >= 1.0 else tfc.DropoutParams({'keep_prob': args.keep_prob}).freeze(),
                                    'MeanSumAlphaEquals1': False,
                                    'rLambda': args.rLambda,  # 0.0005, 0.00005
                                    'make_training_accuracy_graph': False,
                                    # 'use_ctc_loss': args.use_ctc_loss,
                                    "swap_memory": args.swap_memory,
                                    'tf_session_allow_growth': False,
                                    'restore_from_checkpoint': args.restore_logdir is not None,
                                    'num_gpus': 2,
                                    'towers_per_gpu': 1,
                                    'beamsearch_length_penalty': 1.0,
                                    'doValidate': args.doValidate,
                                    'doTest': args.doTest,
                                    'doTrain': not (args.doValidate or args.doTest),
                                    'squash_input_seq': args.squash_input_seq,
                                    'att_model': 'MLP_full', # '1x1_conv', 'MLP_shared', 'MLP_full'
                                    'weights_regularizer': tf.contrib.layers.l2_regularizer(scale=1.0, scope='L2_Regularizer'),
                                    # 'embeddings_regularizer': None,
                                    # 'outputMLP_skip_connections': False,
                                    'output_reuse_embeddings': False,
                                    'REGROUP_IMAGE': None,  # None or (4,1)
                                    'build_att_modulator': False,  # turn off beta-MLP
                                    'build_scanning_RNN': False,
                                    'init_model_input_transform': 'full',
                                    'build_init_model': True,
                                    'adam_beta1': 0.5,
                                    'adam_beta2': 0.9,
                                    'pLambda': 0.0,
                                    'log_bok': args.log_bok
                                    })

    if args.batch_size is not None:
        globalParams.B = args.batch_size

    if args.queue_capacity is not None:
        globalParams.input_queue_capacity = args.queue_capacity
    if args.alpha is not None:
        globalParams.adam_alpha = args.alpha

    if args.restore_logdir is not None:
        globalParams.logdir = args.restore_logdir
    else:
        globalParams.logdir = dtc.makeTBDir(tb.tb_logdir, tb.logdir_tag)

    # args
    globalParams.storedir = dtc.makeLogDir(globalParams.logdir, 'store')
    globalParams.dump(dtc.makeLogfileName(globalParams.storedir, 'args.pkl'))

    # Hyper Params
    hyper = hyper_params.make_hyper(globalParams, freeze=False)
    if args.restore_logdir is not None:
        hyper.dump(dtc.makeLogfileName(globalParams.storedir, 'hyper.pkl'))
    else:
        hyper.dump(globalParams.storedir, 'hyper.pkl')

    # Logger
    fh = logging.FileHandler(dtc.makeLogfileName(globalParams.storedir, 'training.log'))
    fh.setFormatter(dtc.makeFormatter())
    logger.addHandler(fh)
    dtc.setLogLevel(logger, args.logging_level)

    logger.info(' '.join(sys.argv))
    logger.info('\n#################### Default Param Overrides: ####################\n%s',globalParams.pformat())
    logger.info('##################################################################\n')
    logger.info( '\n#########################  Hyper-params: #########################\n%s', hyper.pformat())
    logger.info('##################################################################\n')

    train_multi_gpu.main(raw_data_folder,
          vgg16_folder,
          globalParams,
          hyper.freeze()
         )

main()
