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
import dl_commons as dlc
import tf_commons as tfc
import train_multi_gpu
import logging
import hyper_params
import argparse
import os

def main():
    _data_folder = '../data'
    _logdir = 'tb_metrics'
    _logdir2 = 'tb_metrics_dev'

    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument("--num-steps", "-n", dest="num_steps", type=int,
                        help="Number of training steps to run. Defaults to -1 if unspecified, i.e. run to completion",
                        default=-1)
    parser.add_argument("--num-epochs", "-e", dest="num_epochs", type=int,
                        help="Number of training steps to run. Defaults to 10 if unspecified.",
                        default=10)
    parser.add_argument("--batch-size", "-b", dest="batch_size", type=int,
                        help="Batchsize. If unspecified, defaults to the default value in hyper_params",
                        default=None)
    parser.add_argument("--seq2seq-beam-width", "-w", dest="seq2seq_beam_width", type=int,
                        help="seq2seq Beamwidth. If unspecified, defaults to 10",
                        default=10)
    parser.add_argument("--ctc-beam-width", dest="ctc_beam_width", type=int,
                        help="CTC Beamwidth. If unspecified, defaults to 10",
                        default=10)
    parser.add_argument("--print-steps", "-s", dest="print_steps", type=int,
                        help="Number of training steps after which to log results. Defaults to 50 if unspecified",
                        default=50)
    parser.add_argument("--keep-prob", "-k", dest="keep_prob", type=float,
                        help="Dropout 'keep' probability. Defaults to 0.5",
                        default=0.5)
    parser.add_argument("--adam_alpha", "-a", dest="alpha", type=float,
                        help="Alpha (step / learning-rate) value of adam optimizer.",
                        default=None)
    parser.add_argument("--r-lambda", "-r", dest="rLambda", type=float,
                        help="Sets value of rLambda - lambda value used for regularization. Defaults to 00005.",
                        default=0.00005)
    parser.add_argument("--data-folder", "-d", dest="data_folder", type=str,
                        help="Data folder. If unspecified, defaults to " + _data_folder,
                        default=_data_folder)
    parser.add_argument("--raw-data-folder", dest="raw_data_folder", type=str,
                        help="Raw data folder. If unspecified, defaults to data_folder/training",
                        default=None)
    parser.add_argument("--vgg16-folder", dest="vgg16_folder", type=str,
                        help="vgg16 data folder. If unspecified, defaults to raw_data_folder/vgg16_features_2",
                        default=None)
    parser.add_argument("--image-folder", dest="image_folder", type=str,
                        help="image folder. If unspecified, defaults to data_folder/formula_images_2",
                        default=None)
    parser.add_argument("--partial-batch", "-p",  dest="partial_batch", action='store_true',
                        help="Sets assert_whole_batch hyper param to False. Default hyper_param value will be used if unspecified")
    parser.add_argument("--queue-capacity", "-q", dest="queue_capacity", type=int,
                        help="Capacity of input queue. Defaults to hyperparam defaults if unspecified.",
                        default=None)
    parser.add_argument("--logging-level", "-l", dest="logging_level", type=int,
                        help="Logging verbosity level from 1 to 5 in increasing order of verbosity.",
                        default=4)
    parser.add_argument("--valid-frac", "-f", dest="valid_frac", type=float,
                        help="Fraction of samples to use for validation. Defaults to 0.05",
                        default=0.05)
    parser.add_argument("--validation-epochs", "-v", dest="valid_epochs", type=float,
                        help="Number (or fraction) of epochs after which to run a full validation cycle. Defaults to 1.0",
                        default=1.0)
    parser.add_argument("--print-batch",  dest="print_batch", action='store_true',
                        help="(Boolean): Only for debugging. Prints more stuff once in a while. Defaults to True.",
                        default=True)
    parser.add_argument("--build-image-context", "-i", dest="build_image_context", type=int,
                        help="Sets value of hyper.build_image_context. Default is 2 => build my own convnet.",
                        default=2)
    parser.add_argument("--logdir2", dest="logdir2", action='store_true',
                        help="Log to alternative data_folder " + _logdir2,
                        default=False)
    parser.add_argument("--swap-memory", dest="swap_memory", action='store_true',
                        help="swap_memory option of tf.scan and tf.while_loop. Default to False."
                             " Enabling allows training larger mini-batches at the cost of speed.",
                        default=False)
    parser.add_argument("--restore", dest="restore_logdir", type=str,
                        help="restore from checkpoint. Provide logdir path as argument",
                        default=None)
    parser.add_argument("--use-ctc-loss", dest="use_ctc_loss", action='store_true',
                        help="Sets the use_ctc_loss hyper parameter. Defaults to False.",
                        default=False)

    args = parser.parse_args()
    data_folder = args.data_folder
    if args.image_folder:
        image_folder = args.image_folder
    else:
        image_folder = os.path.join(data_folder,'formula_images_2')

    if args.raw_data_folder:
        raw_data_folder = args.raw_data_folder
    else:
        raw_data_folder = os.path.join(data_folder, 'generated2', 'training')

    if args.vgg16_folder:
        vgg16_folder = args.vgg16_folder
    else:
        vgg16_folder = os.path.join(data_folder, 'vgg16_features_2')

    if args.restore_logdir is not None:
        tb = tfc.TensorboardParams({'tb_logdir': os.path.dirname(args.restore_logdir)})
    elif args.logdir2:
        tb = tfc.TensorboardParams({'tb_logdir':_logdir2})
    else:
        tb = tfc.TensorboardParams({'tb_logdir':_logdir})

    logger = hyper_params.makeLogger()

    globalParams = dlc.Properties({
                                    'tb': tb,
                                    'print_steps': args.print_steps,
                                    'num_steps': args.num_steps,
                                    'num_epochs': args.num_epochs,
                                    'data_dir': data_folder,
                                    'generated_data_dir': os.path.join(data_folder, 'generated2'),
                                    'image_dir': image_folder,
                                    'logger': logger,
                                    'ctc_beam_width': args.ctc_beam_width,
                                    'seq2seq_beam_width': args.seq2seq_beam_width,
                                    'valid_frac': args.valid_frac,
                                    'valid_epochs': args.valid_epochs,
                                    'print_batch': args.print_batch,
                                    'build_image_context':args.build_image_context,
                                    'sum_logloss': False, ## setting to true equalizes ctc_loss and log_loss if y_s == squashed_seq
                                    'dropout': None if args.keep_prob >= 1.0 else tfc.DropoutParams({'keep_prob': args.keep_prob}),
                                    'MeanSumAlphaEquals1': False,
                                    'pLambda': 0.005,
                                    'make_training_accuracy_graph': False,
                                    'use_ctc_loss': args.use_ctc_loss,
                                    "swap_memory": args.swap_memory,
                                    'tf_session_allow_growth': False,
                                    'restore_from_checkpoint': args.restore_logdir is not None,
                                    'num_gpus': 2
                                    })
    if args.batch_size is not None:
        globalParams.B = args.batch_size
    if args.partial_batch:
        globalParams.assert_whole_batch = False
    if args.queue_capacity is not None:
        globalParams.input_queue_capacity = args.queue_capacity
    if args.alpha is not None:
        globalParams.adam_alpha = args.alpha
    if args.rLambda is not None:
        globalParams.rLambda = args.rLambda
        
    hyper = hyper_params.make_hyper(globalParams)

    # Add logging file handler now that we have instantiated hyperparams.
    if args.restore_logdir is not None:
        globalParams.logdir = args.restore_logdir
    else:
        globalParams.logdir = tfc.makeTBDir(hyper.tb)
    fh = logging.FileHandler(os.path.join(globalParams.logdir, 'training.log'))
    fh.setFormatter(hyper_params.makeFormatter())
    logger.addHandler(fh)
    hyper_params.setLogLevel(logger, args.logging_level)

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
