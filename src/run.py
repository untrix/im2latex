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
import train
import logging
import hyper_params
import argparse
import os

def main():
    _data_folder = '../data/generated2'

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
                        default=4)
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

    ## Logger
    logger = logging.getLogger()
    logging_level = (logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    globalParams = dlc.Properties({
                                    'print_steps': args.print_steps,
                                    'num_steps': args.num_steps,
                                    'num_epochs': args.num_epochs,
                                    'logger': logger,
                                    'beam_width':args.beam_width,
                                    'valid_frac': args.valid_frac,
                                    'valid_epochs': args.valid_epochs,
                                    'print_batch': args.print_batch,
                                    'build_image_context':False,
                                    'sum_logloss': False, ## setting to true equalizes ctc_loss and log_loss if y_s == squashed_seq
                                    'dropout': None if args.keep_prob >= 1.0 else tfc.DropoutParams({'keep_prob': args.keep_prob}),
                                    'pLambda': 0.0005,
                                    'MeanSumAlphaEquals1': False
                                    })    
    if args.batch_size is not None:
        globalParams.B = args.batch_size
    if args.partial_batch:
        globalParams.assert_whole_batch = False
    if args.queue_capacity is not None:
        globalParams.input_queue_capacity = args.queue_capacity
    if args.alpha is not None:
        globalParams.adam_alpha = args.alpha
    globalParams.update({
                        })
    hyper = hyper_params.make_hyper(globalParams)

    # Add logging file handler now that we have instantiated hyperparams.
    globalParams.logdir = tfc.makeTBDir(hyper.tb)
    fh = logging.FileHandler(os.path.join(globalParams.logdir, 'training.log'))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging_level[args.logging_level - 1])

    logger.info('\n#################### Default Param Overrides: ####################\n%s',globalParams)
    logger.info('##################################################################\n')
    logger.info( '\n#########################  Hyper-params: #########################\n%s', hyper)
    logger.info('##################################################################\n')
    
    train.train(raw_data_folder,
          vgg16_folder,
          globalParams,
          hyper.freeze()
         )

main()
