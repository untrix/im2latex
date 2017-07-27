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
import tensorflow as tf
import tf_commons as tfc
from Im2LatexModel import Im2LatexModel
from keras import backend as K
import hyper_params
from data_reader import BatchContextIterator, BatchImageIterator


def train_old(batch_iterator, HYPER, num_steps, print_steps, num_epochs):
    graph = tf.Graph()
    with graph.as_default():
        model = Im2LatexModel(HYPER)
        train_ops = model.build_train_graph()
        
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
            if var.name.startswith('VGGNet/'):
                total_vggnet += n
            elif 'CALSTM' in var.name:
                total_calstm += n
            elif var.name.startswith('Im2LatexRNN/Output_Layer/'):
                total_output += n
            elif var.name.startswith('Initializer_MLP/'):
                total_init += n
            elif var.name.startswith('Im2LatexRNN/Ey/Embedding_Matrix'):
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

#        print '\nTrainable Variables Initializers'
#        for var in tf.trainable_variables():
#            print var.initial_value
            
        config=tf.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as session:
            print 'Flushing graph to disk'
            tf_sw = tf.summary.FileWriter(tfc.makeTBDir(HYPER.tb), graph=graph)
            tf_sw.flush()
            tf.global_variables_initializer().run()
        
            if batch_iterator is None or num_steps == 0:
                return
            
            if num_epochs > 0:
                num_epoch_steps = num_epochs * batch_iterator.epoch_size
            else:
                num_epoch_steps = -1

            if num_steps < 0:
                num_steps = num_epoch_steps
            elif num_epoch_steps >= 0 and (num_epoch_steps < num_steps):
                num_steps = num_epoch_steps

            start_time = time.time()
            for b in batch_iterator:
                feed = {train_ops.y_s: b.y_s, train_ops.seq_lens: b.seq_len, train_ops.im: b.im}
                session.run(train_ops.train, feed_dict=feed)
                if (num_steps >= 0) and (b.step >= num_steps):
                    break
                if b.step % print_steps == 0:
                    print 'Elapsed time for %d steps = %f'%(b.step, time.time()-start_time)
            print 'Elapsed time for %d steps = %f'%(b.step, time.clock()-start_time)

                
def train(batch_iterator, HYPER, num_steps, print_steps, num_epochs):
    graph = tf.Graph()
    with graph.as_default():
        model = Im2LatexModel(HYPER)
        train_ops = model.build_train_graph()
        enqueue_op = train_ops.inp_q.enqueue(batch_iterator.get_pyfunc())
        qr = tf.train.QueueRunner(train_ops.inp_q, [enqueue_op])
        coord = tf.train.Coordinator()
        
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
            if var.name.startswith('VGGNet/'):
                total_vggnet += n
            elif 'CALSTM' in var.name:
                total_calstm += n
            elif var.name.startswith('Im2LatexRNN/Output_Layer/'):
                total_output += n
            elif var.name.startswith('Initializer_MLP/'):
                total_init += n
            elif var.name.startswith('Im2LatexRNN/Ey/Embedding_Matrix'):
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

#        print '\nTrainable Variables Initializers'
#        for var in tf.trainable_variables():
#            print var.initial_value
            
        config=tf.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as session:
            print 'Flushing graph to disk'
            tf_sw = tf.summary.FileWriter(tfc.makeTBDir(HYPER.tb), graph=graph)
            tf_sw.flush()
            tf.global_variables_initializer().run()
            
#            num_steps = num_steps_to_run(num_steps, num_epochs, batch_iterator.epoch_size)
            enqueue_threads = qr.create_threads(session, coord=coord, start=True)
            
            start_time = time.time()
#            for step in xrange(1,num_steps+1):
            step = 0
            try:
                while not coord.should_stop():
                    session.run(train_ops.train)
                    step += 1
                    if step % print_steps == 0:
                        print 'Elapsed time for %d steps = %f'%(step, time.time()-start_time)
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
    parser.add_argument("--print-steps", "-s", dest="print_steps", type=int,
                        help="Number of training steps after which to log results. Defaults to 10 if unspecified", 
                        default=10)
    parser.add_argument("--batch-size", "-b", dest="batch_size", type=int,
                        help="Batchsize. If unspecified, defaults to the default value in hyper_params",
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

    hyperVals = {'build_image_context':False}
    if args.batch_size is not None:
        hyperVals['B'] = args.batch_size
    if args.partial_batch:
        hyperVals['assert_whole_batch'] = False
    if args.queue_capacity is not None:
        hyperVals['input_queue_capacity'] = args.queue_capacity

    HYPER = hyper_params.make_hyper(hyperVals)
    b_it = BatchContextIterator(raw_data_folder, vgg16_folder, HYPER,
                                args.num_steps, args.num_epochs)
    train(b_it, HYPER, args.num_steps, args.print_steps, args.num_epochs)
    
main()
