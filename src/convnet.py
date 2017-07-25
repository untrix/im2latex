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
Created on Mon Jul 24 12:28:55 2017

@author: Sumeet S Singh
"""

import os
import time
from six.moves import cPickle as pickle
import tensorflow as tf
from keras import backend as K
from keras.preprocessing import image
from Im2LatexModel_2 import build_image_context
from data_reader import BatchIterator, ImageProcessor, ImagenetProcessor
from hyper_params import HYPER as HYPER_
import tf_commons as tfc

HYPER = HYPER_.copy({'B':128})
data_folder = '../data/generated2'
image_folder = os.path.join(data_folder,'formula_images')
raw_data_folder = os.path.join(data_folder, 'training')
image_features_folder = os.path.join(raw_data_folder, 'vgg16_features')

b_it = BatchIterator(raw_data_folder, image_folder, 
                     batch_size=HYPER.B, 
                     image_processor=ImagenetProcessor())

graph = tf.Graph()
with graph.as_default():

    ## config=tf.ConfigProto(log_device_placement=True)
    ## config.gpu_options.allow_growth = True
    tf_session = tf.Session(config=None)
    with tf_session.as_default():
        K.set_session(tf_session)
        
        tf_im = tf.placeholder(dtype=HYPER.dtype, shape=((HYPER.B,)+HYPER.image_shape), name='image')
        tf_a_batch = build_image_context(HYPER, tf_im)
        tf_a_list = tf.unstack(tf_a_batch, axis=0)
    
        tfc.printVars('Trainable Variables', tf.trainable_variables())
        tfc.printVars('Global Variables', tf.global_variables())
        tfc.printVars('Local Variables', tf.local_variables())
    
        print '\nUninitialized params'
        print tf_session.run(tf.report_uninitialized_variables())
        
        print 'Flushing graph to disk'
        tf_sw = tf.summary.FileWriter(tfc.makeTBDir(HYPER.tb), graph=graph)
        tf_sw.flush()

        print '\n'
        start_time = time.clock()
        for step, b in enumerate(b_it, start=1):
            if b.epoch > 1:
                break
            feed_dict = {tf_im:b.im, K.learning_phase(): 0}
            a_list = tf_session.run(tf_a_list, feed_dict = feed_dict)
            assert len(a_list) == len(b.image_name)
            for i, a in enumerate(a_list):
                ## print 'Writing %s, shape=%s'%(b.image_name[i], a.shape)
                with open(os.path.join(image_features_folder, os.path.splitext(b.image_name[i])[0] + '.pkl'),
                          'wb') as f:
                  pickle.dump(a, f, pickle.HIGHEST_PROTOCOL)
            if step % 10 == 0:
                print('Elapsed time for %d steps = %d seconds'%(step, time.clock()-start_time))
        print('Elapsed time for %d steps = %d seconds'%(step-1, time.clock()-start_time))
        print 'done'
              