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

Created on Mon Jul 17 19:58:00 2017

@author: Sumeet S Singh
"""
import os
import collections
import pandas as pd
from six.moves import cPickle as pickle
import dl_commons as dlc
import threading
import numpy as np
from scipy import ndimage
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input


class ImageProcessor(object):
    def __init__(self, params):
        self._params=params
        
    def get_array(self, image_file_, height_, width_, padded_dim_):
        padded_height = padded_dim_['height']
        padded_width = padded_dim_['width']
        ## Load image and convert to a 3-channel array
        im_ar = ndimage.imread(os.path.join(image_file_), mode='RGB')
        height, width, channels = im_ar.shape
        assert height == height_
        assert width == width_
        assert channels == 3
        if (height < padded_height) or (width < padded_width):
            ar = np.full((padded_height, padded_width, channels), 255.0, dtype=self._params.dtype_np)
            h = (padded_height - height)//2
            ar[h:h+height, 0:width] = im_ar
            im_ar = ar
    
        return im_ar

    @staticmethod
    def whiten(image_ar):
        """
        normalize values to lie between -1.0 and 1.0.
        This is done in place of data whitening - i.e. normalizing to mean=0 and std-dev=0.5
        Is is a very rough technique but legit for images. We assume that the mean is 255/2
        and therefore substract 127.5 from all values. Then we divid everything by 255 to ensure
        that all values lie between -0.5 and 0.5
        Arguments:
            image_batch: (ndarray) Batch of images or a single image. Shape doesn't matter.
        """
        MAX_PIXEL = 255.0
        return (image_ar - 127.5) / 255.0
    

class ImagenetProcessor(ImageProcessor):
    def __init__(self, params):
        ImageProcessor.__init__(self, params)
        
    @staticmethod
    def whiten(image_ar):
        """
        Run Imagenet preprocessing - 
        1) flip RGB to BGR
        2) Adjust mean per Imagenet stats
        3) No std-dev adjustment
        Arguments:
            image_batch: (ndarray) Batch of images of shape (B, H, W, D) - i.e. 'channels-last' format.
            Also, must have 3 channels in order 'RGB' (i.e. mode='RGB')
        """
        return preprocess_input(image_ar, data_format='channels_last')

class VGGProcessor(object):
    def __init__(self, vgg_dir_):
        self._vgg_dir = vgg_dir_
        
    def get_array(self, image_file_):
        pkl_file = os.path.join(self._vgg_dir, os.path.splitext(image_file_)[0] + '.pkl')
        return pd.read_pickle(pkl_file)
        
def make_batch_list(df_, batch_size_, assert_whole_batch=True):
    ## Make a list of batches
    bin_lens = sorted(df_.bin_len.unique())
    bin_counts = [df_[df_.bin_len==l].shape[0] for l in bin_lens]
    batch_list = []
    for i in range(len(bin_lens)):
        bin_ = bin_lens[i]
        num_batches = (bin_counts[i] // batch_size_)
        ## Just making sure bin size is integral multiple of batch_size.
        ## This is not a requirement for this function to operate, rather
        ## is a way of possibly catching data-corrupting bugs
        if assert_whole_batch:
            assert (bin_counts[i] % batch_size_) == 0
        batch_list.extend([(bin_, j) for j in range(num_batches)])

    np.random.shuffle(batch_list)
    return batch_list

class ShuffleIterator(object):
    def __init__(self, df_, hyper, num_steps, num_epochs):
        self._df = df_.sample(frac=1)
        self._batch_size = hyper.B
        self._batch_list = make_batch_list(self._df, self._batch_size, hyper.assert_whole_batch)
        self._next_pos = 0
        self._num_items = len(self._batch_list)
        self._step = 0
        self._epoch = 1
        self._max_steps = self.num_steps_to_run(num_steps, num_epochs, self._num_items)
        self.lock = threading.Lock()
        
        print 'ShuffleIterator initialized with batch_size = %d, steps-per-epoch = %d'%(self._batch_size, 
                                                                                        self._num_items)
    def __iter__(self):
        return self
    @property
    def batch_size(self):
        return self._batch_size
    @property
    def epoch_size(self):
        return self._num_items
    @property
    def steps_completed(self):
        return self._step
    @property
    def max_steps(self):
        return self._max_steps
    @staticmethod
    def num_steps_to_run(num_steps, num_epochs, epoch_size):
        """
        Calculates num_steps to run, based on num_steps and num_epoch arguments.
        A negative value for either num_steps or num_epochs implies infinity.
        num_epochs will be converted into steps and the smaller of num_steps and num_epoch_steps
        will be returned.
        """
        if num_epochs > 0:
            num_epoch_steps = num_epochs * epoch_size
        else:
            num_epoch_steps = -1
    
        if num_steps < 0:
            num_steps = num_epoch_steps
        elif num_epoch_steps >= 0 and (num_epoch_steps < num_steps):
            num_steps = num_epoch_steps
            
        return num_steps

    def next(self):
        ## This is an infinite iterator
        with self.lock:
            if self._next_pos >= self._num_items:
                ## Reshuffle sample-to-batch assignment
                self._df = self._df.sample(frac=1)
                ## Reshuffle the bin-batch list
                np.random.shuffle(self._batch_list)
                ## self._batch_list = make_batch_list(self._df, batch_size_, hyper.assert_whole_batch)
                self._next_pos = 0
                print 'ShuffleIterator finished epoch %d'%self._epoch
                self._epoch += 1
            next_pos = self._next_pos
            epoch= self._epoch
            self._next_pos += 1
            self._step += 1
        
        batch = self._batch_list[next_pos]
        ## print 'ShuffleIterator epoch %d, step %d, bin-batch idx %s'%(self._epoch, self._step, batch)
        df_bin = self._df[self._df.bin_len == batch[0]]
        assert df_bin.bin_len.iloc[batch[1]*self._batch_size] == batch[0]
        assert df_bin.bin_len.iloc[(batch[1]+1)*self._batch_size-1] == batch[0]
        return dlc.Properties({
                'df_batch': df_bin.iloc[batch[1]*self._batch_size : (batch[1]+1)*self._batch_size],
                'epoch': epoch,
                'step': self._step,
                'batch_idx': batch
                })

class BatchImageIterator1(ShuffleIterator):
    def __init__(self, raw_data_dir_, image_dir_, 
                 hyper, 
                 image_processor,
                 df):
        
        self._padded_im_dim = pd.read_pickle(os.path.join(raw_data_dir_, 'padded_image_dim.pkl'))
        self._image_dir = image_dir_
        self._image_processor = image_processor ## ImageProcessor(hyper)
        self._seq_data = pd.read_pickle(os.path.join(raw_data_dir_, 'raw_seq_train.pkl'))
        df = df ## pd.read_pickle(os.path.join(raw_data_dir_, 'df_train.pkl'))
        ShuffleIterator.__init__(self, df, hyper)

    def next(self):
        nxt = ShuffleIterator.next(self)
        df_batch = nxt.df_batch[['image', 'height', 'width', 'bin_len', 'seq_len']]
        im_batch = [
            self._image_processor.get_array(os.path.join(self._image_dir, row[0]), row[1], row[2], self._padded_im_dim)
            for row in df_batch.itertuples(index=False)
        ]
        im_batch = self._image_processor.whiten(np.asarray(im_batch))
        
        bin_len = df_batch.bin_len.iloc[0]
        y_s = self._seq_data[bin_len].loc[df_batch.index].values
        return dlc.Properties({'im':im_batch, 
                               'y_s':y_s,
                               'seq_len': df_batch.seq_len.values,
                               'image_name': df_batch.image.values,
                               'epoch': nxt.epoch,
                               'step': nxt.step
                               })

class BatchImageIterator2(ShuffleIterator):
    def __init__(self,
                raw_data_dir_,
                image_dir_, 
                hyper, 
                image_processor,
                df,
                num_steps=-1,
                num_epochs=-1
                ):
        
        self._padded_im_dim = pd.read_pickle(os.path.join(raw_data_dir_, 'padded_image_dim.pkl'))
        self._image_dir = image_dir_
        self._image_processor = image_processor ## ImageProcessor(hyper)
        df = df ## pd.read_pickle(os.path.join(raw_data_dir_, 'df_train.pkl'))
        ShuffleIterator.__init__(self, df, hyper, num_steps, num_epochs)

    def next(self):
        if self.steps_completed >= self.max_steps:
            raise StopIteration('Max steps executed (%d)'%self.steps_completed)

        nxt = ShuffleIterator.next(self)
        df_batch = nxt.df_batch[['image', 'height', 'width']]
        im_batch = [
            self._image_processor.get_array(os.path.join(self._image_dir, row[0]), row[1], row[2], self._padded_im_dim)
            for row in df_batch.itertuples(index=False)
        ]
        im_batch = self._image_processor.whiten(np.asarray(im_batch))
        
        return dlc.Properties({'im':im_batch, 
                               'image_name': df_batch.image.values,
                               'epoch': nxt.epoch,
                               'step': nxt.step
                               })


InpTup = collections.namedtuple('InpTup', ('y_s', 'seq_len', 'y_ctc', 'ctc_len', 'im'))
class BatchContextIterator(ShuffleIterator):
    def __init__(self, 
                 raw_data_dir_,
                 image_feature_dir_,
                 hyper,
                 num_steps=-1,
                 num_epochs=-1,
                 image_processor_=None):
        self._hyper = hyper
        self._raw_data_dir = raw_data_dir_
        self._image_feature_dir = image_feature_dir_
        self._image_processor = image_processor_ or VGGProcessor(image_feature_dir_)
        self._seq_data = pd.read_pickle(os.path.join(raw_data_dir_, 'raw_seq_train.pkl'))
        self._ctc_seq_data = pd.read_pickle(os.path.join(raw_data_dir_, 'raw_seq_sq_train.pkl'))
        df = pd.read_pickle(os.path.join(raw_data_dir_, 'df_train.pkl'))
        ShuffleIterator.__init__(self, df, hyper, num_steps, num_epochs)

    def next(self):
        if self.steps_completed >= self.max_steps:
            raise StopIteration('Max steps executed (%d)'%self.steps_completed)

        nxt = ShuffleIterator.next(self)
        df_batch = nxt.df_batch[['image', 'bin_len', 'seq_len', 'squashed_len']]
        a_batch = [
            self._image_processor.get_array(row[0]) for row in df_batch.itertuples(index=False)
        ]
        a_batch = np.asarray(a_batch)
        
        bin_len = df_batch.bin_len.iloc[0]
        ##y_s = np.asarray(self._seq_data[bin_len].loc[df_batch.index].values, 
        ##                 dtype=self._hyper.int_type_np)
        y_ctc = np.asarray(self._ctc_seq_data[bin_len].loc[df_batch.index].values, 
                           dtype=self._hyper.int_type_np)
        ctc_len = np.asarray(df_batch.squashed_len.values, dtype=self._hyper.int_type_np)
        return dlc.Properties({'im':a_batch, 
                               ##'y_s':y_s,
                               ##'seq_len': np.asarray(df_batch.seq_len.values, dtype=self._hyper.int_type_np),
                               'y_s':y_ctc,
                               'seq_len': ctc_len,
                               'y_ctc':y_ctc,
                               'ctc_len': ctc_len,
                               'image_name': df_batch.image.values,
                               'epoch': nxt.epoch,
                               'step': nxt.step
                               })
    
    def get_pyfunc(self):
        def func(x=None):
            d = self.next()
            return InpTup(d.y_s, d.seq_len, d.y_ctc, d.ctc_len, d.im)
        
        int_type = self._hyper.int_type
        return tf.py_func(func,[0],[int_type, int_type, int_type, int_type, self._hyper.dtype])
    
