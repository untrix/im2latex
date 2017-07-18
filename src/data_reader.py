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
import pandas as pd
import os
from six.moves import cPickle as pickle
import dl_commons as dlc
import threading
import numpy as np
from scipy import ndimage

def get_image_matrix(image_file_, height_, width_, padded_dim_):
    padded_height = padded_dim_['height']
    padded_width = padded_dim_['width']
    MAX_PIXEL = 255.0 # Make sure this is a float literal
    ## Load image and convert to a 3-channel array
    im_ar = ndimage.imread(os.path.join(image_file_), mode='RGB')
    ## normalize values to lie between -1.0 and 1.0.
    ## This is done in place of data whitening - i.e. normalizing to mean=0 and std-dev=0.5
    ## Is is a very rough technique but legit for images
    im_ar = (im_ar - MAX_PIXEL/2.0) / MAX_PIXEL
    height, width, channels = im_ar.shape
    assert height == height_
    assert width == width_
    assert channels == 3
    if (height < padded_height) or (width < padded_width):
        ar = np.full((padded_height, padded_width, channels), 0.5, dtype=np.float32)
        h = (padded_height - height)//2
        ar[h:h+height, 0:width] = im_ar
        im_ar = ar

    return im_ar

def make_batch_list(df_, batch_size_):
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
        assert (bin_counts[i] % batch_size_) == 0
        batch_list.extend([(bin_, j) for j in range(num_batches)])

    np.random.shuffle(batch_list)
    return batch_list

class ShuffleIterator(object):
    def __init__(self, df_, batch_size_):
        self._df = df_.sample(frac=1)
        self._batch_size = batch_size_
        self._batch_list = make_batch_list(self._df, batch_size_)
        self._next_pos = 0
        self._num_items = (df_.shape[0] // batch_size_)
        self._iter = 0
        self._epoch = 1
        self.lock = threading.Lock()
        
    ##def __iter__(self):
    ##    return self
    
    def next(self):
        ## This is an infinite iterator
        with self.lock:
            if self._next_pos >= self._num_items:
                ## Reshuffle sample-to-batch assignment
                self._df = self._df.sample(frac=1)
                ## Reshuffle the bin-batch list
                np.random.shuffle(self._batch_list)
                ## self._batch_list = make_batch_list(self._df, batch_size_)
                self._next_pos = 0
                self._iter = 0
                print 'ShuffleIterator finished epoch %d'%self._epoch
                self._epoch += 1
            next_pos = self._next_pos
            self._next_pos += 1
            self._iter += 1
        
        batch = self._batch_list[next_pos]
        print 'ShuffleIterator epoch %d, iter %d, bin-batch id %s'%(self._epoch, self._iter, batch)
        df_bin = self._df[self._df.bin_len == batch[0]]
        assert df_bin.bin_len.iloc[batch[1]*self._batch_size] == batch[0]
        assert df_bin.bin_len.iloc[(batch[1]+1)*self._batch_size-1] == batch[0]
        return df_bin.iloc[batch[1]*self._batch_size : (batch[1]+1)*self._batch_size]

class BatchIterator(ShuffleIterator):
    def __init__(self, raw_data_dir_, image_dir_):
        self._padded_im_dim = pd.read_pickle(os.path.join(raw_data_dir_, 'padded_image_dim.pkl'))
        self._image_dir = image_dir_
        self._seq_data = pd.read_pickle(os.path.join(raw_data_dir_, 'raw_seq_train.pkl'))
        df = pd.read_pickle(os.path.join(raw_data_dir_, 'df_train.pkl'))
        batch_size = pd.read_pickle(os.path.join(raw_data_dir_, 'batch_size.pkl'))
        ShuffleIterator.__init__(self, df, batch_size)

    def next(self):
        df_batch = ShuffleIterator.next(self)[['image', 'height', 'width', 'bin_len']]
        im_batch = [
            get_image_matrix(os.path.join(self._image_dir, row[0]), row[1], row[2], self._padded_im_dim)
            for row in df_batch.itertuples(index=False)
        ]   
        im_batch = np.asarray(im_batch)
        
        bin_len = df_batch.bin_len.iloc[0]
        seq_batch = self._seq_data[bin_len].loc[df_batch.index].values
        return dlc.Properties({'im':im_batch, 
                               'seq':seq_batch})

#data_folder = '../data/generated2'
#image_folder = os.path.join(data_folder,'formula_images')
#raw_data_folder = os.path.join(data_folder, 'training')
#print ('starting')
#it = BatchIterator(raw_data_folder, image_folder)
#print ('created batch iterator')
#print it.next()