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
import logging
import pandas as pd
from six.moves import cPickle as pickle
import dl_commons as dlc
import data_commons as dtc
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


class ImageProcessor3(object):
    def __init__(self, params, image_dir_, grayscale):
        self._params=params
        self._image_dir = image_dir_
        self._mode = 'L' if grayscale else 'RGB'
        self._channels = 1 if grayscale else 3

    def get_array(self, image_name_, height_, width_, padded_dim_):
        image_file = os.path.join(self._image_dir, image_name_)
        padded_height = padded_dim_['height']
        padded_width = padded_dim_['width']
        ## Load image and convert to a n-channel array
        im_ar = ndimage.imread(os.path.join(image_file), mode= self._mode)
        if len(im_ar.shape) == 2:
            im_ar = np.expand_dims(im_ar, 2)

        height, width, channels = im_ar.shape
        assert height == height_, 'image height = %d instead of %d'%(height_, height)
        assert width == width_, 'image width = %d instead of %d'%(width_, width)
        assert channels == self._channels, 'image channels = %d instead of %d'%(self._channels, channels)
        if (height < padded_height) or (width < padded_width):
            ar = np.full((padded_height, padded_width, channels), 255.0, dtype=self._params.dtype_np)
            h = (padded_height - height) // 2
            w = (padded_width - width) // 2
            ar[h:h+height, w:w+width] = im_ar
            im_ar = ar
        return im_ar

    def whiten(self, image_ar):
        """
        normalize values to lie between -1.0 and 1.0.
        This is done in place of data whitening - i.e. normalizing to mean=0 and std-dev=0.5
        Is is a very rough technique but legit for images. We assume that the mean is 255/2
        and therefore substract 127.5 from all values. Then we divid everything by 255 to ensure
        that all values lie between -0.5 and 0.5
        Arguments:
            image_batch: (ndarray) Batch of images or a single image. Shape doesn't matter.
        """
        assert image_ar.shape == (self._params.data_reader_B,) + self._params.image_shape, \
            'Got image shape %s instead of %s'%(image_ar.shape, (self._params.data_reader_B,) + self._params.image_shape)
        return (image_ar - 127.5) / 255.0

# class ImageProcessor3_RGB(ImageProcessor3):
#     def __init__(self, params, image_dir_):
#         ImageProcessor3.__init__(self, params, image_dir_, grayscale=False)


class ImagenetProcessor3(ImageProcessor3):
    def __init__(self, params, image_dir_):
        ImageProcessor3.__init__(self, params, image_dir_, grayscale=False)

    # def __init__(self, params, image_dir_):
    #     ImageProcessor3_RGB.__init__(self, params, image_dir_)

    def whiten(self, image_ar):
        """
        Run Imagenet preprocessing -
        1) flip RGB to BGR
        2) Adjust mean per Imagenet stats
        3) No std-dev adjustment
        Arguments:
            image_batch: (ndarray) Batch of images of shape (B, H, W, D) - i.e. 'channels-last' format.
            Also, must have 3 channels in order 'RGB' (i.e. mode='RGB')
        """
        assert image_ar.shape == (self._params.data_reader_B,) + self._params.image_shape, 'Got image shape %s instead of %s'%(image_ar.shape, (self._params.data_reader_B,) + self._params.image_shape)
        return preprocess_input(image_ar, data_format='channels_last')


class ImageProcessor3_BW(ImageProcessor3):
    def __init__(self, params, image_dir_):
        ImageProcessor3.__init__(self, params, image_dir_, grayscale=True)

class VGGProcessor(object):
    def __init__(self, vgg_dir_):
        self._vgg_dir = vgg_dir_

    def get_array(self, image_file_, height_=None, width_=None, padded_dim_=None):
        """
        Simply returns the contents of the pickled ndarray file - image_file_.pkl.
        height_, width_ and padded_dim_ are unused.
        """
        pkl_file = os.path.join(self._vgg_dir, os.path.splitext(image_file_)[0] + '.pkl')
        return pd.read_pickle(pkl_file)

    @staticmethod
    def whiten(image_ar):
        """Is a No-Op. Returns image_ar as-is, right back."""
        return image_ar


def make_batch_list(df_, batch_size_, assert_whole_batch=True):
    # Shuffle the dataframe
    df_ = df_.sample(frac=1)

    # Make a list of batches indices
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
            assert (bin_counts[i] % batch_size_) == 0, 'bin_counts[%d]=%d is not a whole multiple of batch_size=%d'%(i, bin_counts[i], batch_size_)
        # elif (bin_counts[i] % batch_size_) != 0:
        #     shortfall = batch_size_ - (bin_counts[i] % batch_size_)
        #     dtc.logger.warn('Adding %d samples to bin %d. Final size = %d', shortfall, bin_, bin_counts[i]+shortfall)

        batch_list.extend([(bin_, j) for j in range(num_batches)])

    ## Shuffle the bin-batch list
    np.random.shuffle(batch_list)
    return batch_list


class ShuffleIterator(object):
    def __init__(self, df_, hyper, num_steps, num_epochs, name='ShuffleIterator'):
        self._df = df_.sample(frac=1)
        self._batch_size = hyper.data_reader_B
        self._batch_list = make_batch_list(self._df, self._batch_size, hyper.assert_whole_batch)
        self._next_pos = 0
        self._num_items = len(self._batch_list)
        self._step = 0
        self._epoch = 1
        self._max_steps = self.num_steps_to_run(num_steps, num_epochs, self._num_items)
        self.lock = threading.RLock()
        self._hyper = hyper
        self._name = name

        self._hyper.logger.info('%s initialized with batch_size = %d, steps-per-epoch = %d, max-steps = %d', 
                                self._name,
                                self._batch_size,
                                self._num_items,
                                self._max_steps)
    def __iter__(self):
        return self
    @property
    def name(self):
        return self._name
    @property
    def batch_size(self):
        return self._batch_size
    @property
    def epoch_size(self):
        return self._num_items
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
        with self.lock:
            if (self.max_steps >= 0) and (self._step >= self.max_steps):
                raise StopIteration('Max steps executed (%d)'%self._step)

            if self._next_pos >= self._num_items:
                ## Reshuffle sample-to-batch assignment
                self._df = self._df.sample(frac=1)
                ## Reshuffle the bin-batch list
                np.random.shuffle(self._batch_list)
                ## Shuffle the bin composition too
                self._df = self._df.sample(frac=1)
                self._next_pos = 0
                self._hyper.logger.debug('%s finished epoch %d'%(self._name, self._epoch))
                self._epoch += 1
            curr_pos = self._next_pos
            self._next_pos += 1 # value for next iteration
            epoch= self._epoch
            self._step += 1
            curr_step = self._step

        batch = self._batch_list[curr_pos]
        self._hyper.logger.debug('%s epoch %d, step %d, bin-batch idx %s', self._name, self._epoch, self._step, batch)
        df_bin = self._df[self._df.bin_len == batch[0]]
        assert df_bin.bin_len.iloc[batch[1]*self._batch_size] == batch[0]
        assert df_bin.bin_len.iloc[(batch[1]+1)*self._batch_size-1] == batch[0]
        return dlc.Properties({
                'df_batch': df_bin.iloc[batch[1]*self._batch_size : (batch[1]+1)*self._batch_size],
                'epoch': epoch,
                'step': curr_step,
                'batch_idx': batch
                })

class BatchImageIterator1(ShuffleIterator):
    def __init__(self, raw_data_dir_, image_dir_,
                 hyper,
                 image_processor,
                 df,
                 seq_fname='raw_seq_train.pkl'):

        self._padded_im_dim = pd.read_pickle(os.path.join(raw_data_dir_, 'data_props.pkl'))['padded_image_dim']
        self._image_dir = image_dir_
        self._image_processor = image_processor ## ImageProcessor(hyper)
        self._seq_data = pd.read_pickle(os.path.join(raw_data_dir_, seq_fname))

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

        self._padded_im_dim = pd.read_pickle(os.path.join(raw_data_dir_, 'data_props.pkl'))['padded_image_dim']
        self._image_dir = image_dir_
        self._image_processor = image_processor ## ImageProcessor(hyper)

        ShuffleIterator.__init__(self, df, hyper, num_steps, num_epochs)

    def next(self):
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


InpTup = collections.namedtuple('InpTup', ('y_s', 'seq_len', 'y_ctc', 'ctc_len', 'im', 'image_name'))
class BatchImageIterator3(ShuffleIterator):
    def __init__(self,
                 df,
                 raw_data_dir_,
                 seq_fname,  # ='raw_seq_train.pkl',
                 sq_seq_fname,  # ='raw_seq_sq_train.pkl',
                 hyper,
                 num_steps=-1,
                 num_epochs=-1,
                 image_processor_=None,
                 name='BatchImageIterator'
                 ):
        self._hyper = hyper
        self._raw_data_dir = raw_data_dir_
        assert image_processor_ is not None
        self._image_processor = image_processor_
        image_shape = hyper.image_shape
        self._padded_im_dim = {'height': image_shape[0], 'width': image_shape[1]}
        self._seq_data = pd.read_pickle(os.path.join(raw_data_dir_, seq_fname))
        self._ctc_seq_data = pd.read_pickle(os.path.join(raw_data_dir_, sq_seq_fname))
        ShuffleIterator.__init__(self, df, hyper, num_steps, num_epochs, name)

    def next(self):
        nxt = ShuffleIterator.next(self)
        # df_batch = nxt.df_batch[['image', 'bin_len', 'seq_len', 'squashed_len']]
        # a_batch = [
        #     self._image_processor.get_array(row[0]) for row in df_batch.itertuples(index=False)
        # ]
        # a_batch = np.asarray(a_batch)
        df_batch = nxt.df_batch[['image', 'height', 'width', 'bin_len', 'seq_len', 'squashed_len']]
        im_batch = [
            self._image_processor.get_array(row[0], row[1], row[2], self._padded_im_dim)
            for row in df_batch.itertuples(index=False)
        ]
        im_batch = self._image_processor.whiten(np.asarray(im_batch))

        bin_len = df_batch.bin_len.iloc[0]
        y_ctc = np.asarray(self._ctc_seq_data[bin_len].loc[df_batch.index].values, dtype=self._hyper.int_type_np)
        ctc_len = np.asarray(df_batch.squashed_len.values, dtype=self._hyper.int_type_np)
        if self._hyper.squash_input_seq:
            y_s = y_ctc
            seq_len = ctc_len
        else:
            y_s   = np.asarray(self._seq_data[bin_len].loc[df_batch.index].values,     dtype=self._hyper.int_type_np)
            seq_len = np.asarray(df_batch.seq_len.values,      dtype=self._hyper.int_type_np)

        return dlc.Properties({'im':im_batch,
                               'y_s':y_s, # (B,T)
                               'seq_len': seq_len, #(B,)
                               ## 'y_s':y_ctc,
                               ## 'seq_len': ctc_len,
                               'y_ctc':y_ctc, #(B,T)
                               'ctc_len': ctc_len, #(B,)
                               'image_name': df_batch.image.values, #(B,)
                               'epoch': nxt.epoch, # scalar
                               'step': nxt.step # scalar
                               })

    @property
    def out_tup_types(self):
        int_type = self._hyper.int_type
        return (int_type, int_type, int_type, int_type, self._hyper.dtype, tf.string)

    def get_pyfunc(self):
        def func(x=None):
            d = self.next()
            return InpTup(d.y_s, d.seq_len, d.y_ctc, d.ctc_len, d.im, d.image_name)

        return tf.py_func(func, [0], self.out_tup_types)

    def get_pyfunc_with_split(self, num_splits):
        def split(a, num, size):
            """
            Splits a numpy array into num slices along dimension 0 and wraps them
            with an additional dimension (0) of size num. For e.g. if a is shape (10, ...)
            then splitting it into 5 splits will result in a array of shape (5,2,...).
            """
            return np.asarray([ a[i*size:(i+1)*size] for i in range(num) ])

        def func(x=None):
            d = self.next()
            # Note: tf.FIFOQueue.enqueue_many will separate the splits into multiple tuples.
            return InpTup(split(d.y_s,      num_splits,split_size), 
                          split(d.seq_len,  num_splits,split_size), 
                          split(d.y_ctc,    num_splits,split_size), 
                          split(d.ctc_len,  num_splits,split_size), 
                          split(d.im,       num_splits,split_size),
                          split(d.image_name,  num_splits,split_size))

        split_size = self._batch_size // num_splits
        assert (self._batch_size / num_splits) == split_size, 'Batchsize:%d is not divisible by num_splits: %d'%(self._batch_size, num_splits)
        int_type = self._hyper.int_type
        return tf.py_func(func, [0], self.out_tup_types)

class BatchContextIterator(BatchImageIterator3):
    def __init__(self,
                 df,
                 raw_data_dir_,
                 seq_fname,  # ='raw_seq_train.pkl',
                 sq_seq_fname,  # ='raw_seq_sq_train.pkl',
                 image_feature_dir_,
                 hyper,
                 num_steps=-1,
                 num_epochs=-1,
                 image_processor_=None,
                 name='BatchContextIterator'):
        BatchImageIterator3.__init__(self, df, raw_data_dir_, seq_fname, sq_seq_fname, hyper, num_steps, num_epochs, image_processor_ or VGGProcessor(image_feature_dir_), name)


def restore_state(*paths):
    state = dtc.load(*paths)
    return state['props'], state['df_train_idx'], state['df_validation_idx']


def store_state(props, df_train, df_validation, *paths):
    dtc.dump({'props':props, 'df_train_idx':df_train.index, 'df_validation_idx': df_validation.index if df_validation is not None else None}, *paths)


def split_dataset(df_, batch_size_, logger, args, assert_whole_batch=True, validation_frac=None, validation_size=None):
    if validation_frac is None and validation_size is None:
        raise ValueError('Either validation_frac or validation_size must be specified.')
    elif validation_frac is not None and validation_size is not None:
        raise ValueError('Either validation_frac or validation_size must be specified, but not both.')
    elif validation_frac is not None:
        assert 1.0 > validation_frac >= 0
        validation_size = int(df_.shape[0] * validation_frac)
    num_val_batches = validation_size // batch_size_
    validation_size = num_val_batches * batch_size_
    assert num_val_batches >= 0
    if num_val_batches == 0:
        logger.warn('number of validation batches set to 0.')

    if dtc.exists(args.logdir, 'split_state.pkl'):
        split_props, df_train_idx, df_validation_idx = restore_state(args.logdir, 'split_state.pkl')
        assert split_props['batch_size'] == batch_size_, 'batch_size in HD5 store (%d) is different from that specified (%d)'%(split_props['batch_size'], batch_size_)
        assert split_props['num_val_batches'] == num_val_batches, 'num_val_batches in HD5 store (%d) is different from that specified (%d)'%(split_props['num_val_batches'] , num_val_batches)
        logger.info('split_dataset: loaded df_train and df_validate from pickle store.')
        if df_validation_idx is not None:
            return df_.loc[df_train_idx], df_.loc[df_validation_idx] if df_validation_idx is not None else None
        else:
            return df_
    else:
        logger.warn('split_dataset: generating a new train/validate split')

        # Shuffle the dataframe
        df_ = df_.sample(frac=1)

        # Make overall list of batches
        batch_list = make_batch_list(df_, batch_size_, assert_whole_batch)
        # Since batch_list is already randomized, just take num_batch_list
        # values from it.
        val_batches = batch_list[:num_val_batches]

        def get_bin_counts(batch_list):
            bin_counts = {}
            for batch_idx in batch_list:
                if batch_idx[0] in bin_counts:
                    bin_counts[batch_idx[0]] += 1
                else:
                    bin_counts[batch_idx[0]] = 1
            return bin_counts

        # Separate out the training and validation samples
        df_train = df_
        df_validation = None
        if (num_val_batches > 0):
            val_bin_counts = get_bin_counts(val_batches)
            for bin_len, num_batches in val_bin_counts.iteritems():
                df_val_bin = df_[df_.bin_len == bin_len].iloc[:num_batches*batch_size_]
                df_validation = df_val_bin if df_validation is None else df_validation.append(df_val_bin)

            df_train = df_train.drop(df_validation.index)
            assert df_train.shape[0] + df_validation.shape[0] == df_.shape[0]
        else:
            assert df_train.shape[0] == df_.shape[0]

        store_state({'batch_size': batch_size_, 'num_val_batches': num_val_batches},
                    df_train, df_validation, args.logdir, 'split_state.pkl')
        return df_train, df_validation


def _get_data(hyper, args, raw_data_dir_):
    df_train = pd.read_pickle(os.path.join(raw_data_dir_, 'df_train.pkl'))
    df_test = pd.read_pickle(os.path.join(raw_data_dir_, 'df_test.pkl'))
    ret_props = dlc.Properties({
        'df_train': df_train,
        'df_test': df_test,
        'train_seq_fname': 'raw_seq_train.pkl',
        'train_sq_seq_fname': 'raw_seq_sq_train.pkl',
        'valid_seq_fname': 'raw_seq_train.pkl',
        'valid_sq_seq_fname': 'raw_seq_sq_train.pkl',
        'test_seq_fname': 'raw_seq_test.pkl',
        'test_sq_seq_fname': 'raw_seq_sq_test.pkl'
    })

    if os.path.exists(os.path.join(raw_data_dir_, 'df_valid.pkl')):
        df_valid = pd.read_pickle(os.path.join(raw_data_dir_, 'df_valid.pkl'))
        ret_props.valid_seq_fname = 'raw_seq_valid.pkl'
        ret_props.valid_sq_seq_fname = 'raw_seq_sq_valid.pkl'
        hyper.logger.info('Loaded dataframes from %s df_train.shape=%s, df_valid.shape=%s, df_test.shape=%s' % (
            raw_data_dir_, df_train.shape, df_valid.shape, df_test.shape))
    else:
        hyper.logger.warn("Didn't find df_valid.pkl. Will split df_train into df_train and df_valid.")
        df_train, df_valid = split_dataset(df_train,
                                           hyper.data_reader_B,
                                           hyper.logger,
                                           args,
                                           hyper.assert_whole_batch,
                                           validation_frac=args.valid_frac)
        hyper.logger.info('Split dataframes from %s df_train.shape=%s, df_valid.shape=%s, df_test.shape=%s' % (
            raw_data_dir_, df_train.shape, df_valid.shape, df_test.shape))
    ret_props.df_valid = df_valid

    return ret_props


def create_context_iterators(raw_data_dir_,
                             image_feature_dir_,
                             hyper,
                             args,
                             image_processor_=None):
    # df = pd.read_pickle(os.path.join(raw_data_dir_, 'df_train.pkl'))
    # df_train, df_valid = split_dataset(df,
    #                                    hyper.data_reader_B,
    #                                    hyper.logger,
    #                                    args,
    #                                    hyper.assert_whole_batch,
    #                                    validation_frac=args.valid_frac)
    data_p = _get_data(hyper, args, raw_data_dir_)

    batch_iterator_train = batch_iterator_eval = None
    if not (args.doTest or args.doValidate):
        batch_iterator_train = BatchContextIterator(data_p.df_train,
                                                    raw_data_dir_,
                                                    data_p.train_seq_fname,
                                                    data_p.train_sq_seq_fname,
                                                    image_feature_dir_,
                                                    hyper,
                                                    args.num_steps,
                                                    args.num_epochs,
                                                    image_processor_,
                                                    'TrainingIterator')

    if args.doTest:
        batch_iterator_eval =  BatchContextIterator(data_p.df_test,
                                                    raw_data_dir_,
                                                    data_p.test_seq_fname,
                                                    data_p.test_sq_seq_fname,
                                                    image_feature_dir_,
                                                    hyper,
                                                    num_steps=-1,
                                                    num_epochs=-1,
                                                    image_processor_=image_processor_,
                                                    name='TestIterator')
    elif args.doValidate or args.doTrain:
        batch_iterator_eval = BatchContextIterator(data_p.df_valid,
                                                   raw_data_dir_,
                                                   data_p.valid_seq_fname,
                                                   data_p.valid_sq_seq_fname,
                                                   image_feature_dir_,
                                                   hyper,
                                                   num_steps=-1,
                                                   num_epochs=-1,
                                                   image_processor_=image_processor_,
                                                   name='ValidationIterator')

    return batch_iterator_train, batch_iterator_eval


def create_imagenet_iterators(raw_data_dir_, hyper, args):
    # df = pd.read_pickle(os.path.join(raw_data_dir_, 'df_train.pkl'))
    # df_train, df_valid = split_dataset(df,
    #                                    hyper.data_reader_B,
    #                                    hyper.logger,
    #                                    args,
    #                                    hyper.assert_whole_batch,
    #                                    validation_frac=args.valid_frac)

    data_p = _get_data(hyper, args, raw_data_dir_)
    image_processor = ImagenetProcessor3(hyper, args.image_dir)

    batch_iterator_train = batch_iterator_eval = None
    if not (args.doTest or args.doValidate):
        batch_iterator_train = BatchImageIterator3(data_p.df_train,
                                                   raw_data_dir_,
                                                   data_p.train_seq_fname,
                                                   data_p.train_sq_seq_fname,
                                                   hyper,
                                                   args.num_steps,
                                                   args.num_epochs,
                                                   image_processor,
                                                   'TrainingIterator')

    if args.doTest:
        batch_iterator_eval = BatchImageIterator3(data_p.df_test,
                                                  raw_data_dir_,
                                                  data_p.test_seq_fname,
                                                  data_p.test_sq_seq_fname,
                                                  hyper,
                                                  num_steps=-1,
                                                  num_epochs=-1,
                                                  image_processor_=image_processor,
                                                  name='TestIterator')
    elif args.doValidate or args.doTrain:
        batch_iterator_eval = BatchImageIterator3(data_p.df_valid,
                                                  raw_data_dir_,
                                                  data_p.valid_seq_fname,
                                                  data_p.valid_sq_seq_fname,
                                                  hyper,
                                                  num_steps=-1,
                                                  num_epochs=-1,
                                                  image_processor_=image_processor,
                                                  name='ValidationIterator')

    return batch_iterator_train, batch_iterator_eval


def create_BW_image_iterators(raw_data_dir_, hyper, args):
    # df = pd.read_pickle(os.path.join(raw_data_dir_, 'df_train.pkl'))
    # df_train, df_valid = split_dataset(df,
    #                                    hyper.data_reader_B,
    #                                    hyper.logger,
    #                                    args,
    #                                    hyper.assert_whole_batch,
    #                                    validation_frac=args.valid_frac)

    data_p = _get_data(hyper, args, raw_data_dir_)

    image_processor = ImageProcessor3_BW(hyper, args.image_dir)

    batch_iterator_train = batch_iterator_eval = None
    if not (args.doTest or args.doValidate):
        batch_iterator_train = BatchImageIterator3(data_p.df_train,
                                                   raw_data_dir_,
                                                   data_p.train_seq_fname,
                                                   data_p.train_sq_seq_fname,
                                                   hyper,
                                                   args.num_steps,
                                                   args.num_epochs,
                                                   image_processor,
                                                   'TrainingIterator')

    if args.doTest:
        batch_iterator_eval = BatchImageIterator3(data_p.df_test,
                                                  raw_data_dir_,
                                                  data_p.test_seq_fname,
                                                  data_p.test_sq_seq_fname,
                                                  hyper,
                                                  num_steps=-1,
                                                  num_epochs=-1,
                                                  image_processor_=image_processor,
                                                  name='TestIterator')
    elif args.doValidate or args.doTrain:
        batch_iterator_eval = BatchImageIterator3(data_p.df_valid,
                                                  raw_data_dir_,
                                                  data_p.valid_seq_fname,
                                                  data_p.valid_sq_seq_fname,
                                                  hyper,
                                                  num_steps=-1,
                                                  num_epochs=-1,
                                                  image_processor_=image_processor,
                                                  name='ValidationIterator')

    return batch_iterator_train, batch_iterator_eval
