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
import time
import logging
# from six.moves import cPickle as pickle
import dill as pickle
import numpy as np
import h5py

dict_id2word = None
i2w_ufunc = None
logger = logging


def setLogLevel(logger_, level):
    logging_levels = (logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG)
    logger_.setLevel(logging_levels[level - 1])


def makeFormatter():
    return logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')


def makeLogger(logging_level=3, name='default', set_global=False):
    global logger
    logger_ = logging.Logger(name)
    ch = logging.StreamHandler()
    ch.setFormatter(makeFormatter())
    logger_.addHandler(ch)
    setLogLevel(logger_, logging_level)
    if set_global:
        logger = logger_
    return logger_


def initialize(training_data_dir, params):
    global i2w_ufunc, dict_id2word
    # if logger is None:
    #     logger = params.logger
    if i2w_ufunc is None:
        data_props = load(training_data_dir, 'data_props.pkl')
        dict_id2word = data_props['id2word']
        K = len(dict_id2word.keys())
        CTCBlankTokenID = params.CTCBlankTokenID
        if (CTCBlankTokenID is not None) and (CTCBlankTokenID >= K):
            dict_id2word[CTCBlankTokenID] = u'<>' ## CTC Blank Token
        dict_id2word[-1] = u'<-1>' ## Catch -1s that beamsearch emits after EOS.
        def i2w(id):
            try:
                return dict_id2word[id]
            except KeyError as e:
                logger.critical('i2w: KeyError: %s', e)
                return '<%d>'%(id,)
        i2w_ufunc = np.frompyfunc(i2w, 1, 1)
    return i2w_ufunc


def seq2str(arr, label, separator=None):
    """
    Converts a matrix of id-sequences - shaped (B,T) - to an array of strings shaped (B,).
    Uses the supplied dict_id2word to map ids to words. The dictionary must map dtype of
    <arr> to string.
    """
    assert i2w_ufunc is not None, "i2w_ufunc is None. Please call initialize first in order to setup i2w_ufunc."
    str_arr = i2w_ufunc(arr) # (B, T)
    if separator is None:
        func1d = lambda vec: label + u" " + u"".join(vec)
    else:
        func1d = lambda vec: label + u" " + unicode(separator).join(vec)
    return [func1d(vec) for vec in str_arr]


def join(*paths):
    return os.path.join(*paths)


def dump(ar, *paths):
    path = join(*paths)
    assert not os.path.exists(path), 'A file already exists at path %s'%path
    with open(path, 'wb') as f:
        pickle.dump(ar, f, pickle.HIGHEST_PROTOCOL)


def load(*paths):
    with open(join(*paths), 'rb') as f:
        return pickle.load(f)


class Storer(object):
    def __init__(self, args, prefix, step):
        self._path = os.path.join(args.storedir, '%s_%d.h5'%(prefix, step))
        self._h5 = h5py.File(self._path, mode="w-", swmr=False)

    def __enter__(self):
        return self

    def __exit__(self, *err):
        self.close()

    def flush(self):
        self._h5.flush()

    def close(self):
        self._h5.close()

    def write(self, key, ar, dtype=None, batch_axis=0, doUnwrap=True):
        """
        WARNING: ar must either be an numpy.ndarray (not numpy scalar) or a python list/tuple of numpy.ndarray.
        Nothing else will work.
        :param key:
        :param ar:
        :param dtype:
        :param batch_axis:
        :param doUnwrap:
        :return:
        """
        if (isinstance(ar, tuple) or isinstance(ar, list)) and doUnwrap:
            return self._write(key, ar, dtype, batch_axis)
        else:
            return self._write(key, [ar], dtype, batch_axis)

    def _write(self, key, np_ar_list, dtype, batch_axis):
        """
        WARNING: np_ar_list must be a python list/tuple of numpy.ndarray. Nothing else will work.

        Stacks the tensors in the list along axis=batch_axis and writes them to disk.
        Dimensions along axis=batch_axis are summed up (since we're stacking along that dimension).
        Other dimensions are padded to the maximum size
        with a dtype-suitable value (np.nan for float, -2 for integer)
        """
        ## Assuming all arrays have same rank, find the max dims
        shapes = [ar.shape for ar in np_ar_list]
        dims = zip(*shapes)
        max_shape = [max(d) for d in dims]
        ## We'll concatenate all arrays along axis=batch_axis
        max_shape[batch_axis] = sum(dims[batch_axis])
        if dtype == np.unicode_:
            dt = h5py.special_dtype(vlen=unicode)
            dataset = self._h5.create_dataset(key, max_shape, dtype=dt)
        else:
            dataset = self._h5.create_dataset(key, max_shape, dtype=dtype, fillvalue=-2 if np.issubdtype(dtype, np.integer) else np.nan)

        def make_slice(row, shape, batch_axis):
            """
            Create a slice to place shape into the receiving dataset starting at rownum along axis=batch_axis,
            and starting at 0 along all other axes
            """
            s = [slice(0,d) for d in shape]
            s[batch_axis] = slice(row, row+shape[batch_axis])
            return tuple(s), row+shape[batch_axis]

        row = 0
        for ar in np_ar_list:
            s, row = make_slice(row, ar.shape, batch_axis)
            # logger.info('row=%d, slice=%s', row, s)
            dataset[s] = ar


def makeLogfileName(logdir, name):
    prefix, ext = os.path.splitext(os.path.basename(name))
    filenames = os.listdir(logdir)
    if not (prefix + ext) in filenames:
        return os.path.join(logdir, prefix + ext)
    else:
        for i in xrange(2,101):
            if '%s_%d%s'%(prefix,i,ext) not in filenames:
                return os.path.join(logdir, '%s_%d%s'%(prefix,i,ext))

    raise Exception('logfile number limit (100) reached.')


def exists(*paths):
    return os.path.exists(os.path.join(*paths))


def makeLogDir(root, dirname):
    dirpath = makeLogfileName(root, dirname)
    os.makedirs(dirpath)
    return dirpath


def makeTBDir(tb_logdir, logdir_tag=None):
    if logdir_tag is None:
        dirpath = os.path.join(tb_logdir, time.strftime('%Y-%m-%d %H-%M-%S %Z'))
    else:
        dirpath = os.path.join(tb_logdir, time.strftime('%Y-%m-%d %H-%M-%S %Z') + ' ' + logdir_tag)

    os.makedirs(dirpath)
    return dirpath


def readlines_to_df(path, colname):
    #   return pd.read_csv(output_file, sep='\t', header=None, names=['formula'], index_col=False, dtype=str, skipinitialspace=True, skip_blank_lines=True)
    rows = []
    n = 0
    with open(path, 'r') as f:
        print 'opened file %s'%path
        for line in f:
            n += 1
            line = line.strip()  # remove \n
            if len(line) > 0:
                rows.append(line.encode('utf-8'))
    print 'processed %d lines resulting in %d rows'%(n, len(rows))
    return pd.DataFrame({colname:rows}, dtype=np.str_)


def readlines_to_sr(path):
    rows = []
    n = 0
    with open(path, 'r') as f:
        print 'opened file %s'%path
        for line in f:
            n += 1
            line = line.strip()  # remove \n
            if len(line) > 0:
                rows.append(line.encode('utf-8'))
    print 'processed %d lines resulting in %d rows'%(n, len(rows))
    return pd.Series(rows, dtype=np.str_)


def sr_to_lines(sr, path):
#   df.to_csv(path, header=False, index=False, columns=['formula'], encoding='utf-8', quoting=csv.QUOTE_NONE, escapechar=None, sep='\t')
    assert sr.dtype == np.str_ or sr.dtype == np.object_
    with open(path, 'w') as f:
        for s in sr:
            assert '\n' not in s
            f.write(s.strip())
            f.write('\n')
