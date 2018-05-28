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
import itertools
import collections
import math
import re
import csv
import pandas as pd
import numpy as np
import data_commons as dtc
import dl_commons as dlc
import tf_commons as tfc
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
from data_reader import ImagenetProcessor, ImageProcessor3_BW
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib as mpl
from IPython.display import Math, display, Pretty
from PIL import Image


class VGGnetProcessor(ImagenetProcessor):
    def __init__(self, params, image_dir_):
        ImagenetProcessor.__init__(self, params)
        self._image_dir = image_dir_

    def get_one(self, image_name_, height_, width_, padded_dim_):
        """Get image-array padded and aligned as it was done to extract vgg16 features using convnet.py."""
        image_data = ImagenetProcessor.get_array(self, os.path.join(self._image_dir, image_name_), height_, width_, padded_dim_)
        normalized = self.normalize(np.asarray([image_data]))
        return normalized[0]

    def normalize(self, image_batch):
        """
        normalize values to lie between 0.0 and 1.0 as required by matplotlib.axes.Axes.imshow.
        We skip the preprocessing required by VGGnet since we are not going to pass this into VGGnet.
        Arguments:
            image_batch: (ndarray) Batch of images.
        """
        assert image_batch.shape[1:] == tuple(self._params.image_shape_unframed), 'Got image shape %s instead of %s'%(image_batch.shape[1:], tuple(self._params.image_shape_unframed))
        return image_batch / 255.0

    def get_array(self):
        raise NotImplementedError()

    def whiten(self, image_batch):
        raise NotImplementedError()


class CustomConvnetImageProcessor(ImageProcessor3_BW):
    def __init__(self, params, image_dir_):
        ImageProcessor3_BW.__init__(self, params, image_dir_)

    def get_one(self, image_name_, height_, width_, padded_dim_):
        """Get image-array padded and aligned as it was done to extract vgg16 features using convnet.py."""
        image_data = ImageProcessor3_BW.get_array(self, image_name_, height_, width_, padded_dim_)
        normalized = self.normalize(np.asarray([image_data]))
        return normalized[0]

    def normalize(self, image_batch):
        """
        normalize values to lie between 0.0 and 1.0 as required by matplotlib.axes.Axes.imshow.
        Arguments:
            image_batch: (ndarray) Batch of images.
        """
        assert image_batch.shape[1:] == tuple(self._params.image_shape_unframed), 'Got image shape %s instead of %s'%(image_batch.shape[1:], tuple(self._params.image_shape_unframed))
        # print 'CustomConvnetImageProcessor: image_batch_shape = %s, image_shape_unframed = %s'%(image_batch.shape, self._params.image_shape_unframed)
        image_batch /= 255.0
        num_channels = image_batch.shape[3]
        if num_channels == 1:  # Need all three RGB channels so that we can layer on alpha channel to get RGBA
            image_batch = np.repeat(image_batch, 3, axis=3)

        return image_batch

    def get_array(self):
        raise NotImplementedError()

    def whiten(self, image_batch):
        raise NotImplementedError()


def plotImage(image_detail, axes, cmap=None, interp=None):
    path = image_detail[0]
    image_data = image_detail[1]
    title = os.path.basename(path)
    # Valid font size are large, medium, smaller, small, x-large, xx-small, larger, None, x-small, xx-large
    axes.set_title(title, fontdict={'fontsize': 'small'})
    axes.set_ylim(image_data.shape[0], 0)
    axes.set_xlim(0, image_data.shape[1])
    # print 'image %s %s'%(title, image_data.shape)

    axes.imshow(image_data, aspect='equal', extent=None, resample=False, interpolation=interp, cmap=cmap)

    return


def plotImages(image_details, fig=None, dpi=None, cmap=None):
    """ 
    image_details should be an array of image path and image_data - [(path1, image_data1), (path2, image_data2) ...]
    where image_data is a int/float numpy array of shape MxN or MxNx3 or MxNx4 as required by axes.imshow. Also, do note
    that all elements of the arrays should have a value between 0.0 and 1.0 as required by axes.imshow.
    """
    try:
        plt.close(fig)
    except:
        pass

    # fig = plt.figure(num=fig, figsize=(15,15*len(image_details)), dpi=my_dpi)
    # grid = ImageGrid(fig, 111, nrows_ncols=(len(image_details),1), axes_pad=(0.1, 0.5), label_mode="L")
    # for i in range(len(image_details)):
    #     plotImage(image_details[i], grid[i])

    orig_dpi = plt.rcParams['figure.dpi']
    with mpl.rc_context(rc={'figure.dpi': dpi or orig_dpi}):
        fig = plt.figure(figsize=(5, 2*len(image_details)))
        grid = ImageGrid(fig, 111, nrows_ncols=(len(image_details),1), axes_pad=(0.1, 0.5), label_mode="L",
                         aspect=True)
        for i in range(len(image_details)):
            plotImage(image_details[i], grid[i], cmap=image_details[i].cmap or cmap, interp=image_details[i].interpolation)

    return


class VisualizeDir(object):
    def __init__(self, storedir, normalized_dataset=True):
        self._storedir = storedir
        self._logdir = os.path.join(storedir, '..')
        try:
            self._hyper = dlc.Properties.load(self._logdir, 'hyper.pkl')
            self._args = dlc.Properties.load(self._logdir, 'args.pkl')
            dtc.logger.info('Loaded %s and %s'%(dtc.join(self._logdir, 'hyper.pkl'), dtc.join(self._logdir, 'args.pkl')))
        except:
            self._hyper = dlc.Properties.load(self._storedir, 'hyper.pkl')
            self._args = dlc.Properties.load(self._storedir, 'args.pkl')
            dtc.logger.info('Loaded %s and %s' % (dtc.join(self._storedir, 'hyper.pkl'), dtc.join(self._storedir, 'args.pkl')))

        self._image_dir = self._args['image_dir']
        self._data_dir = self._args['data_dir']
        self._raw_data_dir = self._args['raw_data_dir']
        self._SCF = self._hyper.B * self._hyper.num_towers * 1.0 / (64.0)  # conflation factor

        self._data_props = pd.read_pickle(os.path.join(self._raw_data_dir, 'data_props.pkl'))
        self._word2id = self._data_props['word2id']
        i2w = self._data_props['id2word']

        for i in range(-1,-11,-1):
            i2w[i] = '%d'%i
        self._id2word = {}
        if normalized_dataset:  # Append space after all tokens, even backslash alone
            for i, w in i2w.items():
                self._id2word[i] = w + " "
        else:  # Append space after all commands beginning with a backslash (except backslash alone)
            for i, w in i2w.items():
                if w[0] == '\\':
                  self._id2word[i] = w + " "
                else:
                    self._id2word[i] = w
            self._id2word[self._word2id['\\']] = '\\'

        # Image processor to load and preprocess images
        if self._hyper.build_image_context == 0:
            self._image_processor = VGGnetProcessor(self._hyper, self._image_dir)
        elif self._hyper.build_image_context == 2:
            self._image_processor = CustomConvnetImageProcessor(self._hyper, self._image_dir)
        else:
            raise Exception('No available ImageProcessor for this case (build_image_context=%s). You have not written one yet!'%self._hyper.build_image_context)

        # For alpha projection onto image
        self._maxpool_factor = np.power(2, tfc.ConvStackParams.get_numPoolLayers(self._hyper['CONVNET']))
        self._image_pool_factor = self.PoolFactor(self._maxpool_factor*self._hyper.H0/self._hyper.H, self._maxpool_factor*self._hyper.W0/self._hyper.W)
        self._alpha_bleed = self._get_alpha_bleed()


        # Train/Test DataFrames
        self.df_train_images = pd.read_pickle(os.path.join(self._raw_data_dir, 'df_train.pkl'))[['image', 'height', 'width']]
        self.df_train_images.index = self.df_train_images.image
        dtc.logger.info('Loaded image dimensions from %s %s'%(os.path.join(self._raw_data_dir, 'df_train.pkl'), self.df_train_images.shape))

        self._df_test_images = pd.read_pickle(os.path.join(self._raw_data_dir, 'df_test.pkl'))[['image', 'height', 'width']]
        self._df_test_images.index = self._df_test_images.image
        dtc.logger.info('Loaded image dimensions from %s %s'%(os.path.join(self._raw_data_dir, 'df_test.pkl'), self._df_test_images.shape))

        try:
            self._df_valid_images = pd.read_pickle(os.path.join(self._raw_data_dir, 'df_valid.pkl'))[['image', 'height', 'width']]
            self._df_valid_images.index = self._df_valid_images.image
            dtc.logger.info('Loaded image dimensions from %s %s' % (os.path.join(self._raw_data_dir, 'df_valid.pkl'), self._df_valid_images.shape))
        except IOError:  # df_valid.pkl is absent implies df_valid data is included within df_train
            self._df_valid_images = self.df_train_images
        # self._padded_im_dim = {'height': self._hyper.image_shape[0], 'width': self._hyper.image_shape[1]}
        self._padded_im_dim = {'height': self._hyper.image_shape_unframed[0], 'width': self._hyper.image_shape_unframed[1]}

    @property
    def storedir(self):
        return self._storedir
    
    @property
    def w2i(self):
        return self._word2id

    @property
    def i2w(self):
        return self._id2word
    
    @property
    def max_steps(self):
        steps, epoch_steps = self.get_steps()
        num_steps = len(steps)
        num_epoch_steps = len(epoch_steps)
        max_step = steps[-1] if num_steps > 0 else None
        max_epoch_step = epoch_steps[-1] if num_epoch_steps > 0 else None
        return max_step, max_epoch_step
        
    @property
    def args(self):
        return self._args
    
    @property
    def hyper(self):
        return self._hyper

    def _get_alpha_bleed(self):
        CONVNET = self._hyper['CONVNET']
        n_h = n_w = 0
        for layer in CONVNET['layers']:
            if tfc.ConvStackParams.isConvLayer(layer):
                k = tfc.ConvLayerParams.get_kernel_half(layer)
                n_h += k[0]
                n_w += k[1]
            # else:
            #     print('%s is not a ConvLayer'%layer)
        return n_h, n_w

    @staticmethod
    def GET_STEPS(storedir):
        steps = set([int(os.path.basename(f).split('_')[-1].split('.')[0]) for f in os.listdir(storedir) if re.match(r'^training_[0-9]+\.h5$', f)])
        epoch_steps = set([int(os.path.basename(f).split('_')[-1].split('.')[0]) for f in os.listdir(storedir) if re.match(r'^(test|validation)_[0-9]+\.h5$', f)])
        steps = sorted(list(steps))
        epoch_steps = sorted(list(epoch_steps))
        return steps, epoch_steps

    def get_steps(self):
        return self.GET_STEPS(self._storedir)

    @staticmethod
    def GET_SNAPSHOTS(logdir):
        epoch_steps = [int(os.path.basename(f).split('.')[0].split('snapshot-')[1]) for f in os.listdir(logdir) if f.endswith('.index') and f.startswith('snapshot-')]
        epoch_steps = sorted(epoch_steps)
        return epoch_steps

    def get_snapshots(self):
        return self.GET_SNAPSHOTS(self._logdir)

    def view_snapshots(self):
        snapshots = self.get_snapshots()
        print('Num Snapshots: %d'%(len(snapshots),))
        b = self._hyper['data_reader_B']*1.
        print('Snapshots = %s'%[(step, b*step / 64.) for step in snapshots])

    def view_steps(self):
        all_steps, epoch_steps = self.get_steps()
        print('num eval_steps = %d'%len(epoch_steps))
        print('eval_steps = %s'%[(step, self.standardize_step(step)) for step in epoch_steps])
        print('all_steps = %s'%[(step, self.standardize_step(step)) for step in all_steps])
        return all_steps, epoch_steps

    def standardize_step(self, step):
        return (step * self._hyper['data_reader_B']) // 64

    def unstandardize_step(self, step):
        # Since standardize rounds down, unstandardize rounds-up
        return int(math.ceil((step * 64.) / (self._hyper['data_reader_B']*1.)))

    def keys(self, graph, step):
        with h5py.File(os.path.join(self._storedir, '%s_%d.h5'%(graph,step)), mode='r') as h5:
            return h5.keys()

    def nd(self, graph, step, key):
        """
        Args:
            graph: 'training', 'validation' or 'test'
            step:  step who's output is to be fetched
            key:   key of object to fetch - e.g. 'predicted_ids'
        """
        assert graph in ['training', 'validation', 'test', 'metrics_validation', 'metrics_test'], 'graph = %s'%graph
        with h5py.File(os.path.join(self._storedir, '%s_%d.h5'%(graph, step)), mode='r') as h5:
            return h5[key][...]
    
    @staticmethod
    def _assertKeyIsID(key):
        assert (key == 'predicted_ids') or (key == 'y')

    def df(self, graph, step, sortkey, *keys):
        """
        Return optionally sorted DataFrame with *keys as columns
        :param graph:
        :param step:
        :param sortkey: Can be None or a string. If non-None, the returned DF will be
            sorted by this column and the column will appear as one of the DF columns.
        :param keys: Keys to be read from h5 files and converted to a DF column each.
        :return: DataFrame
        """
        if sortkey is not None and sortkey not in keys:
            keys = keys + (sortkey,)
        df_dict = {}
        for key in keys:
            nd = self.nd(graph, step, key)
            # assert nd.ndim <= 2, "Sorry, this function can't handle more than 2 dims"
            df_dict[key] = nd.tolist()
        df = pd.DataFrame(df_dict)
        if sortkey is not None:
            df.sort_values(sortkey, ascending=True, inplace=True)
        return df

    def df_ids(self, graph, step, key, sortkey='ed', trim=False):
        """
        Returns
            1) a list of 'B' word-lists optionally trimmed,
            2) optionally a list of 'B' id-lists and
            3) optionally a list of 'B' sortkey values
        key must represent an id-sequences such as 'predicted_ids' and 'y', i.e. _assertKeyIsID(key) must pass.
        """
        self._assertKeyIsID(key)
        df_ids = self.df(graph, step, sortkey, key)

        # id2word and trim
        def accumRow(d, id):
            d['ids'].append(id)
            d['words'].append(self._id2word[id])
            return d
        def reduce_trim_row(row):
            return reduce(accumRow, itertools.takewhile(lambda id: id>0, row), {'ids':[], 'words':[]})
        def reduce_row(row):
            return reduce(accumRow, row, {'ids':[], 'words':[]})
        def accumCol(d, row):
            rows = reduce_row(row)
            d['ids'].append(rows['ids'])
            d['words'].append(rows['words'])
            return d
        def accumColTrim(d, row):
            d2 = reduce_trim_row(row)
            d['ids'].append(d2['ids'])
            d['words'].append(d2['words'])
            return d
        ids_words = reduce(accumColTrim if trim else accumCol, df_ids[key].values, {'ids': [], 'words': []})
        # return
        data = {'ids': ids_words['ids'], 'words': ids_words['words']}
        if sortkey is not None:
            data[sortkey] = df_ids[sortkey]
        return pd.DataFrame(data, index=df_ids.index)

    def _words(self, graph, step, key, _get_ids=False, _get_ed=False):
        # type: (object, object, object, object, object) -> object
        assert key == 'predicted_ids' or key == 'y'
        df_ids = self.df(graph, step, key)
        df_words = df_ids.applymap(lambda x: self._id2word[x])
        df_words = df_words.assign(ed=self.df(graph, step, 'ed'))
        df_words.sort_values('ed', ascending=True, inplace=True)
        sr_ed = df_words.ed
        df_words.drop('ed', axis=1, inplace=True)

        if (not _get_ids) and (not _get_ed):
            return df_words
        else:
            ret_tuple = (df_words,)
            if _get_ids:
                ret_tuple = ret_tuple + (df_ids,)
            if _get_ed:
                ret_tuple = ret_tuple + (sr_ed,)
            return ret_tuple

    def words(self, graph, step, key):
        return self._words(graph, step, key, _get_ids=False, _get_ed=False)

    def strs(self, graph, step, key, key2=None, mingle=False, trim=True, wrap_strs=False, sortkey='ed', keys=[]):
        df1 = self.df_ids(graph, step, key, sortkey=sortkey, trim=trim)
        df2 = self.df_ids(graph, step, key2, sortkey=sortkey, trim=trim) if (key2 is not None) else None
        df3 = self.df(graph, step, sortkey, *keys) if len(keys) > 0 else None

        # each token's string version - excepting backslash - has a space appended to it,
        # therefore the string output should be compile if the prediction was syntactically correct
        if not wrap_strs:
            ar1 = ["".join(row) for row in df1.words.values]
        else:
            ar1 = ['$%s$'%("".join(row)) for row in df1.words.values]
        ar1_len = [len(seq) for seq in df1.ids]  # ar1_len = [len(s) for s in ar1]

        data = {'%s_len' % key: ar1_len, '_iloc': range(df1.shape[0]), key: ar1}

        colnames = ['_iloc']
        if sortkey is not None:
            data[sortkey] = df1[sortkey]
            colnames.append(sortkey)
        index = df1.index
        colnames.extend(['%s_len' % key, key])

        if key2 is not None:
            if not wrap_strs:
                ar2 = ["".join(row) for row in df2.words.values]
            else:
                ar2 = ['$%s$'%("".join(row)) for row in df2.words.values]
            ar2_len = [len(seq) for seq in df2.ids]

            if not mingle:
                data[key2] = ar2
                data['%s_len' % key2] = ar2_len
                colnames.extend(['%s_len' % key2, key2])
            else:
                assert len(keys) == 0, "keys are not allowed when mingle==True"
                d = [e for t in zip(ar1, ar2) for e in t]
                data = {'%s/%s'%(key, key2): d}
                index_tuples = [(i, l) for i in df1.index for l in (key, key2)]
                index = pd.MultiIndex.from_tuples(index_tuples, names=['sample_id', 'key'])

        if not mingle:
            for k in keys:
                data[k] = df3[k]
                colnames.append(k)

        return pd.DataFrame(data=data, index=index)[colnames]

    def get_preds(self, graph, step, rel_dumpdir, clobber=False, dump=False):
        filepath_preds = os.path.join(self._storedir, rel_dumpdir, 'predictions_%s_%s.pkl' % (graph, step))
        filepath_result = os.path.join(self._storedir, rel_dumpdir, 'eval_image_results.txt')
        filepath_data = os.path.join(self._storedir, rel_dumpdir, 'eval_image_data.txt')
        filepath_label = os.path.join(self._storedir, rel_dumpdir, 'eval_image_labels.txt')
        dirpath = os.path.dirname(filepath_preds)
        if dump:
            if not clobber:
                assert not os.path.exists(filepath_preds), '%s already exits' % filepath_preds
                assert not os.path.exists(filepath_result), '%s already exits' % filepath_result
                assert not os.path.exists(filepath_data), '%s already exits' % filepath_data
                assert not os.path.exists(filepath_label), '%s already exits' % filepath_label
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)

        df1 = self.strs(graph, step, 'y', 'predicted_ids', mingle=False, trim=True, wrap_strs=False, sortkey='ed')
        df2 = self.df(graph, step, 'ed', 'image_name')
        formula_names = df2.image_name.str.extract(r'(.+)_basic\.png', expand=False)
        formula_names.name = 'formula_name'
        df = pd.DataFrame({'_iloc': df1['_iloc'],
                           'image_name': df2.image_name,
                           'ed': df1.ed,
                           'target_len': df1.y_len,
                           'target_seq': df1.y,
                           'pred_len': df1.predicted_ids_len,
                           'pred_seq': df1.predicted_ids}, index=df1.index)
        df.index = df2.image_name.str.replace(r'_basic.png$', '.png')
        df.index.name = 'eval_image_name'
        colsort = ['image_name', '_iloc', 'ed', 'target_len', 'target_seq', 'pred_len', 'pred_seq']
        df = df[colsort]
        df_result = df.assign(score_pred=0, score_gold=0)[['target_seq', 'pred_seq', 'score_pred', 'score_gold']]
        df_data = df[['_iloc']]
        sr_label = df.target_seq
        if dump:
            df.to_pickle(filepath_preds)
            # parser.add_argument('--result-path', dest='result_path',
            #                     type=str, required=True,
            #                     help=(
            #                     'Result file containing <img_path> <label_gold> <label_pred> <score_pred> <score_gold> per line. This should be set to the output file of the model.'
            #                     ))
            # parser.add_argument('--data-path', dest='data_path',
            #                     type=str, required=True,
            #                     help=(
            #                     'Input file which contains the samples to be evaluated. The format is <img_path> <label_idx> per line.'
            #                     ))
            # parser.add_argument('--label-path', dest='label_path',
            #                     type=str, required=True,
            #                     help=(
            #                     'Gold label file which contains a formula per line. Note that this does not necessarily need to be tokenized, and for comparing against the gold standard, the original (un-preprocessed) label file shall be used.'
            #                     ))
            df_result.to_csv(filepath_result, header=False, index=True, encoding='utf-8', quoting=csv.QUOTE_NONE,
                             escapechar=None, sep='\t')
            df_data.to_csv(filepath_data, header=False, index=True, encoding='utf-8', quoting=csv.QUOTE_NONE,
                           escapechar=None, sep='\t')
            dtc.sr_to_lines(sr_label, filepath_label)
        return df, df_result, df_data, sr_label

    PoolFactor = collections.namedtuple("PoolFactor", ('h', 'w'))

    def _project_alpha(self, alpha_t, pool_factor, pad_h, pad_w, expand_dims=True, pad_value=0.0, invert=True,
                       gamma_correction=1):
        # pa = np.zeros(alpha_t.shape)
        pa = np.repeat(alpha_t, repeats=pool_factor.h, axis=0)
        pa = np.repeat(pa, pool_factor.w, axis=1)
        pa = np.pad(pa, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=pad_value)

        # project alpha_bleed
        bleed_h = self._alpha_bleed[0]
        bleed_w = self._alpha_bleed[1]
        pad_t = pad_h // 2
        pad_b = pad_h - pad_t
        pad_l = pad_w // 2
        pad_r = pad_w - pad_l
        # print('bleed_h=%d, bleed_w=%d, pad_t=%d, pad_b=%d, pad_l=%d, pad_r=%d'%(bleed_h, bleed_w, pad_t, pad_b, pad_l, pad_r))
        for h in range(alpha_t.shape[0]):
            for w in range(alpha_t.shape[1]):
                alpha = alpha_t[h,w]
                l = w * pool_factor.w + pad_l
                r = (w+1) * pool_factor.w - 1 + pad_r
                t = h * pool_factor.h + pad_t
                b = (h+1) * pool_factor.h - 1 + pad_b
                t_ = max(0, t - bleed_h)
                b_ = min(b + bleed_h, pa.shape[0])
                l_ = max(0, l - bleed_w)
                r_ = min(r + bleed_w, pa.shape[1])
                # pa[t_:(b_+1), l_:l] += alpha / 2.0
                # pa[t_:(b_+1), (r+1):(r_+1)] += alpha / 2.0
                # pa[t_:t, l_:(r_+1)] += alpha / 2.0
                # pa[(b+1):(b_+1), l_:(r_+1)] += alpha / 2.0
                sl = pa[t_:(b_ + 1), l_:l]
                sl[sl < alpha] = alpha
                sl = pa[t_:(b_ + 1), (r + 1):(r_ + 1)]
                sl[sl < alpha] = alpha
                sl = pa[t_:t, l_:(r_ + 1)]
                sl[sl < alpha] = alpha
                sl = pa[(b + 1):(b_ + 1), l_:(r_ + 1)]
                sl[sl < alpha] = alpha
        pa[pa > 1.] = 1.0

        if invert:
            pa = 1.0 - pa

        # gamma correction
        if gamma_correction != 1:
            pa = np.power(pa, gamma_correction)  # gamma correction

        # expand dims
        if expand_dims:
            pa = np.expand_dims(pa, axis=2)  # (H,W,1)
        return pa

    def _blend_image(self, image_data, alpha_t, pool_factor, pad_0, pad_1, invert_alpha=True, gamma_correction=1):
        alpha = self._project_alpha(alpha_t, pool_factor, pad_0, pad_1, invert=invert_alpha,
                                    gamma_correction=gamma_correction, expand_dims=False)
        assert pad_0 == pad_1 == 0
        # alpha = np.tile(alpha, 3)
        # alpha[:, :, 0] = 1.  # Fix Red channel to 1.0
        # alpha[:, :, 1] = 1.  # Fix Green channel to 1.0
        # alpha[:, :, 2] = 1.  # Fix Blue channel to 1.0
        if image_data.ndim >= 3:
            image = np.mean(image_data, axis=-1)  # collapse RGB dimensions into one
        else:
            image = image_data
        # image[image != 1.0] = 0.0  # set all text values to 0. Will render text better when blended with alpha.
        image_alpha = image * alpha
        image[image == 1.0] = image_alpha[image == 1.0]
        return image
        # return np.concatenate((image_data, self._project_alpha(alpha_t, pool_factor, pad_0, pad_1, invert=invert_alpha)), axis=2)

    def test_alpha_viz(self):
        H = self._hyper.H
        W = self._hyper.W
        pool_factor = self._image_pool_factor
        pad_0 = self._hyper.image_shape_unframed[0] - H*pool_factor.h
        pad_1 = self._hyper.image_shape_unframed[1] - W*pool_factor.w
        print('test_alpha_viz: pad_0=%d, pad_1=%d'%(pad_0, pad_1))
        alphas = np.ones((H*W,H,W), dtype=float)*0.5
        image_details = []
        for h in range(H):
            for w in range(W):
                alpha = alphas[w+h*W]
                alpha[h,w] = 1.0
                pa = self._project_alpha(alpha, pool_factor, pad_0, pad_1, expand_dims=False, pad_value=0.25)
                image_details.append(('alpha[%d,%d]=1.0'%(h,w), pa))

        plotImages(image_details, dpi=200)

    ImgDetail = collections.namedtuple('ImgDetail', ('name', 'array', 'cmap', 'interpolation'))
    def alpha(self, graph, step, sample_num=0, invert_alpha=False, words=None, gamma_correction=1,
              cmap='magma', index=None, show_image=True):
        """
        Display Alpha Scan Through the Sequence.
        :param graph:
        :param step:
        :param sample_num:
        :param invert_alpha:
        :param words: Either the max number of steps/words to display or a tuple/array/sequence of steps starting with 0.
        :param gamma_correction:
        :param cmap:
        :param index:
        :return:
        """
        df_all = self.df_ids(graph, step, 'predicted_ids', trim=False)
        if index is not None:
            df_all = df_all.loc[index]
        sample_id = df_all.iloc[sample_num].name  # position of row in nd-array
        nd_ids = df_all.loc[sample_id].ids  # (B,T) --> (T,)
        nd_words = df_all.loc[sample_id].words  # (B,T) --> (T,)
        # Careful with sample_id
        nd_alpha = self.nd(graph, step, 'alpha')[0][sample_id]  # (N,B,H,W,T) --> (H,W,T)
        image_name = self.nd(graph, step, 'image_name')[sample_id]  # (B,) --> (,)
        T = len(nd_words)
        Ts = None
        if words is not None:
            if dlc.issequence(words):
                T = max(words)+1
                assert len(nd_words) >= T, 'word %d is out of range'%(T-1,)
                Ts = words
            else:
                T = min(T, words)
        if Ts is None:
            Ts = xrange(T)

        assert nd_alpha.shape[2] >= T, 'nd_alpha.shape == %s, T == %d'%(nd_alpha.shape, T)
        df = self.df_train_images if graph == 'training' else self._df_valid_images if graph == 'validation' else self._df_test_images
        image_data = self._image_processor.get_one(image_name,
                                                   df.height.loc[image_name],
                                                   df.width.loc[image_name],
                                                   self._padded_im_dim)  # (H,W,C)
        pool_factor = self._image_pool_factor
        pad_0 = self._hyper.image_shape_unframed[0] - nd_alpha.shape[0]*pool_factor.h
        pad_1 = self._hyper.image_shape_unframed[1] - nd_alpha.shape[1]*pool_factor.w

        predicted_ids = self.strs(graph, step, 'predicted_ids', trim=True).loc[sample_id].predicted_ids
        y = self.strs(graph, step, 'y', trim=True).loc[sample_id].y


        dtc.logger.info('PREDICTED SEQUENCE\n')
        # display(Math(predicted_ids))
        dtc.logger.info(predicted_ids)
        dtc.logger.info('TARGET SEQUENCE\n')
        # display(Math(y))
        dtc.logger.info(y)

        image_details =[self.ImgDetail(image_name, image_data, 'gist_gray', 'bilinear'),
                        # ('alpha_0', self._project_alpha(nd_alpha[:,:,0], pool_factor, pad_0, pad_1,
                        #                                 invert=invert_alpha,
                        #                                 expand_dims=False,
                        #                                 gamma_correction=gamma_correction))
                        ] if show_image else []
        # image_details = [(image_name, image_data),]
        for t in Ts:
            # Blend alpha and image
            composit = self._blend_image(image_data, nd_alpha[:, :, t], pool_factor, pad_0, pad_1,
                                         invert_alpha=invert_alpha, gamma_correction=gamma_correction)
            image_details.append(self.ImgDetail(nd_words[t], composit, cmap, 'bilinear'))
            if nd_ids[t] == 0:
                break

        ## Colormap possible values are: Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, 
        ## CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r,
        # Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, 
        # PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, 
        # Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Vega10, Vega10_r, Vega20, Vega20_r, Vega20b, Vega20b_r, 
        # Vega20c, Vega20c_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r,
        # autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cool, cool_r, coolwarm, coolwarm_r, copper, 
        # copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, 
        # gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, 
        # gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, inferno, inferno_r, jet, jet_r, magma, magma_r, nipy_spectral, 
        # nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, seismic, seismic_r, 
        # spectral, spectral_r, spring, spring_r, summer, summer_r, terrain, terrain_r, viridis, viridis_r, winter, winter_r
        # plotImages(image_details, dpi=72, cmap='gray')
        plotImages(image_details, dpi=200, cmap=cmap)
        return nd_alpha

    def prune_logs(self, save_epochs=1, dry_run=True):
        """Save the latest save_epochs logs and remove the rest."""

        def get_step(f):
            return int(os.path.basename(f).split('_')[-1].split('.')[0])

        def get_sort_order(f):
            if f.endswith('.h5'):
                return get_step(f)
            else:
                return 0

        epoch_steps = [get_step(f) for f in os.listdir(self._storedir) if f.startswith('validation')]
        epoch_steps = sorted(list(set(epoch_steps)))
        print 'epoch_steps: %s'%epoch_steps
        if len(epoch_steps) <= save_epochs:
            print('Only %d full epochs were found. Deleting nothing.'%len(epoch_steps))
            return False
        else:
            epoch_steps.sort(reverse=True)
            max_step = epoch_steps[save_epochs]
            training_files = [f for f in os.listdir(self._storedir) if (f.startswith('training') and f.endswith('.h5'))]
            training_steps = set([get_step(f) for f in training_files])
            steps_to_remove = set(filter(lambda s: (s<max_step) and (s not in epoch_steps), training_steps))
            files_to_remove = set([f for f in training_files if (get_step(f) in steps_to_remove)])
            files_to_keep = set([f for f in os.listdir(self._storedir)]) - files_to_remove
            if dry_run:
                print '%d files will be kept\n'%len(files_to_keep), pd.Series(sorted(list(files_to_keep), key=get_sort_order))
                print '%d files will be removed\n'%len(files_to_remove), pd.Series(sorted(list(files_to_remove), key=get_sort_order))
            else:
                for f in files_to_remove:
                    os.remove(os.path.join(self._storedir, f))
                print 'Removed %d files\n'%len(files_to_remove), pd.Series(sorted(list(files_to_remove), key=get_step))

    def prune_snapshots(self, keep_first=None, keep_last=None, dry_run=True):
        """ Keep the latest 'save' snapshots. Delete the rest. """
        if keep_first is None:
            keep_first = 0

        def get_step(f):
            return int(os.path.basename(f).split('.')[0].split('snapshot-')[1])
        
        files = [f for f in os.listdir(self._logdir) if f.startswith('snapshot-')]
        steps = list(set([get_step(f) for f in files]))
        if keep_last is not None:
            steps_to_keep = set(filter(lambda step: (keep_first <= step <= keep_last), steps))
        else:
            steps_to_keep = set(filter(lambda step: (keep_first <= step), steps))

        steps_to_remove = set(steps) - steps_to_keep
        if len(steps_to_remove) <= 0:
            print 'Nothing to Delete'
            return
        print 'steps to keep: ', sorted(list(steps_to_keep))
        print 'steps to remove: ', sorted(list(steps_to_remove))
        files_to_remove = [f for f in files if (get_step(f) not in steps_to_keep) ]
        files_to_remove = sorted(files_to_remove, key=get_step)

        if dry_run:
            print '%d files will be removed\n'%len(files_to_remove), pd.Series(files_to_remove)
        else:
            for f in files_to_remove:
                os.remove(os.path.join(self._logdir, f))
            print '%d files removed\n'%len(files_to_remove), pd.Series(files_to_remove)


class VisualizeStep():
    def __init__(self, visualizer, graph, step):
        self._step = step
        self._visualizer = visualizer
        self._graph = graph
        
    def keys(self):
        return self._visualizer.keys(self._graph, self._step)
    
    def nd(self, key):
        return self._visualizer.nd(self._graph, self._step, key)
    
    def df(self, *keys):
        return self._visualizer.df(self._graph, self._step, *keys)

    def df_ids(self, key, sortkey='ed', trim=False):
        return self._visualizer.df_ids(self._graph, self._step, key, sortkey, trim)

    def words(self, key):
        return self._visualizer.words(self._graph, self._step, key)

    def strs(self, key, key2=None, mingle=False, trim=False, wrap_strs=False, sortkey='ed', keys=[]):
        return self._visualizer.strs(self._graph, self._step, key, key2, mingle=mingle, trim=trim, wrap_strs=wrap_strs, sortkey=sortkey, keys=keys)

    def get_preds(self, rel_dumpdir='eval_images', clobber=False, dump=True):
        return self._visualizer.get_preds(self._graph, self._step, rel_dumpdir, clobber=clobber, dump=dump)

    def alpha(self, sample_num=0, invert_alpha=False, words=None, gamma_correction=1, cmap='magma', index=None,
              show_image=True):
        return self._visualizer.alpha(self._graph, self._step, sample_num, invert_alpha=invert_alpha,
                                      words=words, gamma_correction=gamma_correction,
                                      cmap=cmap, index=index, show_image=show_image)


class DiffParams(object):
    def __init__(self, dir1, dir2):
        self._dir1 = dir1
        self._dir2 = dir2
        
    def get(self, filename, to_str):
        try:
            f1 = os.path.join(self._dir1, 'store', filename)
            one = dlc.Properties.load(f1)
        except:
            f1 = os.path.join(self._dir1, filename)
            one = dlc.Properties.load(f1)
        print('Loaded %s' % f1)

        try:
            f2 = os.path.join(self._dir2, 'store', filename)
            two = dlc.Properties.load(f2)
        except:
            f2 = os.path.join(self._dir2, filename)
            two = dlc.Properties.load(f2)
        print('Loaded %s' % f2)

        if (to_str):
            one = dlc.to_picklable_dict(one)
            two = dlc.to_picklable_dict(two)
        return one, two, f1, f2

    def print_dict(self, filename, to_str):
        one, two = self.get(filename, to_str)[0:2]
        dtc.pprint(dlc.diff_dict(one, two))
    
    def _table(self, filename, show=True, filter_head=None, filter_tail=None):
        one, two, f1, f2 = self.get(filename, False)
        head, tail = dlc.diff_table(one, two)

        tail = pd.DataFrame(tail, columns=[f1, f2])
        head = pd.DataFrame(head, columns=[f1, f2])

        if filter_head is not None:
            head = head[head[0].str.contains(filter_head)]
        if filter_tail is not None:
            tail = tail[tail[0].str.contains(filter_tail)]

        if show:
            display(head)
            display(tail)
        return head, tail
        
    def args(self, show=True, filter_head=None, filter_tail=None):
        return self._table('args.pkl', show=show, filter_head=filter_head, filter_tail=filter_tail)
        
    def hyper(self, show=True, filter_head=None, filter_tail=None):
        return self._table('hyper.pkl', show=show, filter_head=filter_head, filter_tail=filter_tail)
    
    def get_args(self):
        return self.get('args.pkl', to_str=True)[0:2]

    def get_hyper(self):
        return self.get('hyper.pkl', to_str=True)[0:2]


class Table(object):
    """
    Represents a table. It has one index column and other non-index columns. The table is comprised of
    rows. Empty fields in a row are set to value None. Insert fields using the append method. After
    all fields have been inserted, call freeze. Thereafter you can get a DataFrame object by
    via the df property. Calling df before freeze will return None. Calling freeze more than
    once will beget an exception.
    """
    def __init__(self):
        self._raw_data = dlc.Properties()
        # self._col_names = set()
        self._df = None

    @property
    def df(self):
        return self._df

    def append_fields(self, index, d, index_values=None):
        """
        Insert a row-fragment 'd' into a row indexed by index. If the field was already inserted before, then
        an exception is thrown. indices are additional fields that are also
        inserted into the same row. However, they are not protected agains duplicate insertion because there can
        be multiple index columns in a row which will be repeated with each row-fragment that is inserted (e.g.
        both wall_time and step are index columns, but only one can be passed in as index. The other one must
        be passed into index_values
        :return: Nothing
        """
        assert self._df is None, "You can't insert more data after calling freeze."

        if index not in self._raw_data:
            self._raw_data[index] = {}
        d2 = self._raw_data[index]
        deduped_index = None

        if index_values is not None:
            for k,v in index_values.iteritems():
                d2[k] = v

        for k, v in d.iteritems():
            if k in d2:
                print('WARNING: Duplicate values for tag %s at step %s\n(%s,\n%s)\nOverwriting values' % (k, index, d2[k], v))

            d2[k] = v
            # self._col_names.add(k)

    def freeze(self):
        """
        Create a pandas DataFrame from data collected so far. The object gets frozen thereafter and no new data
        can be inserted.
        :return: Dataframe object that can also be accessed via. the df property.
        """
        assert self._df is None, "You can't call freeze more than once. Obtain DataFrame using the df property."
        self._df = pd.DataFrame.from_dict(self._raw_data, orient='index')
        self._raw_data.freeze()
        return self._df


class TBSummaryReader(object):
    def __init__(self, dirpath, data_reader_B):
        self._dir = dirpath
        self._data_reader_B = data_reader_B
        self._event_files = sorted([os.path.join(self._dir,f) for f in os.listdir(dirpath) if f.startswith('events.out.tfevents')])
        # self._all_tags = self._get_all_summary_tags()
        self._table = Table()

    def _get_all_summary_tags(self):
        """
        Scan the first 20 summary events of all event files and collect the tag names from there. These are deemed
        as the only tag names present in all events.
        :return: A set of tag names.
        """
        all_tags = set()
        for path in self._event_files:
            n = 0
            for e in tf.train.summary_iterator(path):
                if e.HasField('summary'):  # if e.WhichOneof('what') == 'summary':
                    n += 1
                    if n > 20:
                        break
                    else:
                        all_tags |= {v for v in e.summary.value}
        return all_tags

    def _unstandardize_step(self, step):
        return int(math.ceil((step * 64.) / (self._data_reader_B * 1.)))

    def read(self, rexp_):
        """
        Scans all event files in the specified directory and reads tags matching the given regular expression
        and returns a dataframe of all the matched tag values against standardized as well as unstandardized steps.
        :param rexp_: Regular expression string or object to match the event.summary.tag against. Each matched tag that
            has a scalar value type will be represented as a column in the returned dataframe. Non-scalar tags will
            be ignored.
        :return: Returns a dataframe with standardized_step as the index, unstandardized_step as a column and each
            discovered tag as a distinct column as well. The tag columns are populated with the corresponding
            'simple_value' (i.e. a scalar value). Non-scalar tags are ignored.
        """
        rexp = re.compile(rexp_)
        for f in self._event_files:
            self._read_tags(rexp, f, self._table)
        return self._table.freeze()

    def _read_tags(self, rexp, path, table):
        """
        Internal helper function for read_tags. Appends rows of data to table.
        :param rexp: See read_tags
        :param path: Path of event file to load
        :param table: (Table) a table to append rows to
        :return: Nothing
        """
        print('processing file %s'%path)
        try:
            for e in tf.train.summary_iterator(path):
                # w = e.WhichOneof('what')
                if e.HasField('summary'):
                    s = e.summary
                    row = dlc.Properties()
                    row_has_value = False
                    for v in e.summary.value:
                        if v.HasField('simple_value') and rexp.search(v.tag):
                            row[v.tag] = v.simple_value
                            row_has_value = True
                    if row_has_value:
                        table.append_fields(e.step,
                                            row,
                                            {'u_step': self._unstandardize_step(e.step),
                                             'wall_time': e.wall_time,
                                             })
        except tf.errors.DataLossError as e:
            print('WARNING: %s\n'%e)


class EvalRuns(dlc.Properties):
    def __init__(self, logdirs, hyper_names=[], arg_names=[], metric_names=[]):
        dlc.Properties.__init__(self)
        self._logdirs = logdirs
        # self._param_names = hyper_names
        # self._metric_names = metric_names
        # self._arg_names = arg_names
        self.df = None
        self.hypers = None
        self.args = None
        self.metrics = None
        self._make_df(hyper_names, arg_names, metric_names)
        self.freeze()

    def _make_df(self, hyper_names, arg_names, metric_names):
        hypers = [self.load_hyper(logdir) for logdir in self._logdirs]
        self.hypers = {}
        self.hypers.update(zip(self._logdirs, hypers))

        args = [self.load_args(logdir) for logdir in self._logdirs]
        self.args = {}
        self.args.update(zip(self._logdirs, args))

        metrics_n_cols = [self.get_metrics(logdir, hypers[i], metric_names) for (i, logdir) in enumerate(self._logdirs)]
        metric_cols = zip(*metrics_n_cols)[1]
        metrics = zip(*metrics_n_cols)[0]
        self.metrics = {}
        self.metrics.update(zip(self._logdirs, metrics))

        data_d = {}
        for i, logdir in enumerate(self._logdirs):
            data_d[logdir] = row = {}
            for hyper_name in hyper_names:
                row[hyper_name] = hypers[i][hyper_name] if hyper_name in hypers[i] else 'undefined'
            for arg_name in arg_names:
                row[arg_name] = args[i][arg_name] if arg_name in args[i] else 'undefined'
            for tag in metrics[i].keys():
                row[tag] = metrics[i][tag]
        df = pd.DataFrame.from_dict(data_d, orient='index')
        # Sort DF columns as specified in arguments
        self.df = df[hyper_names + arg_names + self.merge_metric_cols(metric_names, metric_cols)]

    @staticmethod
    def merge_metric_cols(exps, cols_seq):
        """
        Merge the column-names reported by each row returned by get_metrics - using sorting order based on the original
        metric_names supplied.
        :param exps: Original metric_names supplied to EvalRuns.__init__
        :type exps: list/sequence
        :param cols_seq: sequence of sequence of column names returned by get_metrics. There are as many sequences
            as the number of rows in the output.
        :type cols_seq: sequence of sequence
        :return: sequence of column names
        """
        merged_colnames = []  # column names from reg-expressions in the right order
        cols = []  # column names from data in the right order
        for t in cols_seq:
            for c in t:
                if c not in cols:
                    cols = cols + [c]
        # cols = reduce(lambda l, t: l+[c for c in t if c not in l], cols_seq, [])
        # print('cols = %s'%cols)
        for exp in exps:
            rexp = exp.split('%')[-1]
            # for cols in cols_seq:
            for col in list(cols):
                if re.search(rexp, col) is not None:
                    if col not in merged_colnames:
                        merged_colnames.append(col)
                    cols.remove(col)

        # for cols in cols_seq:
        assert len(cols) == 0, 'cols = %s,\nexps=%s, \nmerged_colnames=%s'%(cols, exps, merged_colnames)

        return list(merged_colnames)

    def load_hyper(self, logdir):
        return self.load_props(logdir, 'hyper.pkl')

    def load_args(self, logdir):
        return self.load_props(logdir, 'args.pkl')

    def load_props(self, logdir, filename):
        """
        Returns unpickled Properties object from storage.
        :param logdir: logdir of the run
        :param filename: Name of pickled Properties file - hyper.pkl or args.pkl
        :return: Unpickled Properties object. Treat as a dict since all nested Properties are usually converted
            into dict during the pickling process.
        """
        try:
            one = dlc.Properties.load(logdir, 'store', filename)
            print('Loaded %s'%os.path.join(logdir, 'store', filename))
        except:
            one = dlc.Properties.load(logdir, filename)
            print('Loaded %s' % os.path.join(logdir, filename))
        return one

    def get_metrics(self, logdir, hyper, metric_names):
        """
        metric_names is a list of metric_name.
        metric_name format: [<operation>%]<tb_summary_tag_name>
        <operation>: one of (min, max, num)
        :return: dict keyed off of metric_name
        """
        def insert_vals(row, df, tag_exp, op, selected_id=None):
            all_tags = set(df.columns)
            colsort = []
            tags = filter(lambda tag: bool(re.search(tag_exp, tag)), all_tags)
            split_op = op.split('_')
            if split_op[0] == 'select':
                assert len(tags) == 1, 'tag_exp %s_%s maps to %d tags. Must map to exactly one.'%(op, tag_exp, len(tags))
                val, selected_id = get_val(df, tags[0], split_op[1], selected_id)
                colname = '%s%%%s'%(op, tags[0])
                assert colname not in row, 'Multiple metric tag expressions map into colname %s. Fix your tag expressions.' % colname
                row[colname] = val
                colsort.append(colname)
            else:
                for tag in tags:
                    colname = '%s%%%s' % (op, tag)
                    assert colname not in row, 'Multiple metric tag expressions map into colname %s. Fix your tag expressions.' % colname
                    row[colname] = get_val(df, tag, op, selected_id)[0]
                    colsort.append(colname)
            return colsort, selected_id

        def get_val(df, tag, op_, selected_id=None):
            id = val = None
            split_op = op_.split('@')
            op = split_op[0]

            if op == 'min':
                sr = df[df[tag] == df[tag].min()][tag]
                val = sr.values[0]
                id = sr.index[0]
            elif op == 'max':
                sr = df[df[tag] == df[tag].max()][tag]
                val = sr.values[0]
                id = sr.index[0]
            elif op == 'mean':
                val = df[tag].mean()
            elif op == 'num':
                val = df[tag].count()
            elif op == 'selected':
                val = df[tag].loc[selected_id]
                id = selected_id
            elif op == '':
                val = df[tag].iloc[0]

            # Format the value
            val = '%.4f'%val if isinstance(val, float) else val
            # if id is not None:
            if ((len(split_op) > 1) and (split_op[1] == 'id')) or (op == 'selected'):
                val = '%s @ %s'%(val, id)

            return val, id

        reader = TBSummaryReader(logdir, hyper['data_reader_B'])
        op_tags = [name.split('%') for name in metric_names]
        tag_exps = [o_t[-1] for o_t in op_tags]
        rexp = '|'.join(tag_exps)
        df = reader.read(rexp)
        ops = [(o_t[0] if len(o_t)>1 else '') for o_t in op_tags]
        row = {}
        selected_id = None
        colsort = []
        for i, metric_name in enumerate(metric_names):
            csort, selected_id = insert_vals(row, df, tag_exps[i], ops[i], selected_id)
            colsort.extend(csort)
        return row, colsort


def trim_image(np_ar):
    """
    Trims empty rows and columns of an image - i.e. those that have all pixels == 255
    :param np_ar: numpy.ndarray of a image of shape (H,W). Each value should have value between 0 and 255.
    :returns: numpy array of the trimmed image, shape (H', W') where H' <= H and W' <= W
    """
    rows = [(row == 255).all() for row in np_ar]
    cols = [(row == 255).all() for row in np_ar.transpose()]
    top = len([x for x in itertools.takewhile(lambda x: x, rows)])
    bottom = len(rows) - len([x for x in itertools.takewhile(lambda x: x, rows[::-1])])
    left = len([x for x in itertools.takewhile(lambda x: x, cols)])
    right = len(cols) - len([x for x in itertools.takewhile(lambda x: x, cols[::-1])])
    if (top != 0 or left != 0 or bottom != len(rows) or right != len(cols)):
        print('Trimmed image from shape (%d, %d) to (%d, %d)' % (len(rows), len(cols), bottom-top, right-left))
    return np_ar[top:bottom, left:right]


def compare_images(gold_dir, pred_dir, dump=False, clobber=False):
    """
    Finds out how many images in gold_dir exactly match their corresponding image in pred_dir.
    Images in the two directories should have the same name.
    :return:
    """
    unmatched_files = []
    missing_files = []
    gold_files = os.listdir(gold_dir)
    # n = 100
    for i, fname in enumerate(gold_files):
        # if i>=n:
        #     break
        if not os.path.exists(os.path.join(pred_dir, fname)):
            missing_files.append(fname)
            unmatched_files.append(fname)
            continue
        img_gold = trim_image(np.asarray(Image.open(os.path.join(gold_dir, fname)).convert('L')))
        img_gold = np.asarray(img_gold < 255, dtype=np.uint)
        img_pred = trim_image(np.asarray(Image.open(os.path.join(pred_dir, fname)).convert('L')))
        img_pred = np.asarray(img_pred < 255, dtype=np.uint)
        if (img_gold.shape != img_pred.shape) or ((img_gold != img_pred).sum() != 0):
            unmatched_files.append(fname)

    if dump:
        with open(os.path.join(os.path.dirname(pred_dir), 'unmatched_files2.txt'), 'w') as f:
            for fname in unmatched_files:
                f.write('%s\n' % fname)
        with open(os.path.join(os.path.dirname(pred_dir), 'missing_files2.txt'), 'w') as f:
            for fname in missing_files:
                f.write('%s\n' % fname)
    total = len(gold_files)  # min(n, len(gold_files))
    matched = total - len(unmatched_files)
    print('%d (%.2f%%) out of %d images matched binary pixel by binary pixel' % (matched, (matched*100.0)/(total*1.0), total))
    return total, matched, unmatched_files, missing_files
