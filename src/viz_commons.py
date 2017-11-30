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
import pandas as pd
import numpy as np
import data_commons as dtc
import dl_commons as dlc
import h5py
import matplotlib.pyplot as plt
from data_reader import ImagenetProcessor, ImageProcessor3_BW
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib as mpl
from IPython.display import Math, display

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
    # TODO: Deliver image_shape_unframed here instead of image_shape
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
        print 'CustomConvnetImageProcessor: image_batch_shape = %s, image_shape_unframed = %s'%(image_batch.shape, self._params.image_shape_unframed)
        image_batch /= 255.0
        num_channels = image_batch.shape[3]
        if num_channels == 1: ## Need all three RGB channels so that we can layer on alpha channel to get RGBA
            image_batch = np.repeat(image_batch, 3, axis=3)

        return image_batch

    def get_array(self):
        raise NotImplementedError()

    def whiten(self, image_batch):
        raise NotImplementedError()


def plotImage(image_detail, axes, cmap=None):
    path = image_detail[0]
    image_data = image_detail[1]
    title = os.path.splitext(os.path.basename(path))[0]
    axes.set_title(title)
    # axes.set_xlim(-10,1500)
    # print 'image %s %s'%(title, image_data.shape)
    if cmap is not None:
        axes.imshow(image_data, aspect='equal', extent=None, resample=False, interpolation='bilinear', cmap=cmap)
    else:
        axes.imshow(image_data, aspect='equal', extent=None, resample=False, interpolation='bilinear')
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
        fig = plt.figure(figsize=(10.,2.*len(image_details)))
        grid = ImageGrid(fig, 111, nrows_ncols=(len(image_details),1), axes_pad=(0.1, 0.5), label_mode="L", aspect=False)
        for i in range(len(image_details)):
            plotImage(image_details[i], grid[i], cmap)

    return

class VisualizeDir(object):
    def __init__(self, storedir):
        self._storedir = storedir
        self._logdir = os.path.join(storedir, '..')
        try:
            self._hyper = dtc.load(self._logdir, 'hyper.pkl')
            self._args = dtc.load(self._logdir, 'args.pkl')
            print('Loaded %s and %s'%(dtc.join(self._logdir, 'hyper.pkl'), dtc.join(self._logdir, 'args.pkl')))
        except:
            self._hyper = dtc.load(self._storedir, 'hyper.pkl')
            self._args = dtc.load(self._storedir, 'args.pkl')
            print('Loaded %s and %s' % (dtc.join(self._storedir, 'hyper.pkl'), dtc.join(self._storedir, 'args.pkl')))

        self._image_dir = self._args['image_dir']
        self._data_dir = self._args['data_dir']
        self._raw_data_dir = self._args['raw_data_dir']
        self._SCF = self._hyper.B * self._hyper.num_gpus * 1.0 / (64.0)  # conflation factor

        self._data_props = pd.read_pickle(os.path.join(self._raw_data_dir, 'data_props.pkl'))
        self._word2id = self._data_props['word2id']
        i2w = self._data_props['id2word']

        for i in range(-1,-11,-1):
            i2w[i] = '%d'%i
        self._id2word = {}
        ## Append space after all commands beginning with a backslash (except backslash alone)
        for i, w in i2w.items():
            if w[0] == '\\':
              self._id2word[i] = w + " "  
            else:
                self._id2word[i] = w 
        self._id2word[self._word2id['\\']] = '\\'

        ## Image processor to load and preprocess images
        if self._hyper.build_image_context == 0:
            self._image_processor = VGGnetProcessor(self._hyper, self._image_dir)
        elif self._hyper.build_image_context == 2:
            self._image_processor = CustomConvnetImageProcessor(self._hyper, self._image_dir)
        else:
            raise Exception('No available ImageProcessor for this case (build_image_context=%s). You have not written one yet!'%self._hyper.build_image_context)

        self._maxpool_factor = 32
        self._image_pool_factor = self.PoolFactor(self._maxpool_factor*self._hyper.H0/self._hyper.H, self._maxpool_factor*self._hyper.W0/self._hyper.W)


        ## Train/Test DataFrames
        self._df_train_image = pd.read_pickle(os.path.join(self._raw_data_dir, 'df_train.pkl'))[['image', 'height', 'width']]
        print('Loaded %s %s'%(os.path.join(self._raw_data_dir, 'df_train.pkl'), self._df_train_image.shape) )
        # self._df_train_image = self._df_train.copy()
        self._df_train_image.index = self._df_train_image.image
        # self._df_test = pd.read_pickle(os.path.join(self._raw_data_dir, 'df_test.pkl'))
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

    def get_steps(self):
        steps = [int(os.path.basename(f).split('_')[-1].split('.')[0]) for f in os.listdir(self._storedir) if f.endswith('.h5')]
        epoch_steps = [int(os.path.basename(f).split('_')[-1].split('.')[0]) for f in os.listdir(self._storedir) if f.startswith('validation')]
        steps = sorted(steps)
        epoch_steps = sorted(epoch_steps)
        return steps, epoch_steps

    def get_snapshots(self):
        epoch_steps = [int(os.path.basename(f).split('.')[0].split('snapshot-')[1]) for f in os.listdir(self._logdir) if f.endswith('.index') and f.startswith('snapshot-')]
        epoch_steps = sorted(epoch_steps)
        return epoch_steps

    def view_snapshots(self):
        snapshots = self.get_snapshots()
        print('Num Snapshots: %d'%(len(snapshots),))
        b = self._hyper['data_reader_B']*1.
        print('Snapshots = %s'%[(step, b*step / 64.) for step in snapshots])

    def view_steps(self):
        all_steps, epoch_steps = self.get_steps()
        print('num epoch_steps = %d'%len(epoch_steps))
        b = self._hyper['data_reader_B']*1.
        print('epoch_steps = %s'%[(step, b*step / 64.) for step in epoch_steps])
        print('all_steps = %s'%[(step, b*step / 64.) for step in all_steps])

    def standardize_step(self, step):
        return (step * self._hyper['data_reader_B']*1.) / 64.

    def unstandardize_step(self, step):
        return (step * 64.) / (self._hyper['data_reader_B']*1.)

    def keys(self, graph, step):
        with h5py.File(os.path.join(self._storedir, '%s_%d.h5'%(graph,step))) as h5:
            return h5.keys()

    def nd(self, graph, step, key):
        """
        Args:
            graph: 'training' or 'validation'
            step:  step who's output is to be fetched
            key:   key of object to fetch - e.g. 'predicted_ids'
        """
        with h5py.File(os.path.join(self._storedir, '%s_%d.h5'%(graph, step))) as h5:
            return h5[key][...]
    
    @staticmethod
    def _assertKeyIsID(key):
        assert (key == 'predicted_ids') or (key == 'y')

    def df(self, graph, step, key):
        nd = self.nd(graph, step, key)
        return pd.DataFrame(nd, dtype=nd.dtype)

    def df_ids(self, graph, step, key, sortkey='ed', trim=False):
        """
        Returns
            1) a list of 'B' word-lists optionally trimmed,
            2) optionally a list of 'B' id-lists and
            3) optionally a list of 'B' sortkey values
        key must represent an id-sequences such as 'predicted_ids' and 'y', i.e. _assertKeyIsID(key) must pass.
        """
        self._assertKeyIsID(key)
        df_ids = self.df(graph, step, key)

        ## sort
        df_sortkey = self.df(graph, step, sortkey)
        assert len(df_ids) == len(df_sortkey)
        df_ids[sortkey] = df_sortkey
        assert len(df_ids) == len(df_sortkey)
        df_ids.sort_values(sortkey, ascending=True, inplace=True)
        sr_sortkey = df_ids[sortkey]
        df_ids.drop(sortkey, axis=1, inplace=True)

        ## id2word and trim
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

        ids_words = reduce(accumColTrim if trim else accumCol, df_ids.values, {'ids': [], 'words': []})

        ## return
        return pd.DataFrame({sortkey: sr_sortkey, 'ids': ids_words['ids'], 'words': ids_words['words']}, index=df_ids.index)

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

    def strs(self, graph, step, key, key2=None, mingle=False, trim=True):
        df1 = self.df_ids(graph, step, key, trim=trim)
        df2 = self.df_ids(graph, step, key2, trim=trim) if (key2 is not None) else None
        
        # each token's string version - excepting backslash - has a space appended to it,
        # therefore the string output should be compile if the prediction was syntactically correct
        ar1 = ["".join(row) for row in df1.words.values]
        if key2 is None:
            return pd.DataFrame({'edit_distance':df1.ed, key: ar1, '_id': range(df1.shape[0])}, index=df1.index)
        else:
            ar2 = ["".join(row) for row in df2.words.values]
            if mingle:
                d = [e for t in zip(ar1, ar2) for e in t]
                data = {'%s/%s'%(key, key2): d}
                index = ['%s (%s)'%(i,l) for i in df1.index for l in ('k1', 'k2')]
            else:
                data = {'edit_distance': df1.ed, key: ar1, key2: ar2}
                index = df1.index
            df = pd.DataFrame(data=data, index=index)
            return df
    
    # def strs(self, graph, step, key, key2=None, mingle=False, trim=False):
    #     df_words, sr_ed = self._words(graph, step, key, _get_ed=True)
    #     df_words2 = self.words(graph, step, key2) if (key2 is not None) else None
        
    #     ## each token's string version - excepting backslash - has a space appended to it,
    #     ## therefore the string output should be compile if the prediction was syntactically correct
    #     ar1 = ["".join(row) for row in df_words.itertuples(index=False)]
    #     if key2 == None:
    #         return pd.DataFrame({'edit_distance':sr_ed, key: ar1}, index=df_words.index)
    #     else:
    #         ar2 = ["".join(row) for row in df_words2.itertuples(index=False)]
    #         if mingle:
    #             # d = [(i,e) for i,t in enumerate(zip(ar1, ar2)) for e in t]
    #             # d = zip(*d)
    #             # index = d[0]
    #             # data = {'%s / %s   %s_%d   [%s]'%(key, key2, graph, step, self._storedir): d[1]}
    #             d = [e for t in zip(ar1, ar2) for e in t]
    #             data = {'%s/%s'%(key, key2): d}
    #             index = ['%s (%s)'%(i,l) for i in df_words.index for l in ('k1', 'k2')]
    #         else:
    #             data = {'edit_distance':sr_ed, key: ar1, key2: ar2}
    #             index = df_words.index
    #         df = pd.DataFrame(data=data, index=index)
    #         return df

    PoolFactor = collections.namedtuple("PoolFactor", ('h', 'w'))
    @staticmethod
    def _project_alpha(alpha_t, pool_factor, pad_0, pad_1, expand_dims=True, pad_value=0.0, invert=True):
        pa = np.repeat(alpha_t, repeats=pool_factor.h, axis=0)
        pa = np.repeat(pa, pool_factor.w, axis=1)
        pa = np.pad(pa, ((0,pad_0),(0,pad_1)), mode='constant', constant_values=pad_value)
        if invert:
            pa = 1.0 - pa
        if expand_dims:
            pa = np.expand_dims(pa, axis=2) #(H,W,1)
        return pa

    def _composit_image(self, image_data, alpha_t, pool_factor, pad_0, pad_1, invert_alpha=True):
        return np.concatenate((image_data, self._project_alpha(alpha_t, pool_factor, pad_0, pad_1, invert=invert_alpha)), axis=2)

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

    def alpha(self, graph, step, sample_num=0, invert_alpha=True):
        df_all = self.df_ids(graph, step, 'predicted_ids', trim=False)
        sample_idx = df_all.iloc[sample_num].name ## Top index in sort order
        nd_ids = df_all.loc[sample_idx].ids # (B,T) --> (T,)
        nd_words = df_all.loc[sample_idx].words # (B,T) --> (T,)
        ## Careful with sample_idx
        nd_alpha = self.nd(graph, step, 'alpha')[0][sample_idx]  # (N,B,H,W,T) --> (H,W,T)
        image_name = self.nd(graph, step, 'image_name')[sample_idx]  #(B,) --> (,)
        T = len(nd_words)
        assert nd_alpha.shape[2] >= T, 'nd_alpha.shape == %s, T == %d'%(nd_alpha.shape, T)
        df = self._df_train_image
        image_data = self._image_processor.get_one(image_name,
                                                   df.height.loc[image_name],
                                                   df.width.loc[image_name],
                                                   self._padded_im_dim)  # (H,W,C)
        pool_factor = self._image_pool_factor
        pad_0 = self._hyper.image_shape_unframed[0] - nd_alpha.shape[0]*pool_factor.h
        pad_1 = self._hyper.image_shape_unframed[1] - nd_alpha.shape[1]*pool_factor.w

        predicted_ids = self.strs(graph, step, 'predicted_ids', trim=True).loc[sample_idx].predicted_ids
        y =  self.strs(graph, step, 'y', trim=True).loc[sample_idx].y
        display(Math(predicted_ids))
        print(predicted_ids)
        display(Math(y))
        print(y)
        
        image_details = [(image_name, image_data), ('alpha_0', self._project_alpha(nd_alpha[:,:,0], pool_factor, pad_0, pad_1, expand_dims=False))]
        for t in xrange(T):
            ## Project alpha onto image
            composit = self._composit_image(image_data, nd_alpha[:,:,t], pool_factor, pad_0, pad_1, invert_alpha=invert_alpha)
            image_details.append((nd_words[t], composit))
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
        plotImages(image_details, dpi=200)
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
    
    def df(self, key):
        return self._visualizer.df(self._graph, self._step, key)

    def df_ids(self, key, sortkey='ed', trim=False):
        return self._visualizer.df_ids(self._graph, self._step, key, sortkey, trim)

    def words(self, key):
        return self._visualizer.words(self._graph, self._step, key)

    def strs(self, key, key2=None, mingle=True, trim=False):
        return self._visualizer.strs(self._graph, self._step, key, key2, mingle, trim)

    def alpha(self, sample_num=0, invert_alpha=True):
        return self._visualizer.alpha(self._graph, self._step, sample_num, invert_alpha=invert_alpha)

class DiffParams(object):
    def __init__(self, dir1, dir2):
        self._dir1 = dir1
        self._dir2 = dir2
        
    def get(self, filename, to_str):
        try:
            one = dtc.load(self._dir1, 'store', filename)
        except:
            one = dtc.load(self._dir1, filename)

        try:
            two = dtc.load(self._dir2, 'store', filename)
        except:
            two = dtc.load(self._dir2, filename)

        if (to_str):
            one = dlc.to_picklable_dict(one)
            two = dlc.to_picklable_dict(two)
        return one, two

    def print_dict(self, filename, to_str):
        one, two = self.get(filename, to_str)
        dtc.pprint(dlc.diff_dict(one, two))
    
    def _table(self, filename):
        one, two = self.get(filename, False)
        head, tail = dlc.diff_table(one, two)
        display(pd.DataFrame(head))
        display(pd.DataFrame(tail))
        
    def args(self, to_str=True):
        self._table('args.pkl')        
        
    def hyper(self, to_str=True):
        self._table('hyper.pkl')
    
    def get_args(self):
        return self.get('args.pkl', to_str=True)
    def get_hyper(self):
        return self.get('hyper.pkl', to_str=True)
    