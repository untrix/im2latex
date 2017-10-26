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
import pandas as pd
import numpy as np
import data_commons as dtc
import dl_commons as dlc
import h5py
import matplotlib.pyplot as plt
from data_reader import ImagenetProcessor
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib as mpl

class VGGnetProcessor(ImagenetProcessor):
    def __init__(self, params, image_dir_):
        ImagenetProcessor.__init__(self, params)

    def get_one(self, image_name_, height_, width_, padded_dim_):
        """Get image-array padded and aligned as it was done to extract vgg16 features using convnet.py."""
        image_data = ImagenetProcessor.get_array(self, image_name_, height_, width_, padded_dim_)
        whitened =  self.whiten(np.asarray([image_data]))
        return whitened[0]

    def get_array(self):
        raise NotImplementedError()

    def whiten(self, image_batch):
        """
        normalize values to lie between 0.0 and 1.0 as required by matplotlib.axes.Axes.imshow.
        We skip the preprocessing required by VGGnet since we are not going to pass this into VGGnet.
        Arguments:
            image_batch: (ndarray) Batch of images.
        """
        assert image_batch.shape[1:] == tuple(self._params.image_shape), 'Got image shape %s instead of %s'%(image_batch.shape[1:], tuple(self._params.image_shape))
        return image_batch / 255.0

def plotImage(image_detail, axes, cmap=None):
    path = image_detail[0]
    image_data = image_detail[1]
    title = os.path.splitext(os.path.basename(path))[0]
    axes.set_title( title )
    axes.set_xlim(-10,1500)
    if cmap is not None:
        axes.imshow(image_data, aspect='equal', extent=None, resample=False, interpolation='bilinear', cmap=cmap)
    else:
        axes.imshow(image_data, aspect='equal', extent=None, resample=False, interpolation='bilinear')
    # print 'plotted image ', path, ' with shape ', image_data.shape
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
        fig = plt.figure(figsize=(15.,3.*len(image_details)))
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
        except:
            self._hyper = dtc.load(self._storedir, 'hyper.pkl')
            self._args = dtc.load(self._storedir, 'args.pkl')
        self._image_dir = self._args['image_dir']
        self._generated_data_dir = gen_datadir = self._args['generated_data_dir']
        self._data_dir = self._args['data_dir']
        self._raw_data_dir = os.path.join(gen_datadir, 'training')

        self._word2id = pd.read_pickle(os.path.join(gen_datadir, 'dict_vocab.pkl'))
        i2w = pd.read_pickle(os.path.join(gen_datadir, 'dict_id2word.pkl'))
        for i in range(-1,-11,-1):
            i2w[i] = '%d'%i
        self._id2word = {}
        ## Append space after all commands beginning with a backslash (except backslash alone)
        for i, w in i2w.items():
            if w[0] == '\\':
              self._id2word[i] = w + " "  
            else:
                self._id2word[i] = w 
        self._id2word[self._word2id['id']['\\']] = '\\'

        ## Image processor to load and preprocess images
        self._image_processor = VGGnetProcessor(self._hyper, self._image_dir)

        ## Train/Test DataFrames
        self._df_train_image = pd.read_pickle(os.path.join(self._raw_data_dir, 'df_train.pkl'))[['image', 'height', 'width']]
        # self._df_train_image = self._df_train.copy()
        self._df_train_image.index = self._df_train_image.image
        # self._df_test = pd.read_pickle(os.path.join(self._raw_data_dir, 'df_test.pkl'))
        self._padded_im_dim = {'height': self._hyper.image_shape[0], 'width': self._hyper.image_shape[1]}

    @property
    def storedir(self):
        return self._storedir
    
    @property
    def w2i(self):
        return self._word2id['id']

    @property
    def i2w(self):
        return self._id2word
    
    @property
    def max_steps(self):
        steps = [int(os.path.basename(f).split('_')[-1].split('.')[0]) for f in os.listdir(self._storedir) if f.endswith('.h5')]
        epoch_steps = [int(os.path.basename(f).split('_')[-1].split('.')[0]) for f in os.listdir(self._storedir) if f.startswith('validation')]
        return sorted(steps)[-1], sorted(epoch_steps)[-1]
        
    @property
    def args(self):
        return self._args
    
    @property
    def hyper(self):
        return self._hyper
    
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
        with h5py.File(os.path.join(self._storedir, '%s_%d.h5'%(graph,step))) as h5:
            return h5[key][...]
    
    def df(self, graph, step, key):
        return pd.DataFrame(self.nd(graph, step, key))
    
    def words(self, graph, step, key):
        df = self.df(graph, step, key)
        return df.applymap(lambda x: self._id2word[x])

    def strs(self, graph, step, key, key2=None, mingle=True):
        df_str = self.words(graph, step, key)
        df_str2 = self.words(graph, step, key2) if (key2 is not None) else None
        
        ## each token's string version - excepting backslash - has a space appended to it,
        ## therefore the string output should be compile if the prediction was syntactically correct
        ar1 = ["".join(row) for row in df_str.itertuples(index=False)]
        if key2 == None:
            return pd.DataFrame(ar1)
        else:
            ar2 = ["".join(row) for row in df_str2.itertuples(index=False)]
            if mingle:
                # d = [(i,e) for i,t in enumerate(zip(ar1, ar2)) for e in t]
                # d = zip(*d)
                # index = d[0]
                # data = {'%s / %s   %s_%d   [%s]'%(key, key2, graph, step, self._storedir): d[1]}
                d = [e for t in zip(ar1, ar2) for e in t]
                data = {'%s / %s   %s_%d   [%s]'%(key, key2, graph, step, self._storedir): d}
                index = ['%d (%s)'%(i,l) for i in range(len(ar1)) for l in ('k1', 'k2')]
            else:
                data = {'%s\t%s_%d\t[%s]'%(key, graph, step, self._storedir): ar1, '%s\t%s_%d\t[%s]'%(key2, graph, step, self._storedir): ar2}
                index = range(len(ar1))
            df = pd.DataFrame(data=data, index=index)
#             df.style.set_caption('%s/%s_%s'%(self._storedir, graph, step))
            return df
    
    def alpha(self, graph, step, sample_num=0):
        nd_words = self.words(graph, step, 'predicted_ids').iloc[sample_num].values # (B,T) --> (T,)
        nd_alpha = self.nd(graph, step, 'alpha')[0][sample_num] #(N,B,H,W,T) --> (H,W,T)
        image_name = self.nd(graph, step, 'image_name')[sample_num] #(B,) --> (,)
        T = len(nd_words)
        df = self._df_train_image
        image_data = self._image_processor.get_one(os.path.join(self._image_dir, image_name), df.height.loc[image_name], 
                                                   df.width.loc[image_name], self._padded_im_dim) #(H,W,C)
        maxpool_factor = 32
        pad_0 = self._hyper.image_shape[0] - nd_alpha.shape[0]*maxpool_factor
        pad_1 = self._hyper.image_shape[1] - nd_alpha.shape[1]*maxpool_factor

        def project_alpha(alpha_t, expand_dims=True):
            pa = np.repeat(alpha_t, repeats=maxpool_factor, axis=0)
            pa = np.repeat(pa, maxpool_factor, axis=1)
            pa = np.pad(pa, ((0,pad_0),(0,pad_1)), mode='constant', constant_values=0.0)
            pa = 1.0 - pa
            if expand_dims:
                pa = np.expand_dims(pa, axis=2) #(H,W,1)
            return pa
        
        def composit_image(image_data, alpha_t):
            return np.concatenate((image_data, project_alpha(alpha_t)), axis=2)

        image_details=[(image_name, image_data), ('alpha_0', project_alpha(nd_alpha[:,:,0], expand_dims=False))]
        for t in xrange(T):
            if nd_words[t] == 'NUL':
                break
            else:
                ## Project alpha onto image
                # pa = np.repeat(nd_alpha[:,:,t], repeats=maxpool_factor, axis=0)
                # pa = np.repeat(pa, maxpool_factor, axis=1)
                # pa = np.pad(pa, ((0,pad_0),(0,pad_1)), mode='constant', constant_values=0 )
                # pa = np.expand_dims(pa, axis=2) #(H,W,1)
                # pa = pa*0.0 + 1
                composit = composit_image(image_data, nd_alpha[:,:,t])
                image_details.append((nd_words[t], composit))

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
        plotImages(image_details, dpi=144, cmap='gray')

    def prune_logs(self, save_epochs=1, dry_run=True):
        """Save the latest save_epochs logs and remove the rest."""
        def get_step(f):
            return int(os.path.basename(f).split('_')[-1].split('.')[0])
        
        epoch_steps = [get_step(f) for f in os.listdir(self._storedir) if f.startswith('validation')]
        epoch_steps = list(set(epoch_steps))
        print 'epoch_steps: %s'%epoch_steps
        if len(epoch_steps) <= save_epochs:
            print('Only %d full epochs were found. Deleting nothing.'%epoch_steps)
            return False
        else:
            epoch_steps.sort(reverse=True)
            max_step = epoch_steps[save_epochs]
            training_files = [f for f in os.listdir(self._storedir) if f.startswith('training')]
            training_steps = set([get_step(f) for f in training_files])
            steps_to_remove = set(filter(lambda s: (s<max_step) and (s not in epoch_steps), training_steps))
            files_to_remove = set([f for f in training_files if (get_step(f) in steps_to_remove)])
            files_to_keep = set([f for f in os.listdir(self._storedir)]) - files_to_remove
            if dry_run:
                print '%d files will be kept\n'%len(files_to_keep), pd.Series(sorted(list(files_to_keep), key=get_step))
                print '%d files will be removed\n'%len(files_to_remove), pd.Series(sorted(list(files_to_remove), key=get_step))
            else:
                for f in files_to_remove:
                    os.remove(os.path.join(self._storedir, f))
                print 'Removed %d files\n'%len(files_to_remove), pd.Series(sorted(list(files_to_remove), key=get_step))

    def prune_snapshots(self, keep=10, dry_run=True):
        """ Keep the latest 'save' snapshots. Delete the rest. """
        def get_step(f):
            return int(os.path.basename(f).split('.')[0].split('snapshot-')[1])
        
        files = [f for f in os.listdir(self._logdir) if f.startswith('snapshot-')]
        steps = list(set([get_step(f) for f in files]))
        if len(steps) <= keep:
            print 'Nothing to delete'
            return
        else:
            steps.sort(reverse=True)
            steps_to_keep = set(steps[:keep])
            steps_to_remove = set(steps) - steps_to_keep
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
    
    def df(self, step, key):
        return pd.DataFrame.df(self.nd(self._graph, step, key))
    
    def words(self, key):
        return self._visualizer.words(self._graph, self._step, key)

    def strs(self, key, key2=None, mingle=True):
        return self._visualizer.strs(self._graph, self._step, key, key2, mingle)

    def alpha(self, sample_num=0):
        return self._visualizer.alpha(self._graph, self._step, sample_num)

class DiffParams(object):
    def __init__(self, dir1, dir2):
        self._dir1 = dir1
        self._dir2 = dir2
        
    def get(self, filename, to_str):
        one = dtc.load(self._dir1, filename)
        two = dtc.load(self._dir2, filename)
        if (to_str):
            one = dlc.to_dict(one)
            two = dlc.to_dict(two)
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
    