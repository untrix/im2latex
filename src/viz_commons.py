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

class VisualizeDir(object):
    def __init__(self, storedir, gen_datadir='../data/generated2'):
        self._storedir = storedir
        self._logdir = os.path.join(storedir, '..')
        try:
            self._hyper = dtc.load(self._logdir, 'hyper.pkl')
            self._args = dtc.load(self._logdir, 'args.pkl')
        except:
            self._hyper = dtc.load(self._storedir, 'hyper.pkl')
            self._args = dtc.load(self._storedir, 'args.pkl')

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
        steps = [int(os.path.basename(f).split('_')[-1].split('.')[0]) for f in os.listdir(self._storedir)]
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

    def np(self, graph, step, key):
        """
        Args:
            graph: 'training' or 'validation'
            step:  step who's output is to be fetched
            key:   key of object to fetch - e.g. 'predicted_ids'
        """
        with h5py.File(os.path.join(self._storedir, '%s_%d.h5'%(graph,step))) as h5:
            return h5[key][...]
    
    def df(self, graph, step, key):
        return pd.DataFrame(self.np(graph, step, key))
    
    def words(self, graph, step, key, key2=None):
        df = self.df(graph, step, key)
        df2 = self.df(graph, step, key2) if (key2 is not None) else None
        
        if key2 is None:
            return df.applymap(lambda x: self._id2word[x])
        else:
            return pd.DataFrame({'%s'%key: df.applymap(lambda x: self._id2word[x]), '%s'%key2: df2.applymap(lambda x: self._id2word[x])})

    def strs(self, graph, step, key, key2=None, mingle=True):
        df_str = self.words(graph, step, key)
        df_str2 = self.words(graph, step, key2) if (key2 is not None) else None
        
        ## each token's string version - excepting backslash - has a space appended to it,
        ## therefore the string output should be compile if the prediction was syntactically correct
        if key2 == None:
            return pd.DataFrame(["".join(row) for row in df_str.itertuples(index=False)])
        else:
            if mingle:
                ar1 = ["".join(row) for row in df_str.itertuples(index=False)]
                ar2 = ["".join(row) for row in df_str2.itertuples(index=False)]
                data = {'%s_%d %s / %s\t\t(%s)'%(graph, step, key, key2, self._storedir): [e for t in zip(ar1, ar2) for e in t]}
            else:
                data = {'%s_%d.%s\t\t(%s)'%(graph, step, key, self._storedir): ["".join(row) for row in df_str.itertuples(index=False)], '%s_%d.%s\t\t(%s)'%(graph, step, key2, self._storedir): ["".join(row) for row in df_str2.itertuples(index=False)]}

            df = pd.DataFrame(data)
#             df.style.set_caption('%s/%s_%s'%(self._storedir, graph, step))
            return df
        
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
    def __init__(self, visualizer, step):
        self._step = step
        self._visualizer = visualizer
        
    def keys(self, graph):
        return self._visualizer.keys(graph, self._step)
    
    def np(self, graph, key):
        return self._visualizer.np(graph, self._step, key)
    
    def df(self, graph, step, key):
        return pd.DataFrame.df(self.np(graph, step, key))
    
    def words(self, graph, key, key2=None):
        return self._visualizer.words(graph, self._step, key, key2)

    def strs(self, graph, key, key2=None, mingle=True):
        return self._visualizer.strs(graph, self._step, key, key2, mingle)

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
    