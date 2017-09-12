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
import logging
import pandas as pd
from six.moves import cPickle as pickle
import dl_commons as dlc
import numpy as np

dict_id2word = None
i2w_ufunc = None

def initialize(data_dir):
    global i2w_ufunc, dict_id2word
    if i2w_ufunc is None:
        dict_id2word = dict_id2word or pd.read_pickle(os.path.join(data_dir, 'dict_id2word.pkl'))
        id2word = lambda id: dict_id2word[id]
        i2w_ufunc = np.frompyfunc(id2word, 1, 1)
    return i2w_ufunc

def seq2str(arr):
    """
    Converts a matrix of id-sequences - shaped (B,T) - to an array of strings shaped (B,).
    Uses the supplied dict_id2word to map ids to words. The dictionary must map dtype of
    <arr> to string.
    """
    assert i2w_ufunc is not None, "i2w_ufunc is None. Please call initialize first in order to setup i2w_ufunc."
    str_arr = i2w_ufunc(arr)
    return np.apply_along_axis(lambda vec: " ".join(vec), axis=1, arr=str_arr)


