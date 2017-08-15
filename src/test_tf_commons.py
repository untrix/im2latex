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

Created on Sun Jul  9 11:44:46 2017
Tested on python 2.7

@author: Sumeet S Singh
"""
import tensorflow as tf
import tf_commons as tfc
from tf_commons import K

h1 = tf.constant([[[1,2,3],[4,5,6]],
                  [[7,8,9],[10,11,12]],
                  [[13,14,15],[16,17,18]]  ])
l1 = tf.constant([[3, 3],
                  [3, 3],
                  [3, 3]])
h2 = tf.constant([[[1,2,3,0,0,0,0],[4,5,6,0,0,0,0]],
                  [[7,8,100,9,0,0,0],[100,10,100,11,12,0,0]],
                  [[13,14,15,100,100,100,0],[100,16,17,18,0,0,0]]  ])
l2 = tf.constant([[3, 3],
                  [4, 5],
                  [6, 4]])
ed1 = tfc.k_edit_distance(3, 2, h2, l2, h1, l1, 100)
mean1 = tf.reduce_mean(ed1)
h2_2 = tf.constant([[[1,2,3,99,0,0,0],[4,5,6,0,0,0,0]],
                  [[7,8,100,99,0,0,0],[100,10,100,11,12,0,0]],
                  [[13,100,15,100,100,100,0],[100,16,17,18,0,0,0]]  ])
l2_2 = tf.constant([[4, 3],
                  [4, 5],
                  [6, 4]])

ed2 = tfc.k_edit_distance(3, 2, h2_2, l2_2, h1, l1, 100)
sum2 = tf.reduce_sum(ed2)
with tf.Session():
    # print ed1.eval()
    assert mean1.eval() == 0.
    print ed2.eval()
    assert sum2.eval() == 1.
    print K.int_shape(ed1)
    print K.int_shape(h1)
