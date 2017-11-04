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
import numpy as np

def flatten(h,l):
    B, k, T = K.int_shape(h)
    return tf.reshape(h, (B*k, -1)), tf.reshape(l, (B*k,))

with tf.device('/cpu:*'):
    ############### Tensors with ED == 0
    h1 = tf.constant([[[1,2,3],[4,5,6]],
                    [[7,8,9],[10,11,12]],
                    [[13,14,15],[16,17,18]]  ])
    l1 = tf.constant([[3, 3],
                    [3, 3],
                    [3, 3]])
    print 'Shapes: h1:%s, l1:%s'%(K.int_shape(h1), K.int_shape(l1))
    h2 = tf.constant([[[1,2,3,0,0,0,0],[4,5,6,0,0,0,0]],
                    [[7,8,100,9,101,0,0],[100,10,100,11,12,0,0]],
                    [[13,14,15,100,100,100,0],[101,16,17,18,0,0,0]]  ])
    l2 = tf.constant([[3, 3],
                    [5, 5],
                    [6, 4]])
    print 'Shapes: h2:%s, l2:%s'%(K.int_shape(h2), K.int_shape(l2))
    h1_s, l1_s = tfc.squash_3d(3, 2, h1, l1, 100)
    print 'Shapes: h1_s:%s, l1_s:%s'%(K.int_shape(h1_s), K.int_shape(l1_s))
    h2_s, l2_s = tfc.squash_3d(3, 2, h2, l2, 100)
    print 'Shapes: h2_s:%s, l2_s:%s'%(K.int_shape(h2_s), K.int_shape(l2_s))
    ed1 = tfc.edit_distance3D(3, 2, h2, l2, h1, l1, 100, 101)
    mean1 = tf.reduce_mean(ed1)
    acc1 = tf.reduce_mean(tf.to_float(tf.equal(ed1, 0)))
    ed1_s = tfc.edit_distance3D(3, 2, h2_s, l2_s, h1_s, l1_s, 100, 101)
    mean1_s = tf.reduce_mean(ed1_s)
    acc1_s = tf.reduce_mean(tf.to_float(tf.equal(ed1_s, 0)))

    _h1, _l1 = flatten(h1, l1)
    _h1_s, _l1_s = flatten(h1_s, l1_s)
    _h2, _l2 = flatten(h2, l2)
    _h2_s, _l2_s = flatten(h2_s, l2_s)

    _ed1 = tfc.edit_distance2D(6, _h2, _l2, _h1, _l1, 100, 101)
    _mean1 = tf.reduce_mean(_ed1)
    _acc1 = tf.reduce_mean(tf.to_float(tf.equal(_ed1, 0)))
    _ed1_s = tfc.edit_distance2D(6, _h2_s, _l2_s, _h1_s, _l1_s, 100, 101)
    _mean1_s = tf.reduce_mean(_ed1_s)
    _acc1_s = tf.reduce_mean(tf.to_float(tf.equal(_ed1_s, 0)))

    ######################## Tensor with ED > 0
    h2_2 = tf.constant([[[1,2,3,99,0,0,0],[4,5,6,0,0,0,0]],
                    [[7,8,100,99,0,0,0],[100,10,100,11,12,0,0]],
                    [[13,100,15,100,100,100,0],[100,16,17,18,0,0,0]]  ])
    l2_2 = tf.constant([[4, 3],
                    [4, 5],
                    [6, 4]])
    h2_2_s, l2_2_s = tfc.squash_3d(3, 2, h2_2, l2_2, 100)
    _h2_2, _l2_2 = flatten(h2_2, l2_2)
    _h2_2_s, _l2_2_s = flatten(h2_2_s, l2_2_s)

    ed2 = tfc.edit_distance3D(3, 2, h2_2, l2_2, h1, l1, 100, 101)
    acc2 = tf.reduce_mean(tf.to_float(tf.equal(ed2, 0)))
    sum2 = tf.reduce_sum(ed2)
    ed2_s = tfc.edit_distance3D(3, 2, h2_2_s, l2_2_s, h1_s, l1_s, 100, 101)
    acc2_s = tf.reduce_mean(tf.to_float(tf.equal(ed2_s, 0)))
    sum2_s = tf.reduce_sum(ed2_s)
    print 'Shape of ed1=%s'%(K.int_shape(ed1),)
    print 'Shape of ed2_s=%s'%(K.int_shape(ed2_s),)
    _ed2 = tfc.edit_distance2D(6, _h2_2, _l2_2, _h1, _l1, 100, 101)
    _ed2_s = tfc.edit_distance2D(6, _h2_2_s, _l2_2_s, _h1_s, _l1_s, 100, 101)
    _sum2 = tf.reduce_sum(_ed2)
    _sum2_s = tf.reduce_sum(_ed2_s)
    _acc2 = tf.reduce_mean(tf.to_float(tf.equal(_ed2, 0)))
    _acc2_s = tf.reduce_mean(tf.to_float(tf.equal(_ed2_s, 0)))

# tf.reduce_mean(tf.to_float(tf.equal(top1_ed, 0)))

## Test seqlens
    b = []
    for i in range(11):
        b.append([i]*11)
        b[i][i] = 0
    b.append([11]*11)
    b = np.asarray(b)

    tf_b = tf.constant(b)
    tf_lens1 = tfc.seqlens(tf.constant(b))
    tf_lens2 = tfc.seqlens(tf.constant(b.reshape((3,4,11)) ) )
    tf_lens1_2 = tfc.seqlens(tf.constant(b), include_eos_token=False)
    tf_lens2_2 = tfc.seqlens(tf.constant(b.reshape((3,4,11))) ,include_eos_token=False)
    len_1 = np.arange(1,13); len_1[11] = 11
    len_2 = np.arange(12)

with tf.Session():
    print 'ed1 = \n%s'%ed1.eval()
    assert mean1.eval() == 0.
    assert acc1.eval() == 1
    print '_ed1 = \n%s'%_ed1.eval()
    assert _mean1.eval() == 0.
    assert _acc1.eval() == 1
    print 'ed1_s = \n%s'%ed1_s.eval()
    assert mean1_s.eval() == 0.
    assert acc1_s.eval() == 1
    print '_ed1_s = \n%s'%_ed1_s.eval()
    assert _mean1_s.eval() == 0.
    assert _acc1_s.eval() == 1


    print 'ed2 = \n%s'%ed2.eval()
    assert sum2.eval() == 1.
    assert acc2.eval() == 1./2.
    print '_ed2 = \n%s'%_ed2.eval()
    assert _sum2.eval() == 1.
    assert _acc2.eval() == 1./2.
    print 'ed2_s = \n%s'%ed2_s.eval()
    assert sum2_s.eval() == 1.
    assert acc2_s.eval() == 1./2.
    print '_ed2_s = \n%s'%_ed2_s.eval()
    assert _sum2_s.eval() == 1.
    assert _acc2_s.eval() == 1./2.

    print tf_lens1.eval()
    print tf_lens1_2.eval()
    assert np.all(tf_lens1.eval() == len_1 )
    assert np.all(tf_lens1_2.eval() == len_2)
    print tf_lens2.eval()
    print tf_lens2_2.eval()
    assert np.all(tf_lens2.eval() == len_1.reshape(3,4))
    assert np.all(tf_lens2_2.eval() == len_2.reshape(3,4))
print "Success !"
