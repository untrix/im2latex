#!/usr/bin/env python2

import tensorflow as tf
import tf_commons as tfc
import numpy as np

b = []
for i in range(11):
    b.append([i]*11)
    b[i][i] = 0
b.append([11]*11)
b = np.asarray(b)

tf_b = tf.constant(b)
tf_lens1 = tfc.seqlens(tf.constant(b))
tf_lens2 = tfc.seqlens(tf.constant(b.reshape((3,4,11)) ) )

with tf.Session() as session:
    print tf_lens1.eval()
    print tf_lens2.eval()
