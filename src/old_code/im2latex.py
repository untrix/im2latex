
# coding: utf-8

# # im2latex(S): Deep Learning Model
# 
# &copy; Copyright 2017 Sumeet S Singh
# 
#     This file is part of the im2latex solution (by Sumeet S Singh in particular since there are other solutions out there).
# 
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the Affero GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
# 
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     Affero GNU General Public License for more details.
# 
#     You should have received a copy of the Affero GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.

# ## The Model
# * Follows the [Show, Attend and Tell paper](https://www.semanticscholar.org/paper/Show-Attend-and-Tell-Neural-Image-Caption-Generati-Xu-Ba/146f6f6ed688c905fb6e346ad02332efd5464616)
# * [VGG ConvNet (16 or 19)](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) without the top-3 layers
#     * Pre-initialized with the VGG weights but allowed to train
#     * The ConvNet outputs $D$ dimensional vectors in a WxH grid where W and H are 1/16th of the input image size (due to 4 max-pool layers). Defining $W.H \equiv L$ the ConvNet output represents L locations of the image $i \in [1,L]$ and correspondingly outputs to L annotation vectors $a_i$, each of size $D$.
# * A dense (FC) attention model: The deterministic soft-attention model of the paper computes $\alpha_{t,i}$ which is used to select or blend the $a_i$ vectors before being fed as inputs to the decoder LSTM network (see below).
#     * Inputs to the attention model are $a_i$ and $h_{t-1}$ (previous hidden state of LSTM network - see below)
#     and $$\alpha_{t,i} = softmax ( f_{att}(a_i, h_{t-1}) )$$
# * A Decoder model: A conditioned LSTM that outputs probabilities of the text tokens $y_t$ at each step. The LSTM is conditioned upon $z_t = \sum_i^L(\alpha_{t,i}.a_i)$ and takes the previous hidden state $h_{t-1}$ as input. In addition, an embedding of the previous output $Ey_{t-1}$ is also input to the LSTM. At training time, $y_{t-1}$ would be derived from the training samples, while at inferencing time it would be fed-back from the previous predicted word.
#     * $y$ is taken from a fixed vocabulary of K words. An embedding matrix $E$ is used to narrow its representation. The embedding weights $E$ are learnt end-to-end by the model as well.
#     * The decoder LSTM uses a deep layer between $h_t$ and $y_t$. It is called a deep output layer and is described in [section 3.2.2 of this paper](https://www.semanticscholar.org/paper/How-to-Construct-Deep-Recurrent-Neural-Networks-Pascanu-G%C3%BCl%C3%A7ehre/533ee188324b833e059cb59b654e6160776d5812). That is:
#     $$ p(y_t) = Softmax \Big( f_out(Ey_{t-1}, h_t, \hat{z}_t) \Big) $$
# * Initialization MLPs: Two MLPs are used to produce the initial memory-state of the LSTM as well as $h_{t-1}$ value. Each MLP takes in the entire image's features (i.e. average of $a_i$) as its input and is trained end-to-end.
#     $$ c_o = f_{init,c}\Big( \sum_i^L a_i \Big) $$
#     $$ h_o = f_{init,h}\Big( \sum_i^L a_i \Big) $$
# * Training:
#     * 3 models from above - all except the conv-net - are trained end-to-end using SGD
#     * The model is trained for a variable number of time steps - depending on each batch

# ## References
# 1. Show, Attend and Tell
#     * [Paper](https://www.semanticscholar.org/paper/Show-Attend-and-Tell-Neural-Image-Caption-Generati-Xu-Ba/146f6f6ed688c905fb6e346ad02332efd5464616)
#     * [Slides](https://pdfs.semanticscholar.org/b336/f6215c3c15802ca5327cd7cc1747bd83588c.pdf?_ga=2.52116077.559595598.1498604153-2037060338.1496182671)
#     * [Author's Theano code](https://github.com/kelvinxu/arctic-captions)
# 1. [Simonyan, Karen and Andrew Zisserman. “Very Deep Convolutional Networks for Large-Scale Image Recognition.” CoRR abs/1409.1556 (2014): n. pag.](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)
# 1. [im2latex solution of Harvard NLP](http://lstm.seas.harvard.edu/latex/)
# 1. [im2latex-dataset tools forked from Harvard NLP](https://github.com/untrix/im2latex-dataset)

# In[1]:


import pandas as pd
import os
from six.moves import cPickle as pickle
import dl_commons as dlc
import tf_commons as tfc
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Embedding, Dense, Activation, Dropout, Concatenate, Permute
from keras.callbacks import LambdaCallback
from keras.models import Model
from keras import backend as K
from keras.engine import Layer
import keras
import threading
import numpy as np
import collections
import data_reader as dr
import hyper_params


# # TODOs
# * Implement the beta scalar ('selector') that scales alpha.

# In[2]:


data_folder = '../data/generated2'
image_folder = os.path.join(data_folder,'formula_images')
raw_data_folder = os.path.join(data_folder, 'training', 'temp_dir')
vgg16_folder = os.path.join(data_folder, 'training', 'vgg16_features')


# ### HyperParams

# ### Encoder Model
# [VGG ConvNet (16 or 19)](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) without the top-3 layers
# * Pre-initialized with the VGG weights but allowed to train
# * The ConvNet outputs $D$ dimensional vectors in a WxH grid where W and H are scaled-down dimensions of the input image size (due to 5 max-pool layers). Defining $W.H \equiv L$ the ConvNet output represents L locations of the image $i \in [1,L]$ and correspondingly outputs to L annotation vectors $a_i$, each of size $D$.
# 
# The conv-net is *not trained* in the original paper and therefore the files can be separately preprocessed and their outputs directly fed into the model.

# #### Decoder Model
# A dense (FC) attention model: The deterministic soft-attention model of the paper computes $\alpha_{t,i}$ which is used to select or blend the $a_i$ vectors before being fed as inputs to the decoder LSTM network (see below).
# * Inputs to the attention model are $a_i$ and $h_{t-1}$ (previous hidden state of LSTM network - see below) and $$\alpha_{t,i} = softmax ( f_{att}(a_i, h_{t-1}) )$$
# * Note that the model $f_{att}$ shares weights across all values of a_i (i.e. for all i = 1-L). Therefore the shared weight matrix for all a_i has shape (D, D), while shape of a is (B, L, D) where is B=batch-size. Weight matrix of h_i is separate and has the expected shape (n, D). This sharing of weights across a_i is interesting.
# 
# A Decoder model: A conditioned LSTM that outputs probabilities of the text tokens $y_t$ at each step. The LSTM is conditioned upon $z_t = \sum_i^L(\alpha_{t,i}.a_i)$ and takes the previous hidden state $h_{t-1}$ as input. In addition, an embedding of the previous output $Ey_{t-1}$ is also input to the LSTM. At training time, $y_{t-1}$ would be derived from the training samples, while at inferencing time it would be fed-back from the previous predicted word.
# * $y$ is taken from a fixed vocabulary of K words. An embedding matrix $E$ is used to narrow its representation to an $m$ dimensional dense vector. The embedding weights $E$ are learnt end-to-end by the model as well.
# * The decoder LSTM uses a deep layer between $h_t$ and $y_t$. It is called a deep output layer and is described in [section 3.2.2 of this paper](https://www.semanticscholar.org/paper/How-to-Construct-Deep-Recurrent-Neural-Networks-Pascanu-G%C3%BCl%C3%A7ehre/533ee188324b833e059cb59b654e6160776d5812). That is:
# $$ p(y_t) = Softmax \Big( f_out(Ey_{t-1}, h_t, \hat{z}_t) \Big) $$
# * Optionally $z_t = \beta \sum_i^L(\alpha_{t,i}.a_i)$ where $\beta = \sigma(f_{\beta}(h_{t-1}))$ is a scalar used to modulate the strength of the context. It turns out that for the original use-case of caption generation, the network would learn to emphasize objects by turning up the value of this scalar when it was focusing on objects. It is not clear at this time whether we'll need this feature for im2latex.
# 

# ### Input Generator

# In[7]:


HYPER = hyper_params.make_hyper({'B':128, 'assert_whole_batch':False})
b_it = dr.BatchContextIterator(raw_data_folder, vgg16_folder, HYPER)


# In[8]:


def func(x=None):
    d = b_it.next()
    return (d.y_s, d.seq_len, d.im)
pyfunc = tf.py_func(func,[0],[tf.int32, tf.int32, tf.float32])


# In[9]:


q = tf.FIFOQueue(10, [tf.int32, tf.int32, tf.float32])
enqueue_op = q.enqueue(pyfunc)
inputs = q.dequeue()


# In[18]:


b_it.next()
b_it.next()


# In[10]:


sess = tf.Session()
for i in range(100):
    sess.run([enqueue_op])
    print sess.run([inputs])[0].shape


# In[13]:


sess.run([enqueue_op])


# In[14]:


print sess.run([enqueue_op, inputs])[1].shape


# # End
