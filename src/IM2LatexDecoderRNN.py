#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 15:33:36 2017

@author: sumeet
"""

class Im2LatexDecoderRNN(tf.nn.rnn_cell.RNNCell):
    """
    One timestep of the decoder model. The entire function can be seen as a complex RNN-cell
    that includes a LSTM stack and an attention model.
    """

  def __init__(self, params, reuse=None):
    super(Im2LatexDecoderRNN, self).__init__(_reuse=reuse)
    

  @property
  def state_size(self):
    # lstm_states_t, a, yProbs_t, yLogits_t, alpha_t
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

    def _build_rnn_step(self, out_t_1, x_t, isStep1=False, testing=False):
        """
        TODO: Incorporate Dropout
        Builds/threads tf graph for one RNN iteration.
        Conforms to loop function fn required by tf.scan. Takes in previous lstm states (h and c), 
        the current input and the image annotations (a) as input and outputs the states and outputs for the
        current timestep.
        Note that input(t) = Ey(t-1). Input(t=0) = Null. When training, the target output is used for Ey
        whereas at prediction time (via. beam-search for e.g.) the actual output is used.
        Args:
            x_t (tensor): is a input for one time-step. Should be a tensor of shape (batch-size, 1).
            out_t_1 (tuple of tensors): Output returned by this function at previous time-step.
        Returns:
            out_t (tuple of tensors): The output y_t shape= (B,K) - the probability of words/tokens. Also returns
                states needed in the next iteration of the RNN - i.e. (h_t, lstm_states_t and a). lstm_states_t = 
                (h_t, c_t) - which means h_t is included twice in the returned tuple.            
        """
        #x_t = input at t             # shape = (B,)
        step = out_t_1[0] + 1
        h_t_1 = out_t_1[1]            # shape = (B,n)
        lstm_states_t_1 = out_t_1[2]  # shape = ((B,n), (B,n)) = (c_t_1, h_t_1)
        a = out_t_1[3]                # shape = (B, L, D)
        if not isStep1: ## init_accum does not have everything
            yProbs_t_1 = out_t_1[4]           # shape = (B, Kv)
        #yLogits_t_1 = out_t_1[5]          # shape = (B, Kv)
        #alpha_t_1 = out_t_1[6]
        
        B = HYPER.B
        m = HYPER.m
        n = HYPER.n
        L = HYPER.L
        D = HYPER.D
        Kv = HYPER.K
        
        assert K.int_shape(h_t_1) == (B, n)
        assert K.int_shape(a) == (B, L, D)
        assert K.int_shape(lstm_states_t_1[1]) == (B, n)
        
        if not isStep1:
            assert K.int_shape(yProbs_t_1) == (B, Kv)
            tf.get_variable_scope().reuse_variables()
            if testing:
                x_t = tf.argmax(yProbs_t_1, axis=1)
        elif testing:
            tf.get_variable_scope().reuse_variables()
        
        ################ Attention Model ################
        with tf.variable_scope('Attention'):
            alpha_t = self._build_attention_model(a, h_t_1) # alpha.shape = (B, L)

        ################ Soft deterministic attention: z = alpha-weighted mean of a ################
        ## (B, L) batch_dot (B,L,D) -> (B, D)
        with tf.variable_scope('Phi'):
            z_t = K.batch_dot(alpha_t, a, axes=[1,1]) # z_t.shape = (B, D)

        ################ Embedding layer ################
        with tf.variable_scope('Ey'):
            Ex_t = self._embedding(K.expand_dims(x_t, axis=-1) ) # output.shape= (B,1,m)
            Ex_t = K.squeeze(Ex_t, axis=1) # output.shape= (B,m)

        ################ Decoder Layer ################
        with tf.variable_scope("Decoder_LSTM") as var_scope:
            (h_t, lstm_states_t) = self._decoder_lstm(Ex_t, lstm_states_t_1) # h_t.shape=(B,n)
            
        ################ Decoder Layer ################
        with tf.variable_scope('Output_Layer'):
            yProbs_t, yLogits_t = self._build_output_layer(Ex_t, h_t, z_t) # yProbs_t.shape = (B,K)
        
        assert K.int_shape(h_t) == (B, n)
        assert K.int_shape(a) == (B, L, D)
        assert K.int_shape(lstm_states_t[1]) == (B, n)
        assert K.int_shape(yProbs_t) == (B, Kv)
        assert K.int_shape(yLogits_t) == (B, Kv)
        assert K.int_shape(alpha_t) == (B, L)
        
        return step, h_t, lstm_states_t, a, yProbs_t, yLogits_t, alpha_t
      
  def call(self, inputs, state):
    """Most basic RNN: output = new_state = act(W * input + U * state + B)."""
    output = self._activation(_linear([inputs, state], self._num_units, True))
    return output, output
