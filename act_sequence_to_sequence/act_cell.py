from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops.nn import rnn_cell
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops

class ACTCell(rnn_cell.RNNCell):
  """An RNN cell implementing Graves' Adaptive Computation Time algorithm"""
  def __init__(self, cell, epsilon, max_computation):
    self.epsilon = epsilon
    self.cell = cell
    self.state_is_tuple = isinstance(cell.state_size, (tuple, list))
    self.max_computation = max_computation
    self.ACT_remainder = []
    self.ACT_iterations = []

  @property
  def output_size(self):
    return self.cell.output_size

  @property
  def state_size(self):
    return self.cell.state_size

  def __call__(self, inputs, state, scope=None):
    with vs.variable_scope(scope or type(self).__name__):
      # define within cell constants/ counters used to control while loop for ACTStep
      if self.state_is_tuple:
        state = array_ops.concat(1, state)

      self.batch_size = tf.shape(inputs)[0]
      self.one_minus_eps = tf.fill([self.batch_size], tf.constant(1.0 - self.epsilon, dtype=tf.float32))
      prob = tf.fill([self.batch_size], tf.constant(0.0, dtype=tf.float32), "prob")
      counter = tf.zeros_like(prob, tf.float32, name="counter")
      acc_outputs = tf.fill([self.batch_size, self.output_size], 0.0, name='output_accumulator')
      acc_states = tf.zeros_like(state, tf.float32, name="state_accumulator")
      flag = tf.fill([self.batch_size], True, name="flag")

      pred = lambda flag, prob, counter, state, inputs, acc_outputs, acc_states: tf.reduce_any(flag)

      _, probs, iterations, _, _, output, next_state = control_flow_ops.while_loop(pred, self.act_step, loop_vars=[flag, prob, counter, state, inputs, acc_outputs, acc_states])

    self.ACT_remainder.append(1 - probs)
    self.ACT_iterations.append(iterations)

    if self.state_is_tuple:
      next_c, next_h = array_ops.split(1, 2, next_state)
      next_state = rnn_cell._LSTMStateTuple(next_c, next_h)

    return output, next_state

  def calculate_ponder_cost(self):
    """returns tensor of shape [1] which is the total ponder cost"""
    remainder = [tf.reduce_mean(R) for R in self.ACT_remainder]
    iteration = [tf.reduce_mean(I) for I in self.ACT_iterations]
    return tf.reduce_sum(tf.add_n(remainder) / len(remainder) + tf.to_float(tf.add_n(iteration) / len(iteration)))

  def act_step(self, flag, prob, counter, state, inputs, acc_outputs, acc_states):
    if self.state_is_tuple:
      (c, h) = array_ops.split(1, 2, state)
      state = rnn_cell._LSTMStateTuple(c, h)
      state_size = sum(self.state_size)
    else:
      state_size = self.state_size

    binary_flag = tf.select(tf.equal(prob, 0.0), tf.ones([self.batch_size,1], dtype=tf.float32), tf.zeros([self.batch_size,1], tf.float32))
    input_with_flags = tf.concat(1, [binary_flag, inputs])

    output, new_state = self.cell(input_with_flags, state)

    if self.state_is_tuple:
      new_state = array_ops.concat(1, new_state)
    with tf.variable_scope('sigmoid_activation_for_pondering'):
      p = tf.squeeze(tf.sigmoid(tf.nn.rnn_cell._linear([inputs, new_state], 1, bias=True)), squeeze_dims=1)  # haulting unit
      # p = tf.squeeze(tf.sigmoid(tf.nn.rnn_cell._linear([new_state], 1, bias=True)), squeeze_dims=1)  # haulting unit

    curr_flag = tf.logical_and(tf.less(prob+p, self.one_minus_eps), tf.less(counter+1, self.max_computation))

    # Normal
    p_expanded = tf.expand_dims(p, 1)
    tiled_p_outputs = tf.tile(p_expanded, [1, self.output_size])
    tiled_p_states = tf.tile(p_expanded, [1, state_size])
    norm_acc_output = (output * tiled_p_outputs) + acc_outputs
    norm_acc_state = (new_state * tiled_p_states) + acc_states

    # Remainder
    remainder = tf.fill([self.batch_size], tf.constant(1.0, dtype=tf.float32)) - prob
    remainder_expanded = tf.expand_dims(remainder, 1)
    tiled_remainder_output = tf.tile(remainder_expanded, [1, self.output_size])
    tiled_remainder_states = tf.tile(remainder_expanded, [1, state_size])
    remd_acc_output = (output * tiled_remainder_output) + acc_outputs
    remd_acc_state = (new_state * tiled_remainder_states) + acc_states

    # Select
    acc_outputs = tf.select(flag, tf.select(curr_flag, norm_acc_output, remd_acc_output), acc_outputs)
    acc_states = tf.select(flag, tf.select(curr_flag, norm_acc_state, remd_acc_state), acc_states)

    # Update
    prob = tf.select(flag, tf.select(curr_flag, prob+p, prob), prob)
    counter = tf.select(flag, counter+1, counter)
    flag = tf.logical_and(flag, curr_flag)

    return [flag, prob, counter, state, inputs, acc_outputs, acc_states]
