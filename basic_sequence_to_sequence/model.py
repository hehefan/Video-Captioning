from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
import data_utils

from seq2seq import embedding_rnn_decoder

linear = tf.nn.rnn_cell._linear

class Seq2Seq(object):
  def __init__(self, num_units, use_lstm, encoder_max_sequence_length, decoder_max_sentence_length, feature_size, vocab_size, learning_rate, learning_rate_decay_factor, max_gradient_norm, forward_only=False):
    self.feature_size = feature_size
    self.vocab_size = vocab_size
    self.encoder_max_sequence_length = encoder_max_sequence_length
    self.decoder_max_sentence_length = decoder_max_sentence_length
    self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
    self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)

    encoder = tf.nn.rnn_cell.GRUCell(num_units)
    if use_lstm:
      encoder = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units, state_is_tuple=False)
    
    decoder = tf.nn.rnn_cell.GRUCell(num_units)
    if use_lstm:
      decoder = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units, state_is_tuple=False)
    decoder = tf.nn.rnn_cell.OutputProjectionWrapper(decoder, vocab_size)

    # Feeds for inputs.
    self.encoder_inputs = []
    self.decoder_inputs = []
    self.target_weights = []
    for i in xrange(encoder_max_sequence_length):
      self.encoder_inputs.append(tf.placeholder(tf.float32, shape=[None, feature_size], name="encoder{0}".format(i)))
    for i in xrange(decoder_max_sentence_length):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
      self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))
    self.targets = [self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) -1)]
    self.targets.append(tf.placeholder(tf.int32, shape=[None], name="last_target"))

    with variable_scope.variable_scope("embedding_rnn_seq2seq") as scope:
      encoder_outputs, self.encoder_state = rnn.rnn(cell=encoder, inputs=self.encoder_inputs, dtype=tf.float32)
      if forward_only:
        self.outputs, self.decoder_states = embedding_rnn_decoder(self.decoder_inputs, self.encoder_state, decoder, vocab_size, num_units, feed_previous=True, update_embedding_for_previous=False)
      else:
        self.outputs, self.decoder_states = embedding_rnn_decoder(self.decoder_inputs, self.encoder_state, decoder, vocab_size, num_units, feed_previous=False)

    # Training outputs and losses.
    if not forward_only:
      self.loss = tf.nn.seq2seq.sequence_loss(self.outputs, self.targets, self.target_weights)
      gradients = tf.gradients(self.loss, tf.trainable_variables())
      clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
      self.update = tf.train.AdamOptimizer(learning_rate=self.learning_rate).apply_gradients(zip(clipped_gradients, tf.trainable_variables()), global_step=self.global_step)

    self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=99999999)

  def get_batch(self, features, sentences):
    batch_size = len(sentences)
    encoder_inputs, decoder_inputs = [], []
    feature_pad = np.array([0.0] * self.feature_size)
    for (vid, sen) in sentences:
      feature = features[vid]
      if len(feature) > self.encoder_max_sequence_length:
        feature = random.sample(feature, self.encoder_max_sequence_length)
      pad_size = self.encoder_max_sequence_length - len(feature)
      encoder_inputs.append(feature + [feature_pad] * pad_size)

      pad_size = self.decoder_max_sentence_length - len(sen) - 2
      decoder_inputs.append([data_utils.GO_ID] + sen + [data_utils.EOS_ID] + [data_utils.PAD_ID] * pad_size)

    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
    for length_idx in xrange(self.encoder_max_sequence_length):
      batch_encoder_inputs.append(np.array([encoder_inputs[batch_idx][length_idx] for batch_idx in xrange(batch_size)], dtype=np.float32))

    for length_idx in xrange(self.decoder_max_sentence_length):
      batch_decoder_inputs.append(np.array([decoder_inputs[batch_idx][length_idx] for batch_idx in xrange(batch_size)], dtype=np.int32))
      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(batch_size, dtype=np.float32)
      for batch_idx in xrange(batch_size):
        if length_idx < self.decoder_max_sentence_length - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == self.decoder_max_sentence_length - 1 or target == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights

  def step(self, session, encoder_inputs, decoder_inputs, target_weights, forward_only):
    batch_size, _ = encoder_inputs[0].shape
    input_feed = {}
    for l in xrange(self.encoder_max_sequence_length):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]

    for l in xrange(self.decoder_max_sentence_length):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]
    last_target = self.targets[-1].name
    input_feed[last_target] = np.zeros([batch_size], dtype=np.int32)
    
    if forward_only:
      output_feed = [self.outputs, self.encoder_state, self.decoder_states]
    else:
      output_feed = [self.loss, self.encoder_state, self.decoder_states, self.update]

    outputs = session.run(output_feed, input_feed)
    return outputs[0], outputs[1], outputs[2]

  # For beam search.
  def state(self, session, encoder_inputs):
    input_feed = {}
    for l in xrange(self.encoder_max_sequence_length):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]

    output_feed = [self.encoder_state]
    outputs = session.run(output_feed, input_feed)

    return outputs[0]
  
  def one_step(self, session, decoder_state, decoder_input):
    batch_size = decoder_input[0].shape[0]
    input_feed = {}
    input_feed[self.encoder_state.name] = decoder_state
    
    for l in xrange(self.decoder_max_sentence_length): # decoder_max_sentence_length = 1
      input_feed[self.decoder_inputs[l].name] = decoder_input[l]
      input_feed[self.target_weights[l].name] = np.zeros([batch_size], dtype=np.int32)
    last_target = self.targets[-1].name
    input_feed[last_target] = np.zeros([batch_size], dtype=np.int32)
    
    output_feed = [self.outputs, self.decoder_states]
    outputs = session.run(output_feed, input_feed)

    return outputs[0], outputs[1]
