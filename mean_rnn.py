from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import os
import sys
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import cPickle
import math

linear = tf.nn.rnn_cell._linear
class MeanRNN(object):
  def __init__(self, feature_size, vocab_size, max_sentence_length, size, num_layers, use_lstm=False, forward_only=False, dtype=tf.float32):
    self.feature_size = feature_size
    self.vocab_size = vocab_size
    self.max_sentence_length = max_sentence_length
    self.global_step = tf.Variable(0, trainable=False)

    single_cell = tf.nn.rnn_cell.GRUCell(size)
    if use_lstm:
      single_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=size, state_is_tuple=False)
    cell = single_cell
    if num_layers > 1:
      cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=False)
    cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, vocab_size)
    
    # Feeds for inputs.
    self.feature_inputs = tf.placeholder(tf.float32, shape=[None, feature_size], name="feature")
    self.decoder_inputs = []
    self.target_weights = []
    for i in xrange(max_sentence_length):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
      self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))
    self.targets = [self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) -1)]
    self.targets.append(tf.placeholder(tf.int32, shape=[None], name="last_target"))
    
    state = linear(self.feature_inputs, cell.state_size, True)
    # Training outputs and losses.
    if forward_only:
      self.outputs, _ = tf.nn.seq2seq.embedding_rnn_decoder(self.decoder_inputs, state, cell, vocab_size, size, feed_previous=True, update_embedding_for_previous=False)
    else:
      self.outputs, _ = tf.nn.seq2seq.embedding_rnn_decoder(self.decoder_inputs, state, cell, vocab_size, size, feed_previous=False)
      self.loss = tf.nn.seq2seq.sequence_loss(self.outputs, self.targets, self.target_weights)
      self.update = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss, self.global_step, tf.trainable_variables())

    self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=999999999)

  def step(self, session, feature_inputs, decoder_inputs, target_weights, forward_only, batch_size):
    input_feed = {}
    input_feed[self.feature_inputs.name] = feature_inputs

    for l in xrange(self.max_sentence_length):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]
    last_target = self.targets[-1].name
    input_feed[last_target] = np.zeros([batch_size], dtype=np.int32)
    
    if forward_only:
      output_feed = [self.outputs]
    else:
      output_feed = [self.loss, self.update]

    outputs = session.run(output_feed, input_feed)
    return outputs[0]

  def get_batch(self, features, sentences, NO, batch_size):
    feature_inputs, decoder_inputs = [], []
    for i in NO:
      sen, vid = sentences[i]
      pad_size = self.max_sentence_length - len(sen) - 2
      decoder_inputs.append([data_utils.GO_ID] + sen + [data_utils.EOS_ID] + [data_utils.PAD_ID] * pad_size)
      feature_inputs.append(features[vid])

    batch_decoder_inputs, batch_weights = [], []
    for length_idx in xrange(self.max_sentence_length):
      batch_decoder_inputs.append(np.array([decoder_inputs[batch_idx][length_idx] for batch_idx in xrange(batch_size)], dtype=np.int32))
      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(batch_size, dtype=np.float32)
      for batch_idx in xrange(batch_size):
        if length_idx < self.max_sentence_length - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == self.max_sentence_length - 1 or target == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return feature_inputs, batch_decoder_inputs, batch_weights

def create_model(session, feature_size, vocab_size, max_sentence_length, size, num_layers, use_lstm, forward_only):
  model = MeanRNN(feature_size, vocab_size, max_sentence_length, size, num_layers, use_lstm, forward_only)
  ckpt = tf.train.get_checkpoint_state('CheckPoint')
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model

def train(feature_size, vocab_size, max_sentence_length, size, num_layers, use_lstm, batch_size, steps_per_checkpoint):
  with open ('data/msr-vtt-video-mean-pooling-feature.pkl', 'rb') as f:
    feature = cPickle.load(f)
  with open ('data/sentences.pkl', 'rb') as f:
    sentence = cPickle.load(f)
  with open ('data/sentence_info.pkl', 'rb') as f:
    info = cPickle.load(f)['train']

  with tf.Session() as sess:
    print("Creating %d layers of %d units." % (num_layers, size))
    model = create_model(sess, feature_size, vocab_size, max_sentence_length, size, num_layers, use_lstm, False)
    
    current_step = 0
    while True:
      np.random.shuffle(info)
      for start,end in zip(range(0, len(info), batch_size), range(batch_size, len(info), batch_size)):
        feature_inputs, batch_decoder_inputs, batch_weights = model.get_batch(feature, sentence, info[start:end], batch_size)
        step_loss = model.step(sess, feature_inputs, batch_decoder_inputs, batch_weights, False, batch_size)
        current_step += 1
        if current_step % steps_per_checkpoint == 0:
          print ("global step %d - loss %.3f" % (model.global_step.eval(), step_loss))
          checkpoint_path = os.path.join('CheckPoint', 'ckpt')
          model.saver.save(sess, checkpoint_path, global_step=model.global_step)
          sys.stdout.flush()

def evaluate(feature_size, vocab_size, max_sentence_length, size, num_layers, use_lstm):
  from pycocoevalcap.bleu.bleu import Bleu
  from pycocoevalcap.rouge.rouge import Rouge
  from pycocoevalcap.cider.cider import Cider
  from pycocoevalcap.meteor.meteor import Meteor
  from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

  with open('data/validate.pkl', 'rb') as f:
    data = cPickle.load(f)
  scorers = [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]), (Meteor(),"METEOR"), (Rouge(), "ROUGE_L"), (Cider(), "CIDEr")]
  vocab, re_vocab = data_utils.initialize_vocabulary('data/vocab.txt')
  GTS = {}
  RES = {}
  batch_size = 1
  with tf.Session() as sess:
    model = create_model(sess, feature_size, vocab_size, max_sentence_length, size, num_layers, use_lstm, True)
    for idx, (feature, caption) in enumerate(data):
      feature_inputs, batch_decoder_inputs, batch_weights = model.get_batch([feature], [([0], 0)], [0], 1)
      output_logits = model.step(sess, feature_inputs, batch_decoder_inputs, batch_weights, forward_only=True, batch_size=1)
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      sentence = " ".join([tf.compat.as_str(re_vocab[output]) for output in outputs])
      print ("%s: %s"%(sentence, caption[9]))
      GTS[idx] = caption
      RES[idx] = [sentence]
  for scorer, method in scorers:
    score, scores = scorer.compute_score(GTS, RES)
    if isinstance(method, list):
      for k, v in zip(method, score):
        print("%s:\t%s"%(k, v))
    else:
      print("%s:\t%s"%(method, score))
if __name__ == "__main__":
  train(feature_size=2048, vocab_size=8110, max_sentence_length=40, size=1024, num_layers=4, use_lstm=False, batch_size=64, steps_per_checkpoint=200)
#  evaluate(feature_size=2048, vocab_size=8110, max_sentence_length=40, size=1024, num_layers=1, use_lstm=False)

