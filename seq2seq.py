from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import os
import sys
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import rnn
from tensorflow.python.ops import variable_scope
import data_utils
import cPickle
import math
from tensorflow.python.util import nest
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("encoder_size", 1024, "Size of each encoder layer.")
tf.app.flags.DEFINE_integer("encoder_num_layers", 1, "Number of layers in the encoder.")
tf.app.flags.DEFINE_integer("feature_size", 2048, "Size of frame feature vector.")
tf.app.flags.DEFINE_integer("encoder_max_sequence_length", 16, "Max length of encoder sequence.")
tf.app.flags.DEFINE_boolean("encoder_use_lstm", False, "LSTM or GRU for encoder.")
tf.app.flags.DEFINE_integer("decoder_size", 1024, "Size of each decoder layer.")
tf.app.flags.DEFINE_integer("decoder_num_layers", 1, "Number of layers in the decoder.")
tf.app.flags.DEFINE_integer("vocab_size", 8110, "Size of English vocabulary.")
tf.app.flags.DEFINE_integer("decoder_max_sentence_length", 40, "Max length of decoder sentence.")
tf.app.flags.DEFINE_boolean("decoder_use_lstm", False, "LSTM or GRU for decoder.")
tf.app.flags.DEFINE_string("checkpoint_dir", "CheckPoint", "Checkpoint directory.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200, "How many training steps to do per checkpoint.")

FLAGS = tf.app.flags.FLAGS

linear = tf.nn.rnn_cell._linear

def embedding_rnn_seq2seq(encoder_inputs, decoder_inputs, encoder_cell, decoder_cell, num_decoder_symbols, decoder_size, feed_previous=False, dtype=None, scope=None):
  with variable_scope.variable_scope(scope or "embedding_rnn_seq2seq") as scope:
    _, state = rnn.rnn(encoder_cell, encoder_inputs, dtype=tf.float32)
    state = linear(state, decoder_cell.state_size, True)
    
    decoder = tf.nn.rnn_cell.OutputProjectionWrapper(decoder_cell, num_decoder_symbols)
    if feed_previous:
      return tf.nn.seq2seq.embedding_rnn_decoder(decoder_inputs, state, decoder, num_decoder_symbols, decoder_size, feed_previous=True, update_embedding_for_previous=False)
    else:
      return tf.nn.seq2seq.embedding_rnn_decoder(decoder_inputs, state, decoder, num_decoder_symbols, decoder_size, feed_previous=False)

class Seq2Seq(object):
  def __init__(self, feature_size, vocab_size, learning_rate, max_gradient_norm, encoder_max_sequence_length, encoder_size, encoder_num_layers, encoder_use_lstm, decoder_max_sentence_length, decoder_size, decoder_num_layers, decoder_use_lstm, forward_only=False):
    self.feature_size = feature_size
    self.vocab_size = vocab_size
    self.encoder_max_sequence_length = encoder_max_sequence_length
    self.decoder_max_sentence_length = decoder_max_sentence_length

    self.global_step = tf.Variable(0, trainable=False)

    encoder = tf.nn.rnn_cell.GRUCell(encoder_size)
    if encoder_use_lstm:
      encoder = tf.nn.rnn_cell.BasicLSTMCell(num_units=encoder_size, state_is_tuple=False)
    if encoder_num_layers > 1:
      encoder = tf.nn.rnn_cell.MultiRNNCell([encoder] * encoder_num_layers, state_is_tuple=False)
    
    decoder = tf.nn.rnn_cell.GRUCell(decoder_size)
    if decoder_use_lstm:
      decoder = tf.nn.rnn_cell.BasicLSTMCell(num_units=decoder_size, state_is_tuple=False)
    if decoder_num_layers > 1:
      decoder = tf.nn.rnn_cell.MultiRNNCell([decoder] * decoder_num_layers, state_is_tuple=False)

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

    self.outputs, _ = embedding_rnn_seq2seq(self.encoder_inputs, self.decoder_inputs, encoder, decoder, num_decoder_symbols=vocab_size,decoder_size=decoder_size, feed_previous=forward_only)
    # Training outputs and losses.
    if not forward_only:
      self.loss = tf.nn.seq2seq.sequence_loss(self.outputs, self.targets, self.target_weights)
      gradients = tf.gradients(self.loss, tf.trainable_variables())
      clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
      self.update = tf.train.AdamOptimizer(learning_rate=learning_rate).apply_gradients(zip(clipped_gradients, tf.trainable_variables()), global_step=self.global_step)

    self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=999999999)

  def step(self, session, encoder_inputs, decoder_inputs, target_weights, forward_only, batch_size):
    input_feed = {}
    for l in xrange(self.encoder_max_sequence_length):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]

    for l in xrange(self.decoder_max_sentence_length):
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
    encoder_inputs, decoder_inputs = [], []
    feature_pad = np.array([0.0] * self.feature_size)
    for i in NO:
      sen, vid = sentences[i]
      feature = features[vid]
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

def create_model(session, forward_only, step=None):
  model = Seq2Seq(FLAGS.feature_size, FLAGS.vocab_size, FLAGS.learning_rate, FLAGS.max_gradient_norm, FLAGS.encoder_max_sequence_length, FLAGS.encoder_size, FLAGS.encoder_num_layers, FLAGS.encoder_use_lstm, FLAGS.decoder_max_sentence_length, FLAGS.decoder_size, FLAGS.decoder_num_layers, FLAGS.decoder_use_lstm, forward_only)
  ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    if step == None:
      print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
      ckpt_path = os.path.join(FLAGS.checkpoint_dir,'ckpt-%d'%step)
      print("Reading model parameters from %s" % ckpt_path)
      model.saver.restore(session, ckpt_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model

def train():
  with open ('data/msr-vtt-video-feature-15.pkl', 'rb') as f:
    feature = cPickle.load(f)
  with open ('data/sentences.pkl', 'rb') as f:
    sentence = cPickle.load(f)
  with open ('data/sentence_info.pkl', 'rb') as f:
    info = cPickle.load(f)['train']

  with tf.Session() as sess:
    model = create_model(sess, False)
    current_step = 0
    while True:
      np.random.shuffle(info)
      for start,end in zip(range(0, len(info), FLAGS.batch_size), range(FLAGS.batch_size, len(info), FLAGS.batch_size)):
        feature_inputs, batch_decoder_inputs, batch_weights = model.get_batch(feature, sentence, info[start:end], FLAGS.batch_size)
        step_loss = model.step(sess, feature_inputs, batch_decoder_inputs, batch_weights, False, FLAGS.batch_size)
        current_step += 1
        if current_step % FLAGS.steps_per_checkpoint == 0:
          print ("global step %d - loss %.3f" % (model.global_step.eval(), step_loss))
          checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'ckpt')
          model.saver.save(sess, checkpoint_path, global_step=model.global_step)
          sys.stdout.flush()

def evaluate():
  from pycocoevalcap.bleu.bleu import Bleu
  from pycocoevalcap.rouge.rouge import Rouge
  from pycocoevalcap.cider.cider import Cider
  from pycocoevalcap.meteor.meteor import Meteor
  from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

  with open('data/validate-15.pkl', 'rb') as f:
    data = cPickle.load(f)
  scorers = [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]), (Meteor(),"METEOR"), (Rouge(), "ROUGE_L"), (Cider(), "CIDEr")]
  vocab, re_vocab = data_utils.initialize_vocabulary('data/vocab.txt')
  GTS = {}
  RES = {}
  batch_size = 1

  with tf.Session() as sess:
    model = Seq2Seq(FLAGS.feature_size, FLAGS.vocab_size, FLAGS.learning_rate, FLAGS.max_gradient_norm, FLAGS.encoder_max_sequence_length, FLAGS.encoder_size, FLAGS.encoder_num_layers, FLAGS.encoder_use_lstm, FLAGS.decoder_max_sentence_length, FLAGS.decoder_size, FLAGS.decoder_num_layers, FLAGS.decoder_use_lstm, forward_only=True)
    step = 0
    while True:
      step += FLAGS.steps_per_checkpoint
      ckpt_path = os.path.join(FLAGS.checkpoint_dir,'ckpt-%d'%step)
      if os.path.isfile(ckpt_path):
        model.saver.restore(sess, ckpt_path)
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
        print('STEP: %d'%step)
        for scorer, method in scorers:
          score, scores = scorer.compute_score(GTS, RES)
          if isinstance(method, list):
            for k, v in zip(method, score):
              print("%s:\t%s"%(k, v))
          else:
            print("%s:\t%s"%(method, score))
        sys.stdout.flush()
      else:
        break

if __name__ == "__main__":
#  train()
  evaluate()
