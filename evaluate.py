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

import slim.ops as ops

import mean_rnn

def create_model(session, feature_size, vocab_size, max_sentence_length, size, num_layers, use_lstm, forward_only, step=200):
  model = mean_rnn.MeanRNN(feature_size, vocab_size, max_sentence_length, size, num_layers, use_lstm, forward_only)
  model.saver.restore(session, 'CheckPoint/.ckpt-%d'%step)
  return model

def evaluate(feature_size, vocab_size, max_sentence_length, size, num_layers, use_lstm, steps_per_checkpoint=200):
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
    model = mean_rnn.MeanRNN(feature_size, vocab_size, max_sentence_length, size, num_layers, use_lstm, True)
    step = 0
    while True:
      step += steps_per_checkpoint
      ckpt_path = 'CheckPoint-0.0001/ckpt-%d'%step
      if os.path.isfile(ckpt_path):
        model.saver.restore(sess, ckpt_path)
        for idx, (feature, caption) in enumerate(data):
          feature_inputs, batch_decoder_inputs, batch_weights = model.get_batch([feature], [([0], 0)], [0], 1)
          output_logits = model.step(sess, feature_inputs, batch_decoder_inputs, batch_weights, forward_only=True, batch_size=1)
          outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
          if data_utils.EOS_ID in outputs:
            outputs = outputs[:outputs.index(data_utils.EOS_ID)]
          sentence = " ".join([tf.compat.as_str(re_vocab[output]) for output in outputs])
#          print ("%s: %s"%(sentence, caption[9]))
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
  evaluate(feature_size=2048, vocab_size=8110, max_sentence_length=40, size=1024, num_layers=4, use_lstm=False)

