from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import data_utils
import cPickle

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

from config import *
from model import Seq2Seq
from caption_generator import CaptionGenerator

def evaluate():
  with open (os.path.join(FLAGS.data_dir, 'feature.test'), 'rb') as f:
    feature = cPickle.load(f)
  with open(os.path.join(FLAGS.data_dir, 'caption.test'), 'rb') as f:
    sentence = cPickle.load(f)

  scorers = [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]), (Meteor(),"METEOR"), (Rouge(), "ROUGE_L"), (Cider(), "CIDEr")]
  vocab, re_vocab = data_utils.initialize_vocabulary()
  GTS = {}
  RES = {}
  batch_size = 1
  max_meteor = 0

  with tf.Session() as sess:
    model = Seq2Seq(FLAGS.num_units, FLAGS.use_lstm, FLAGS.encoder_max_sequence_length, 1, FLAGS.feature_size, FLAGS.vocab_size, FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, FLAGS.max_gradient_norm, forward_only=True)
    step = 0
    while True:
      step += FLAGS.steps_per_checkpoint
      ckpt_path = os.path.join(FLAGS.checkpoint_dir,'ckpt-%d'%step)
      if os.path.isfile(ckpt_path+'.meta'):
        model.saver.restore(sess, ckpt_path)
        cg = CaptionGenerator(model=model,start_id=data_utils.GO_ID,end_id=data_utils.EOS_ID, beam_size=3, max_caption_length=FLAGS.decoder_max_sentence_length, length_normalization_factor=0.0)
        for vid, _ in feature.iteritems():
          feature_inputs, batch_decoder_inputs, batch_weights = model.get_batch(feature, [(vid, [0])])
          sen = cg.beam_search(sess, feature_inputs) 
          sen = " ".join([tf.compat.as_str(re_vocab[w]) for w in sen])
          print ("%s: %s"%(sen, sentence[vid][9]))
          GTS[vid] = sentence[vid]
          RES[vid] = [sen]
        print('STEP: %d'%step)
        for scorer, method in scorers:
          score, scores = scorer.compute_score(GTS, RES)
          if method == "METEOR" and score > max_meteor:
            max_meteor = score
          if isinstance(method, list):
            for k, v in zip(method, score):
              print("%s:\t%f"%(k, v))
          else:
            print("%s:\t%f"%(method, score))
        sys.stdout.flush()
      else:
        break
  print("Max METEOR:\t%f"%max_meteor)

if __name__ == "__main__":
  evaluate()