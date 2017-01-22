import os
import sys
import re
from config import *
data_dir = FLAGS.data_dir
# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(_WORD_SPLIT.split(space_separated_fragment))
  return [w for w in words if w]

def create_vocabulary(data, max_frequency, normalize_digits=True):
  if not os.path.isfile(os.path.join(data_dir, 'vocab.txt')):
    print("Creating vocabulary vocab.txt.")
    vocab = {}
    for _, sen in data:
      tokens = basic_tokenizer(sen)
      for w in tokens:
        word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
        vocab[word] = vocab.get(word, 0) + 1
    with open(os.path.join(data_dir, 'vocab.txt'), 'w') as vocab_file:
      for w in _START_VOCAB:
        vocab_file.write(w + b"\n")
      for w, f in vocab.iteritems():
        if f >= max_frequency:
          vocab_file.write(w + b"\n")

def initialize_vocabulary():
  if os.path.isfile(os.path.join(data_dir, 'vocab.txt')):
    rev_vocab = []
    with open(os.path.join(data_dir, 'vocab.txt'), 'r') as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def sentence_to_token_ids(sentence, vocabulary, normalize_digits=True):
  words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]

def data_to_token_ids(data, normalize_digits=True):
  vocab, _ = initialize_vocabulary()
  rst = []
  for vid, sen in data:
    token_ids = sentence_to_token_ids(sen, vocab,normalize_digits)
    rst.append((vid, token_ids))
  return rst

if __name__ == '__main__':
  import cPickle
  mapping = {}
  with open('youtube_mapping.txt','r') as f:
    for line in f:
      arr = line.strip().split()
      mapping[arr[0].strip()] = arr[1].strip()
  data = []
  with open('video-descriptions.csv', 'r') as f:
    for line in f:
      arr = line.strip().split(',')
      vid = '%s_%s_%s'%(arr[0], arr[1], arr[2])
      if mapping.has_key(vid):
        vid = mapping[vid]
        sen = arr[7].strip('.').lower() 
        data.append((vid, sen))
  create_vocabulary(data, 2)
  rst = data_to_token_ids(data)
  train = []
  for vid, sen in rst:
    i = int(vid.strip('vid'))
    if i <= 1300:
      train.append((vid, sen))
  with open('caption.train', 'wb') as f:
    cPickle.dump(train, f)
  test = {}
  for vid, sen in data:
    i = int(vid.strip('vid'))
    if i > 1300:
      if test.has_key(vid):
        test[vid].append(sen)
      else:
        test[vid] = [sen]
  with open('caption.test', 'wb') as f:
    cPickle.dump(test, f)
