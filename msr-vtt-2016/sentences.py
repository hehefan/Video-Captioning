import json
import os

with open('/home/hehe/dataset/VideoToText/MSR-VTT/train_val.json', 'r') as f:
  json_data = json.load(f)
  for k, v in json_data.iteritems():
    print "%s: %s"%(k, type(v))
  sentences = json_data['sentences']
  sens = [''] * 140200
  for sentence in sentences:
    sens[sentence['sen_id']] = sentence['caption'].encode('utf-8')
with open('sentences.txt', 'w') as f:
  for i in range(140200):
    f.write(sens[i] + b'\n')
