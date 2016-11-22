# [([6791, 5754, 6445, 546, 1238, 5501, 3630, 783, 3083, 6791, 376, 4887], u'video2960')]
import cPickle
import json
with open('/home/hehe/dataset/VideoToText/MSR-VTT/train_val.json', 'r') as f:
  json_data = json.load(f)
  sentences = json_data['sentences']
  video_id = [0] * 140200
  for sentence in sentences:
    vid = sentence['video_id'].strip('video')
    video_id[sentence['sen_id']] = int(vid)

sentences = []
with open('ids.txt', 'r') as f:
  idx = 0
  for line in f:
    sen = []
    for i in line.split():
      sen.append(int(i))
    sentences.append((sen, video_id[idx]))
    idx += 1
print sentences[1000]
with open('sentences.pkl', 'wb') as f:
  cPickle.dump(sentences, f)
