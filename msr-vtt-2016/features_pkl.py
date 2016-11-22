import cPickle 
import sys
import os

DIR = '/home/hehe/dataset/VideoToText/MSR-VTT/features'
feature = [[]] * 10000
for i in range(10000):
  curDIR = os.path.join(DIR, 'video%d.mp4'%i)
  for j in range(1,1000):
    pkl = os.path.join(curDIR, '%04d.pkl'%j)
    if os.path.isfile(pkl):
      with open(pkl, 'rb') as f:
        frame = cPickle.load(f)
        feature[i].append(frame)
    else:
      break

with open('feature.pkl', 'wb') as f:
  cPickle.dump(feature, f)
