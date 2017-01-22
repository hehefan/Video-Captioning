# Video-Captioning
## 1. Preprocessed Dataset
### 1.1. MSVD
1. [feature.train](https://drive.google.com/uc?export=download&id=0B7NctsDC2gmLOF8xYTZPcFoySHc): cPickle file, {'vid1': [numpy.array(2048 frame feature), ..., numpy.array(2048 frame feature)], ...}
2. [feature.test](https://drive.google.com/uc?export=download&id=0B7NctsDC2gmLMWxBYWhXVUNZSFE): cPickle file, {'vid1301': [numpy.array(2048 frame feature), ..., numpy.array(2048 frame feature)], ...}
3. [caption.train](https://drive.google.com/uc?export=download&id=0B7NctsDC2gmLWjVwMG51UElKQWs): cPickle file, [('vid1', [word_id, ..., word_id]), ...]
4. [caption.test](https://drive.google.com/uc?export=download&id=0B7NctsDC2gmLTU9WaTNwM2VvRVk): cPickle file, {'vid1301': [sentence, ..., sentence]}
5. [video.length](0B7NctsDC2gmLVDRhQWpHbGpMQ0U): cPickle file, {'vid1':1, ...}
6. [vocab.txt](https://drive.google.com/uc?export=download&id=0B7NctsDC2gmLWE1MVmpKTm5yVEk): text file, one word per line.

## 2. Basic sequence to sequence
| No | Encoder                       | Attention | Decoder                       | MSVD(METEOR)  |
| -- |:-----------------------------:|:---------:|:-----------------------------:|:-------------:| 
| 1  | 1024 GRU, 1 layer, dp 0.0, 30 | No        | 1024 GRU, 1 layer, dp 0.0, 15 |29.9, b3s 31.2 |
| 2  | 1024 GRU, 1 layer, dp 0.0, 30 | Yes       | 1024 GRU, 1 layer, dp 0.0, 15 |31.4, b3s 32.2 |
| 3  | 1024 GRU, 1 layer, dp 0.3, 30 | Yes       | 1024 GRU, 1 layer, dp 0.3, 15 |31.8, b3s 32.2 |
| 4  | 1024 GRU, 3 layer, dp 0.0, 30 | Yes       | 1024 GRU, 3 layer, dp 0.0, 15 |30.7, b3s 31.4 |
| 5  | 1024 GRU, 2 layer, dp 0.3, 15 | Yes       | 1024 GRU, 2 layer, dp 0.3, 15 |31.8, b3s 32.7 |

