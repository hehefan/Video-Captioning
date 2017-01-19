# Video-Captioning
## Preprocessed Dataset
### MSVD
1. [feature.train](https://drive.google.com/uc?export=download&id=0B7NctsDC2gmLOF8xYTZPcFoySHc): cPickle file, {'vid1': [numpy.array(2048 frame feature), ..., numpy.array(2048 frame feature)], ...}
2. [feature.test](https://drive.google.com/uc?export=download&id=0B7NctsDC2gmLMWxBYWhXVUNZSFE): cPickle file, {'vid1301': [numpy.array(2048 frame feature), ..., numpy.array(2048 frame feature)], ...}
3. [caption.train](https://drive.google.com/uc?export=download&id=0B7NctsDC2gmLWjVwMG51UElKQWs): cPickle file, [('vid1', [word_id, ..., word_id]), ...]
4. [caption.test](https://drive.google.com/uc?export=download&id=0B7NctsDC2gmLTU9WaTNwM2VvRVk): cPickle file, {'vid1301': [sentence, ..., sentence]}
5. [vocab.txt](https://drive.google.com/uc?export=download&id=0B7NctsDC2gmLWE1MVmpKTm5yVEk): text file, one word per line.

## Sequence to sequence
| Directory              | Encoder                   | Attention | Decoder                   | MSVD(METEOR)  |
| ---------------------- |:-------------------------:|:---------:|:-------------------------:|:-------------:| 
| 1024_gru_1024_gru      | 1024 GRU, 1 layer, dp 0.0 | No        | 1024 GRU, 1 layer, dp 0.0 |29.9, b3s 31.2 |
| 1024_gru_attn_1024_gru | 1024 GRU, 1 layer, dp 0.0 | Yes       | 1024 GRU, 1 layer, dp 0.0 |31.4, b3s 32.2 |
| 1024_gru_attn_1024_gru | 1024 GRU, 1 layer, dp 0.3 | Yes       | 1024 GRU, 1 layer, dp 0.3 |31.8, b3s  |
| 1024_gru_attn_1024_gru | 1024 GRU, 3 layer, dp 0.0 | Yes       | 1024 GRU, 3 layer, dp 0.0 |  |

