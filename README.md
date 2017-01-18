# Video-Captioning
## Sequence to sequence
| Directory              | Encoder                   | Attention | Decoder                   | MSVD(METEOR)           |
| ---------------------- |:-------------------------:|:---------:|:-------------------------:|:----------------------:| 
| 1024_gru_1024_gru      | 1024 GRU, 1 layer, dp 0.0 | No        | 1024 GRU, 1 layer, dp 0.0 |0.299071, bs 0.312598   |
| 1024_gru_attn_1024_gru | 1024 GRU, 1 layer, dp 0.0 | Yes       | 1024 GRU, 1 layer, dp 0.0 |0.314424, bs 0.317402   |

