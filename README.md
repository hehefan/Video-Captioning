# Video-Captioning
## Sequence to sequence
| Directory              | Encoder                   | Attention | Decoder                   | MSVD(METEOR) |
| ---------------------- |:-------------------------:|:---------:|:-------------------------:|:------------:| 
| 1024_gru_1024_gru      | 1024 GRU, 1 layer, dp 0.0 | No        | 1024 GRU, 1 layer, dp 0.0 |29.9, bs 31.2 |
| 1024_gru_attn_1024_gru | 1024 GRU, 1 layer, dp 0.0 | Yes       | 1024 GRU, 1 layer, dp 0.0 |31.4, bs 31.7 |

