# Video-Captioning
## Sequence to sequence
| Directory              | Encoder            | Attention | Decoder            | MSVD(METEOR)           |
| ---------------------- |:------------------:|:---------:|:------------------:|:----------------------:| 
| 1024_gru_1024_gru      | GRU, 1024, 1 layer, dp 0.0 | No        | GRU, 1024, 1 layer |0.299071, beam 0.312598 |
| 1024_gru_attn_1024_gru | GRU, 1024, 1 layer | Yes       | GRU, 1024, 1 layer |0.314424, beam 0.317402 |

