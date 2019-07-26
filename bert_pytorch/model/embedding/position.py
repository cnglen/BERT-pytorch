#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""desc"""


import math
import torch.nn as nn
import torch


class PositionalEncoding(nn.Module):
    """
    Implement the Positional Encoding function.

    Position Encoding:
    - position: t表示了offset不同
    - 不同的d in [0, d_model-1], 表示的周期不同，idx_dim越小, 周期越小，频率越高

    args:
      d_model: dim of position embedding (因为和transformer中的word_embedding相加, 故和d_model相等)
      max_position: max position supported, 理论上只要不OOM, 可以无穷大
    """

    def __init__(self, d_model, max_position=5000):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_position, d_model)  # 本质上就是一个计算出来的矩阵(T_max, d_model)
        position = torch.arange(0, max_position).unsqueeze(1).float()  # (T_max, 1)
        div_term = (torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)).exp().unsqueeze(0)  # (1, d_model/2)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)    # (B=1, T_max, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        将embeddding `x`和position embeedding 直接相加

        args:
          x: embedding (B, T, d_model)

        returns:
          (B, T, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x
