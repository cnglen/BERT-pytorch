#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MultiHead(Q, K, V)"""

import torch.nn as nn
from .single import Attention


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.

    args:
      head_count: number of heads
      d_model: dim of model (key/query/value)
      dropout:

    """

    def __init__(self, head_count: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % head_count == 0

        self.dim_per_head = d_model // head_count
        self.head_count = head_count
        self.dropout = dropout

        self.linear_keys = nn.Linear(d_model, self.dim_per_head * self.head_count)
        self.linear_values = nn.Linear(d_model, self.dim_per_head * self.head_count)
        self.linear_querys = nn.Linear(d_model, self.dim_per_head * self.head_count)
        self.linear_output = nn.Linear(self.dim_per_head * self.head_count, d_model)

        self.attention = Attention()

    def forward(self, query, key, value, mask=None):
        """
        MultiHead:
        - 映射到子空间
        - attention
        - 拼接
        - 映射到目标空间

        Args:
          query: (B, Ty, d)
          key: (B, Tx, d)
          value: (B, Tx, d)

        Returns:
          query: (B, Ty, d)
          p_attn: attention weight, (B, Ty, Tx)

        """
        batch_size = query.size(0)

        # linear: 映射到子空间
        # 实现细节:
        #   方案I: 分别映射到h个子空间, 每个子空间的维度是d/h (串行）
        #     - for i in range(h): (B, T, d) --linear--> (B, T, d/h)
        #   方案II(*): 映射到一个维度为d的大空间(一个大矩阵乘法，并行)，然后再分割成h个子空间
        #     - (B, T, d) --linear--> (B, T, d) --view--> (B, T, h, d/h) --transpose--> (B, h, T, d/h)
        query, key, value = self.linear_querys(query), self.linear_keys(key), self.linear_values(value)
        query = query.view(batch_size, -1, self.head_count, self.dim_per_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.head_count, self.dim_per_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.head_count, self.dim_per_head).transpose(1, 2)

        # attention
        x, p_attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # "Concat": (B, h, T, d/h) --transpose--> (B, T, h, d/h) --concat--> (B, T, d)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        # linear
        x = self.linear_output(x)

        return x, p_attn
