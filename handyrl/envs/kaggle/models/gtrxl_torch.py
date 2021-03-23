"""
based on https://github.com/alantess/gtrxl-torch/blob/main/gtrxl_torch/gtrxl_torch.py
"""

from typing import Optional

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TEL(TransformerEncoderLayer):
    '''
    Recreate the transfomer layers done in the following paper
    https://arxiv.org/pdf/1910.06764.pdf
    '''

    def __init__(self, d_model, nhead, n_layers=1, dim_feedforward=256, activation="relu", dropout=0):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        # 2 GRUs are needed - 1 for the beginning / 1 at the end
        self.gru_1 = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.gru_2 = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=n_layers, batch_first=True)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        h = (src).sum(dim=1).unsqueeze(dim=0)
        src = self.norm1(src)
        out = self.self_attn(src, src, src, attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)[0]

        out, h = self.gru_1(out, h)
        out = self.norm2(out)
        out = self.activation(self.linear1(out))
        out = self.activation(self.linear2(out))
        out, h = self.gru_2(out, h)
        return out


class GTrXL(nn.Module):
    '''
    Implementation of transfomer model using GRUs
    '''

    def __init__(self, d_model, nheads, transformer_layers, hidden_dims=256, n_layers=1, activation='relu'):
        super(GTrXL, self).__init__()
        # Module layers
        encoded = TEL(d_model, nheads, n_layers, dim_feedforward=hidden_dims, activation=activation)
        self.transfomer = TransformerEncoder(encoded, transformer_layers)

    def forward(self, x):
        x = self.transfomer(x)
        return x
