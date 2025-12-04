"""
MAX-accelerated version of the WikiText2 Transformer model.
This replaces key operations (GEMM, Softmax, LayerNorm) with MAX Graph ops.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import max.torch
from max.graph import ops


# Define MAX Graph operations
@max.torch.graph_op
def max_matmul(A, B):
    """MAX-accelerated matrix multiplication (GEMM)"""
    return ops.matmul(A, B)


@max.torch.graph_op
def max_softmax(x):
    """MAX-accelerated softmax"""
    return ops.softmax(x, axis=-1)


@max.torch.graph_op
def max_log_softmax(x):
    """MAX-accelerated log softmax"""
    return ops.logsoftmax(x, axis=-1)


@max.torch.graph_op
def max_layer_norm(x, weight, bias):
    """MAX-accelerated layer normalization"""
    return ops.layer_norm(x, weight, bias, epsilon=1e-5)


class MaxLinear(nn.Module):
    """Linear layer using MAX Graph matmul"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        # Allocate output tensor
        out_shape = list(x.shape[:-1]) + [self.weight.shape[0]]
        output = torch.empty(out_shape, dtype=x.dtype, device=x.device)

        # Reshape input for batched matmul if needed
        original_shape = x.shape
        if x.dim() > 2:
            x_2d = x.reshape(-1, x.shape[-1])
        else:
            x_2d = x

        # MAX matmul - detach weight to avoid gradient tracking
        output_2d = torch.empty(x_2d.shape[0], self.weight.shape[0], dtype=x.dtype, device=x.device)
        max_matmul(output_2d, x_2d, self.weight.t().detach())

        # Reshape back
        if len(original_shape) > 2:
            output = output_2d.reshape(out_shape)
        else:
            output = output_2d

        if self.bias is not None:
            output = output + self.bias.detach()

        return output


class MaxLayerNorm(nn.Module):
    """LayerNorm using MAX Graph"""
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        output = torch.empty_like(x)
        max_layer_norm(output, x, self.weight.detach(), self.bias.detach())
        return output


class PositionalEncoding(nn.Module):
    """Standard positional encoding (no MAX acceleration needed)"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModelMAX(nn.Module):
    """
    Transformer model with MAX-accelerated operations.

    Key accelerations:
    - Linear layers use MAX matmul (GEMM)
    - Final log_softmax uses MAX
    - LayerNorm uses MAX (via replacing PyTorch's implementation)
    """
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5, use_max=True):
        super().__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.use_max = use_max

        # Embedding (no MAX acceleration needed)
        self.input_emb = nn.Embedding(ntoken, ninp)
        self.ninp = ninp

        # Create standard PyTorch transformer, then replace Linear/LayerNorm
        if use_max:
            # Use PyTorch Transformer but we'll replace internals
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout),
                nlayers
            )
            # Replace Linear and LayerNorm layers with MAX versions
            self._replace_with_max_ops()

            # Decoder with MAX
            self.decoder = MaxLinear(ninp, ntoken)
        else:
            # Standard PyTorch
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout),
                nlayers
            )
            self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _replace_with_max_ops(self):
        """Replace Linear and LayerNorm in transformer with MAX versions"""
        for layer in self.encoder.layers:
            # Replace linear layers in attention
            if hasattr(layer.self_attn, 'in_proj_weight'):
                # Multi-head attention linear layers
                # Note: This is complex due to PyTorch's fused implementation
                # For now, we keep attention as-is and only replace FFN and LayerNorm
                pass

            # Replace FFN linear layers
            old_linear1 = layer.linear1
            old_linear2 = layer.linear2

            layer.linear1 = MaxLinear(old_linear1.in_features, old_linear1.out_features)
            layer.linear1.weight.data = old_linear1.weight.data
            layer.linear1.bias.data = old_linear1.bias.data

            layer.linear2 = MaxLinear(old_linear2.in_features, old_linear2.out_features)
            layer.linear2.weight.data = old_linear2.weight.data
            layer.linear2.bias.data = old_linear2.bias.data

            # Replace LayerNorm with MAX version
            old_norm1 = layer.norm1
            old_norm2 = layer.norm2

            layer.norm1 = MaxLayerNorm(old_norm1.normalized_shape[0])
            layer.norm1.weight.data = old_norm1.weight.data
            layer.norm1.bias.data = old_norm1.bias.data

            layer.norm2 = MaxLayerNorm(old_norm2.normalized_shape[0])
            layer.norm2.weight.data = old_norm2.weight.data
            layer.norm2.bias.data = old_norm2.bias.data

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz, sz)))

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        if self.use_max:
            nn.init.zeros_(self.decoder.bias)
            nn.init.uniform_(self.decoder.weight, -initrange, initrange)
        else:
            nn.init.zeros_(self.decoder.bias)
            nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.input_emb(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.encoder(src, mask=self.src_mask)
        output = self.decoder(output)

        # Use MAX log_softmax if enabled
        if self.use_max:
            result = torch.empty_like(output)
            max_log_softmax(result, output)
            return result
        else:
            return F.log_softmax(output, dim=-1)
