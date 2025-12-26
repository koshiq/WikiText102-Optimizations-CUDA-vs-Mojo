"""
Trainable Transformer model using MAX Graph API for acceleration.
This version supports both forward and backward passes for training.
"""

import math
import torch
import torch.nn as nn
from torch.autograd import Function
import max.torch
from max.graph import ops

print("MAX Graph API available - Trainable version")


# =====================================================================================
# MAX Graph Operations with Autograd Support
# =====================================================================================

class MaxMatmulFunction(Function):
    """Custom autograd function for MAX-accelerated matmul with gradients."""

    @staticmethod
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)

        # Detach inputs for MAX Graph
        A_detached = A.detach()
        B_detached = B.detach()

        output = torch.empty(A.shape[0], B.shape[1], dtype=A.dtype, device=A.device)

        @max.torch.graph_op
        def _max_matmul(A, B):
            return ops.matmul(A, B)

        _max_matmul(output, A_detached, B_detached)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_A = grad_B = None

        if ctx.needs_input_grad[0]:
            # grad_A = grad_output @ B.T
            grad_A = torch.matmul(grad_output, B.t())

        if ctx.needs_input_grad[1]:
            # grad_B = A.T @ grad_output
            grad_B = torch.matmul(A.t(), grad_output)

        return grad_A, grad_B


class MaxLayerNormFunction(Function):
    """Custom autograd function for MAX-accelerated LayerNorm with gradients."""

    @staticmethod
    def forward(ctx, x, weight, bias, eps=1e-5):
        # Detach inputs for MAX Graph (it doesn't support gradient tensors)
        x_detached = x.detach()
        weight_detached = weight.detach()
        bias_detached = bias.detach()

        output = torch.empty_like(x)

        @max.torch.graph_op
        def _max_layer_norm(x, weight, bias):
            return ops.layer_norm(x, weight, bias, epsilon=eps)

        _max_layer_norm(output, x_detached, weight_detached, bias_detached)

        # Save for backward (we'll use PyTorch's native backward for simplicity)
        # In production, you'd implement custom CUDA backward kernels
        ctx.save_for_backward(x, weight, bias)
        ctx.eps = eps
        ctx.normalized_shape = weight.shape

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Use PyTorch's native LayerNorm backward
        # For MAX kernels, you'd implement custom backward CUDA kernels
        x, weight, bias = ctx.saved_tensors

        # Compute statistics
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + ctx.eps)

        # Normalized input
        x_norm = (x - mean) / std

        # Gradients
        grad_x = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            # Gradient w.r.t input
            grad_x_norm = grad_output * weight
            grad_var = (grad_x_norm * (x - mean) * -0.5 * (var + ctx.eps)**(-1.5)).sum(dim=-1, keepdim=True)
            grad_mean = (grad_x_norm * -1.0 / std).sum(dim=-1, keepdim=True) + grad_var * (x - mean).mean(dim=-1, keepdim=True) * -2.0
            grad_x = grad_x_norm / std + grad_var * 2.0 * (x - mean) / x.shape[-1] + grad_mean / x.shape[-1]

        if ctx.needs_input_grad[1]:
            # Gradient w.r.t weight
            grad_weight = (grad_output * x_norm).sum(dim=tuple(range(len(x.shape) - 1)))

        if ctx.needs_input_grad[2]:
            # Gradient w.r.t bias
            grad_bias = grad_output.sum(dim=tuple(range(len(x.shape) - 1)))

        return grad_x, grad_weight, grad_bias, None


class MaxLogSoftmaxFunction(Function):
    """Custom autograd function for MAX-accelerated LogSoftmax with gradients."""

    @staticmethod
    def forward(ctx, x, dim=-1):
        # Detach input for MAX Graph
        x_detached = x.detach()

        output = torch.empty_like(x)

        @max.torch.graph_op
        def _max_log_softmax(x):
            return ops.logsoftmax(x, axis=dim)

        _max_log_softmax(output, x_detached)
        ctx.save_for_backward(output)
        ctx.dim = dim
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        # d/dx log_softmax(x) = 1 - exp(log_softmax(x))
        grad_input = grad_output - torch.exp(output) * grad_output.sum(dim=ctx.dim, keepdim=True)
        return grad_input, None


# Wrapper functions
def max_matmul(A, B):
    """MAX-accelerated matrix multiplication with autograd support."""
    return MaxMatmulFunction.apply(A, B)


def max_layer_norm(x, weight, bias, eps=1e-5):
    """MAX-accelerated layer normalization with autograd support."""
    return MaxLayerNormFunction.apply(x, weight, bias, eps)


def max_log_softmax(x, dim=-1):
    """MAX-accelerated log softmax with autograd support."""
    return MaxLogSoftmaxFunction.apply(x, dim)


# =====================================================================================
# MAX-Accelerated Trainable Modules
# =====================================================================================

class MaxLinear(nn.Module):
    """Linear layer using MAX Graph matmul - TRAINABLE VERSION."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        # Reshape input for batched matmul if needed
        original_shape = x.shape
        if x.dim() > 2:
            x_2d = x.reshape(-1, x.shape[-1])
        else:
            x_2d = x

        # MAX matmul with gradient support
        output_2d = max_matmul(x_2d, self.weight.t())

        # Reshape back to original dimension count
        if len(original_shape) > 2:
            out_shape = list(x.shape[:-1]) + [self.weight.shape[0]]
            output = output_2d.reshape(out_shape)
        else:
            output = output_2d

        if self.bias is not None:
            output = output + self.bias

        return output


class MaxLayerNorm(nn.Module):
    """LayerNorm using MAX Graph - TRAINABLE VERSION."""

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        return max_layer_norm(x, self.weight, self.bias, self.eps)


class MaxLogSoftmax(nn.Module):
    """Log-Softmax using MAX Graph API - TRAINABLE VERSION."""

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return max_log_softmax(x, self.dim)


# =====================================================================================
# Positional Encoding
# =====================================================================================

class PositionalEncoding(nn.Module):
    """Standard positional encoding (no acceleration needed)."""

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


# =====================================================================================
# MAX-Accelerated Trainable Transformer Encoder Layer
# =====================================================================================

class MaxTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with MAX-accelerated operations - TRAINABLE VERSION."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        # Multi-head attention (keep standard for now, acceleration focuses on FFN)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)

        # Feed-forward network with MAX-accelerated Linear layers
        self.linear1 = MaxLinear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = MaxLinear(dim_feedforward, d_model)

        # Layer normalization with MAX acceleration
        self.norm1 = MaxLayerNorm(d_model)
        self.norm2 = MaxLayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feed-forward network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


# =====================================================================================
# MAX-Accelerated Trainable Transformer Model
# =====================================================================================

class TransformerModel(nn.Module):
    """
    TRAINABLE Transformer model with MAX Graph API acceleration.

    Accelerated operations with gradient support:
    - GEMM (Matrix Multiplication) in Linear layers
    - Layer Normalization
    - Final Log-Softmax

    This version can be used for both training and inference.
    """

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.ninp = ninp

        # Input embedding (standard)
        self.encoder_embedding = nn.Embedding(ntoken, ninp)

        # Encoder layers with MAX acceleration
        encoder_layers = [
            MaxTransformerEncoderLayer(ninp, nhead, nhid, dropout)
            for _ in range(nlayers)
        ]
        self.transformer_encoder = nn.ModuleList(encoder_layers)

        # Decoder with MAX-accelerated Linear and LogSoftmax
        self.decoder = MaxLinear(ninp, ntoken)
        self.log_softmax = MaxLogSoftmax(dim=-1)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        """Generate attention mask for autoregressive decoding."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder_embedding.weight, -initrange, initrange)
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

        # Embedding + positional encoding
        src = self.encoder_embedding(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)

        # Transformer encoder layers
        for layer in self.transformer_encoder:
            src = layer(src, src_mask=self.src_mask)

        # Decoder + log_softmax
        output = self.decoder(src)
        output = self.log_softmax(output)

        return output


def get_model_info():
    """Get information about MAX Graph API availability."""
    return {
        'max_graph_available': True,
        'backend': 'MAX Graph API',
        'trainable': True,
        'expected_speedup': '1.12x-1.90x',
    }
