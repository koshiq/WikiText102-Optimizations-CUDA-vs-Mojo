"""
CUDA-Optimized Transformer Model for WikiText-102
Uses custom CUDA kernels for GEMM, LayerNorm, and Softmax operations
Optimized for NVIDIA RTX 4070 (Ada Lovelace)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import custom CUDA operations
try:
    from custom_ops import CustomLayerNorm, CustomLinear, CustomSoftmax, use_custom_ops # type: ignore
    CUDA_OPS_AVAILABLE = use_custom_ops()

    if CUDA_OPS_AVAILABLE:
        print("Using Custom CUDA Kernels (RTX 4070 Optimized)")
        # Use custom implementations
        LayerNorm = CustomLayerNorm
        Linear = CustomLinear
        Softmax = CustomSoftmax

        # Wrap C++ custom modules so their parameters are visible to PyTorch
        class WrappedCustomLinear(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.inner = CustomLinear(in_features, out_features)
                # expose weight and bias as nn.Parameter that reference the same tensors
                # create Parameters from the C++ tensors and then assign them back
                # so both Python and C++ see the same storage
                self.weight = nn.Parameter(self.inner.weight)
                self.bias = nn.Parameter(self.inner.bias)
                try:
                    self.inner.weight = self.weight
                    self.inner.bias = self.bias
                except Exception:
                    pass

            @property
            def in_features(self):
                try:
                    return int(self.inner.in_features)
                except Exception:
                    return int(self.weight.size(1))

            @property
            def out_features(self):
                try:
                    return int(self.inner.out_features)
                except Exception:
                    return int(self.weight.size(0))

            def forward(self, input):
                # Accept 3D inputs [seq_len, batch, features] by flattening
                if input.dim() == 3:
                    seq_len, batch, features = input.size()
                    inp = input.contiguous().view(seq_len * batch, features)
                    out = self.inner(inp)
                    out_features = out.size(-1)
                    return out.view(seq_len, batch, out_features)
                return self.inner(input)

            def to(self, *args, **kwargs):
                # move python-side parameters first
                super().to(*args, **kwargs)
                # After moving Python-side parameters, ensure C++ inner references the same tensors
                try:
                    self.inner.weight = self.weight
                    self.inner.bias = self.bias
                except Exception:
                    # fallback: call inner.cuda()/cpu() if direct assignment fails
                    pass
                # ensure inner tensors are on the same device as the Python params
                if torch.cuda.is_available():
                    try:
                        self.inner.cuda()
                    except Exception:
                        pass
                else:
                    try:
                        self.inner.cpu()
                    except Exception:
                        pass
                return self

        class WrappedCustomLayerNorm(nn.Module):
            def __init__(self, normalized_shape, eps=1e-5):
                super().__init__()
                self.inner = CustomLayerNorm(normalized_shape, eps)
                self.weight = nn.Parameter(self.inner.weight)
                self.bias = nn.Parameter(self.inner.bias)
                try:
                    self.inner.weight = self.weight
                    self.inner.bias = self.bias
                except Exception:
                    pass

            def forward(self, input):
                return self.inner(input)

            def to(self, *args, **kwargs):
                super().to(*args, **kwargs)
                try:
                    self.inner.weight = self.weight
                    self.inner.bias = self.bias
                except Exception:
                    pass
                if torch.cuda.is_available():
                    try:
                        self.inner.cuda()
                    except Exception:
                        pass
                else:
                    try:
                        self.inner.cpu()
                    except Exception:
                        pass
                return self

        # Use the wrapped classes so parameters are visible to optimizers and counts
        Linear = WrappedCustomLinear
        LayerNorm = WrappedCustomLayerNorm
    else:
        print("Warning: Custom CUDA ops not loaded, using PyTorch defaults")
        LayerNorm = nn.LayerNorm
        Linear = nn.Linear
        Softmax = nn.Softmax
except ImportError as e:
    print(f"Could not import custom CUDA ops: {e}")
    print("   Using PyTorch defaults. To enable custom ops:")
    print("   cd cuda && python setup.py install")
    CUDA_OPS_AVAILABLE = False
    LayerNorm = nn.LayerNorm
    Linear = nn.Linear
    Softmax = nn.Softmax


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

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
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CustomTransformerEncoderLayer(nn.Module):
    """
    Custom Transformer Encoder Layer using optimized CUDA kernels
    Replaces PyTorch's TransformerEncoderLayer with custom LayerNorm and Linear
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CustomTransformerEncoderLayer, self).__init__()

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)

        # Feedforward network with custom Linear layers
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        # Custom LayerNorm layers
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: [seq_len, batch, d_model]
            src_mask: [seq_len, seq_len]
            src_key_padding_mask: [batch, seq_len]
        """
        # Self-attention block
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)  # Using custom LayerNorm

        # Feedforward block
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))  # Using custom Linear
        src = src + self.dropout2(src2)
        src = self.norm2(src)  # Using custom LayerNorm

        return src


class CustomTransformerEncoder(nn.Module):
    """
    Custom Transformer Encoder using optimized encoder layers
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(CustomTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(
                encoder_layer.linear1.in_features,
                encoder_layer.self_attn.num_heads,
                encoder_layer.linear1.out_features,
                encoder_layer.dropout.p
            ) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        Args:
            src: [seq_len, batch, d_model]
            mask: [seq_len, seq_len]
        """
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerModel(nn.Module):
    """
    CUDA-Optimized Transformer Language Model

    This is a drop-in replacement for the original TransformerModel
    that uses custom CUDA kernels when available.

    Custom optimizations:
    - CustomLinear: Optimized GEMM kernels for matrix multiplication
    - CustomLayerNorm: Fused LayerNorm with Welford's algorithm
    - CustomSoftmax: Numerically stable softmax with warp-level reductions
    """

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()

        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        # Embedding layer (standard)
        self.input_emb = nn.Embedding(ntoken, ninp)
        self.ninp = ninp

        # Output decoder with custom Linear layer
        self.decoder = Linear(ninp, ntoken)

        # Build custom transformer encoder
        encoder_layer_template = CustomTransformerEncoderLayer(
            d_model=ninp,
            nhead=nhead,
            dim_feedforward=nhid,
            dropout=dropout
        )

        encoder_norm = LayerNorm(ninp)  # Custom LayerNorm

        self.encoder = CustomTransformerEncoder(
            encoder_layer_template,
            num_layers=nlayers,
            norm=encoder_norm
        )

        # Initialize weights
        self.init_weights()

        # Print info about custom ops
        if CUDA_OPS_AVAILABLE:
            print(f"Model initialized with custom CUDA kernels:")
            print(f"   - Linear layers: {type(self.decoder).__name__}")
            print(f"   - LayerNorm: {type(self.encoder.norm).__name__}")
            param_count = sum(p.numel() for p in self.parameters())
            print(f"   - Total parameters: {param_count:,}")

    def _generate_square_subsequent_mask(self, sz):
        """Generate causal mask for autoregressive language modeling"""
        return torch.log(torch.tril(torch.ones(sz, sz)))

    def init_weights(self):
        """Initialize model weights"""
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        """
        Forward pass through the transformer

        Args:
            src: [seq_len, batch] - Input token indices
            has_mask: bool - Whether to use causal masking

        Returns:
            output: [seq_len, batch, ntoken] - Log probabilities
        """
        # Generate or update attention mask
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        # Input embedding with scaling
        src = self.input_emb(src) * math.sqrt(self.ninp)

        # Add positional encoding
        src = self.pos_encoder(src)

        # Pass through transformer encoder (uses custom kernels)
        output = self.encoder(src, mask=self.src_mask)

        # Output projection (uses custom Linear layer)
        output = self.decoder(output)

        # Log softmax (PyTorch's is already optimized, but could use custom if needed)
        return F.log_softmax(output, dim=-1)


# ============================================================================
# Utility Functions
# ============================================================================

def get_model_info():
    """Get information about custom CUDA ops availability"""
    return {
        'cuda_ops_available': CUDA_OPS_AVAILABLE,
        'layer_norm_type': LayerNorm.__name__,
        'linear_type': Linear.__name__,
        'softmax_type': Softmax.__name__,
    }


def print_model_info():
    """Print information about the model configuration"""
    info = get_model_info()
    print("\n" + "="*60)
    print("Model Configuration")
    print("="*60)
    print(f"Custom CUDA Ops: {'Enabled' if info['cuda_ops_available'] else 'Disabled'}")
    print(f"LayerNorm Type:  {info['layer_norm_type']}")
    print(f"Linear Type:     {info['linear_type']}")
    print(f"Softmax Type:    {info['softmax_type']}")
    print("="*60 + "\n")


# Print info when module is imported
if __name__ != '__main__':
    print_model_info()


def move_custom_modules_to_device(module: nn.Module, device: torch.device):
    """Recursively find attributes that are custom ops (CustomLinear/CustomLayerNorm/CustomSoftmax)
    and call their .cuda()/.cpu() methods so their internal tensors are on the correct device.
    This is necessary because the custom pybind classes are not native Python nn.Module subclasses
    and therefore are not moved automatically by `nn.Module.to()`.
    """
    custom_types = ()
    try:
        custom_types = (CustomLinear, CustomLayerNorm, CustomSoftmax)
    except NameError:
        return

    # iterate all registered submodules and look for an `.inner` attribute
    # which holds the underlying pybind custom op, then move it
    for sub in module.modules():
        inner = getattr(sub, 'inner', None)
        if inner is not None:
            try:
                if device.type == 'cuda':
                    inner.cuda()
                else:
                    inner.cpu()
            except Exception:
                pass
