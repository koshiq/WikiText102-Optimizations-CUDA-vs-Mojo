import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class PositionalEncoding(nn.Module):
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

class TransformerModel(nn.Transformer):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__(d_model=ninp, nhead=nhead, dim_feedforward=nhid, num_encoder_layers=nlayers)
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        self.input_emb = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz,sz)))

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
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
        return F.log_softmax(output, dim=-1)


class ScriptedTransformerModel(nn.Module):
    """
    TorchScript-compatible transformer model for fair comparison with MAX Graph.

    This version uses torch.jit.script() to enable:
    - Operator fusion (similar to MAX Graph's graph compilation)
    - Kernel selection optimization
    - Reduced Python overhead
    - Graph-level optimizations

    Changes from base TransformerModel:
    - Uses type annotations for TorchScript compatibility
    - Avoids dynamic control flow where possible
    - Pre-generates mask as buffer instead of dynamic creation
    """

    def __init__(self, ntoken: int, ninp: int, nhead: int, nhid: int, nlayers: int, dropout: float = 0.5):
        super(ScriptedTransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.ninp = ninp
        self.ninp_sqrt = math.sqrt(ninp)  # Pre-compute for TorchScript optimization

        # Positional encoding
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        # Embedding layers
        self.input_emb = nn.Embedding(ntoken, ninp)
        self.decoder = nn.Linear(ninp, ntoken)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=ninp,
            nhead=nhead,
            dim_feedforward=nhid,
            dropout=dropout,
            activation='relu',
            batch_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        self.init_weights()

        # Pre-register mask buffer (will be resized if needed)
        # Use empty tensor that will be populated on first forward pass
        self.register_buffer('_mask_cache', torch.empty(0, 0))
        self._cached_seq_len: int = 0

    def init_weights(self) -> None:
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for autoregressive training."""
        return torch.log(torch.tril(torch.ones(sz, sz)))

    def _get_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Get or generate causal mask for the given sequence length.
        TorchScript-compatible mask caching.
        """
        if self._cached_seq_len != seq_len:
            # Generate new mask on the correct device
            mask = torch.log(torch.tril(torch.ones(seq_len, seq_len, device=device)))
            self._mask_cache = mask
            self._cached_seq_len = seq_len
        else:
            # Ensure cached mask is on correct device
            if self._mask_cache.device != device:
                self._mask_cache = self._mask_cache.to(device)

        return self._mask_cache

    def forward(self, src: torch.Tensor, has_mask: bool = True) -> torch.Tensor:
        """
        Args:
            src: Input tensor of shape [seq_len, batch_size]
            has_mask: Whether to apply causal masking

        Returns:
            Log probabilities of shape [seq_len, batch_size, vocab_size]
        """
        seq_len = src.size(0)

        # Generate or reuse mask
        mask: Optional[torch.Tensor] = None
        if has_mask:
            mask = self._get_mask(seq_len, src.device)

        # Embedding with positional encoding
        # TorchScript will fuse: embedding -> mul -> add operations
        src = self.input_emb(src) * self.ninp_sqrt
        src = self.pos_encoder(src)

        # Encoder forward pass (graph will be optimized by TorchScript)
        output = self.encoder(src, mask=mask)

        # Decoder projection and softmax
        # TorchScript will fuse: linear -> log_softmax
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)


def create_scripted_model(ntoken: int, ninp: int, nhead: int, nhid: int, nlayers: int,
                          dropout: float = 0.5, device: str = 'cuda') -> torch.jit.ScriptModule:
    """
    Factory function to create and compile a TorchScript transformer model.

    This provides a fair comparison point against MAX Graph by enabling:
    - Operator fusion (elementwise ops, linear+activation, etc.)
    - Graph compilation and optimization
    - Kernel selection based on shape/device
    - Eliminated Python interpreter overhead

    Args:
        ntoken: Vocabulary size
        ninp: Embedding dimension
        nhead: Number of attention heads
        nhid: Feed-forward hidden dimension
        nlayers: Number of transformer layers
        dropout: Dropout probability
        device: Device to place model on

    Returns:
        Compiled TorchScript model ready for inference/training
    """
    model = ScriptedTransformerModel(ntoken, ninp, nhead, nhid, nlayers, dropout)
    model = model.to(device)
    model.eval()  # Put in eval mode for scripting

    # Script the model - this performs graph compilation
    print("Compiling model with TorchScript...")
    scripted_model = torch.jit.script(model)

    # Optionally optimize for inference (freezes weights, additional fusion)
    # Note: Only use freeze for inference, not training
    # scripted_model = torch.jit.freeze(scripted_model)

    print("[OK] TorchScript compilation complete")
    return scripted_model
