from typing import Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F

# Try to import FlashAttention, but provide fallback
FLASH_ATTENTION_AVAILABLE = False
try:
    from flash_attn_interface import flash_attn_func  # type: ignore[import]
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    try:
        from flash_attn import flash_attn_func  # type: ignore[import]
        FLASH_ATTENTION_AVAILABLE = True
    except ImportError:
        print("⚠️  FlashAttention not available, using PyTorch native attention (slower but works on CPU)")
        flash_attn_func = None

from models.common import trunc_normal_init_


CosSin = Tuple[torch.Tensor, torch.Tensor]


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class CastedLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)


class CastedEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_std: float,
                 cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Truncated LeCun normal init
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor, key_value_states: torch.Tensor = None) -> torch.Tensor:
        """
        multi-head attention with optional cross-attention support.
        
        Args:
            cos_sin: Tuple of (cos, sin) for RoPE positional encodings, or None
            hidden_states: Query states [batch, seq_len_q, hidden_size]
            key_value_states: Optional. If provided, performs cross-attention where:
                            - Queries come from hidden_states
                            - Keys/Values come from key_value_states
                            Shape: [batch, seq_len_kv, hidden_size]
                            If None, performs self-attention (default behavior)
        
        Returns:
            Attention output [batch, seq_len_q, hidden_size]
            
        notes:
        - self-attention: Q, K, V all from same sequence (hidden_states)
        - cross-attention: Q from one sequence, K/V from another (key_value_states)
        - this enables H <--> L communication in hierarchical reasoning
        """
        batch_size, seq_len_q, _ = hidden_states.shape
        
        # now compute queries from hidden_states 
        # project to (num_heads * head_dim) for queries and then split it 
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len_q, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]  # [batch, seq_len_q, num_heads, head_dim]
        
        # now compute keys and values
        if key_value_states is not None:
            # cross-attention mode
            # keys and values come from a different sequence (key_value_states)
            seq_len_kv = key_value_states.shape[1]
            
            # project key_value_states to get K and V
            kv = self.qkv_proj(key_value_states)
            kv = kv.view(batch_size, seq_len_kv, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
            
            # extract key and value from the projection
            # we only need K and V, not Q (Q comes from hidden_states)
            key = kv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
            value = kv[:, :, self.num_heads + self.num_key_value_heads:]
        else:
            # self-attention mode (original behavior)
            # keys and values come from the same sequence as queries
            key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
            value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
            seq_len_kv = seq_len_q

        # now apply RoPE (Rotary Position Embeddings)
        if cos_sin is not None:
            cos, sin = cos_sin
            
            # check if we can apply RoPE to both Q and K
            # for cross-attention with different lengths, we might need different handling
            if seq_len_q == seq_len_kv:
                # same length: apply RoPE to both Q and K using same positions
                query, key = apply_rotary_pos_emb(query, key, cos, sin)
            else:
                # different lengths (cross-attention between H and L)
                # query gets its own positional encoding (from cos_sin)
                cos_q = cos[:seq_len_q] if cos.shape[0] >= seq_len_q else cos
                sin_q = sin[:seq_len_q] if sin.shape[0] >= seq_len_q else sin
                query = (query * cos_q.unsqueeze(-2)) + (rotate_half(query) * sin_q.unsqueeze(-2))
                
                # for cross-attention, keys use their own positional encoding
                # (passed via separate cos_sin in the cross-attention module)
                # so we don't apply position here for keys in cross-attention

        # now do attention (FlashAttention if available, otherwise PyTorch native)
        if FLASH_ATTENTION_AVAILABLE and flash_attn_func is not None:
            # efficient attention computation using FlashAttention
            attn_output = flash_attn_func(q=query, k=key, v=value, causal=self.causal)
            if isinstance(attn_output, tuple):  # fa2 and fa3 compatibility
                attn_output = attn_output[0]
        else:
            # fallback to PyTorch native attention (works on CPU, slower)
            attn_output = self._native_attention(query, key, value, causal=self.causal)

        # now project back to hidden_size
        # reshape from [batch, seq_len_q, num_heads, head_dim] to [batch, seq_len_q, hidden_size]
        attn_output = attn_output.view(batch_size, seq_len_q, self.output_size)  # type: ignore
        return self.o_proj(attn_output)
    
    def _native_attention(
        self, 
        query: torch.Tensor,  # [batch, seq_len_q, num_heads, head_dim]
        key: torch.Tensor,    # [batch, seq_len_kv, num_heads, head_dim]
        value: torch.Tensor,  # [batch, seq_len_kv, num_heads, head_dim]
        causal: bool = False
    ) -> torch.Tensor:
        """
        fallback attention using PyTorch's built-in operations.
        slower than FlashAttention but works on CPU and doesn't require compilation.
        
        implements: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
        """
        batch_size, seq_len_q, num_heads, head_dim = query.shape
        seq_len_kv = key.shape[1]
        
        # ensure all tensors have same dtype (fix for bfloat16 vs float32 mismatch)
        # convert everything to query's dtype to maintain consistency
        target_dtype = query.dtype
        key = key.to(target_dtype)
        value = value.to(target_dtype)
        
        # transpose to [batch, num_heads, seq_len, head_dim] for matmul
        query = query.transpose(1, 2)  # [batch, num_heads, seq_len_q, head_dim]
        key = key.transpose(1, 2)      # [batch, num_heads, seq_len_kv, head_dim]
        value = value.transpose(1, 2)  # [batch, num_heads, seq_len_kv, head_dim]
        
        # compute attention scores: Q @ K^T / sqrt(d_k)
        scale = torch.tensor(head_dim ** -0.5, dtype=target_dtype, device=query.device)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        # shape: [batch, num_heads, seq_len_q, seq_len_kv]
        
        # apply causal mask if needed (for autoregressive models)
        if causal:
            # create causal mask: upper triangular matrix of -inf
            causal_mask = torch.triu(
                torch.full((seq_len_q, seq_len_kv), float('-inf'), device=query.device),
                diagonal=1
            )
            attn_scores = attn_scores + causal_mask
        
        # apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        # shape: [batch, num_heads, seq_len_q, seq_len_kv]
        
        # apply attention weights to values: weights @ V
        attn_output = torch.matmul(attn_weights, value)
        # shape: [batch, num_heads, seq_len_q, head_dim]
        
        # transpose back to [batch, seq_len_q, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        return attn_output


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj    = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)