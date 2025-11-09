# Inspiration has been taken from the ROFORMER paper and their git repository
# ROFORMER: ENHANCED TRANSFORMER WITH ROTARYPOSITION EMBEDDING - https://arxiv.org/pdf/2104.09864
# ROFORMER GitHub Repository - https://github.com/ZhuiyiTechnology/roformer

import math
import torch
import torch.nn as nn
from seq2seq import utils
from seq2seq.models import register_model, register_model_architecture
from seq2seq.models import Seq2SeqModel, Seq2SeqEncoder, Seq2SeqDecoder
import sentencepiece as spm

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Helper used in RoPE: rotate last dimension (even,odd) -> (-odd, even)
    x: [..., d], d must be even
    """
    x_even = x[..., 0::2] # even positions, starting 0, step 2
    x_odd = x[..., 1::2] # odd positions, starting 1, step 2
    x_rot_even = -x_odd
    x_rot_odd = x_even
    out = torch.zeros_like(x)
    out[..., 0::2] = x_rot_even
    out[..., 1::2] = x_rot_odd
    return out


def get_rotary_embeddings(
    seq_len: int,
    dim: int,
    device: torch.device,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
):
    """
    Compute RoPE cos/sin tables of shape [1, 1, seq_len, dim]
    """
    assert dim % 2 == 0, "RoPE head dimension must be even"
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    # positions [seq_len]
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # [seq_len, dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)       # [seq_len, dim]
    cos = emb.cos()[None, None, :, :]            # [1, 1, seq_len, dim]
    sin = emb.sin()[None, None, :, :]
    return cos, sin


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    base: float = 10000.0,
):
    """
    Apply RoPE to query and key.

    query: [B, H, L_q, D]
    key:   [B, H, L_k, D]
    returns rotated (query, key) with same shapes.
    """
    B, H, L_q, D = query.shape
    L_k = key.size(2)

    cos_q, sin_q = get_rotary_embeddings(L_q, D, query.device, base=base, dtype=query.dtype)
    cos_k, sin_k = get_rotary_embeddings(L_k, D, key.device, base=base, dtype=key.dtype)

    query_rot = (query * cos_q) + (rotate_half(query) * sin_q)
    key_rot   = (key   * cos_k) + (rotate_half(key)   * sin_k)
    return query_rot, key_rot
