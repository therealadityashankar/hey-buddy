# Released by Benjamin Paine under the Unlicense, see the LICENSE file or unlicense.org, 2024.

import math
import torch
import torch.nn as nn

from typing import Optional

from heybuddy.modules.base import Module
from heybuddy.modules.mixed_precision import FloatLayerNorm

__all__ = ["Attention"]

class Attention(Module):
    """
    The attention mechanism is foundational to deep learning as
    a whole. In essence, it allows the model to focus on specific
    parts of the input sequence when making predictions.

    It is fundementally composed of two parts; *keys* and *values*,
    similar to a dictionary. The *keys* are the input sequence, and
    the *values* are the *weights* that are assigned to each key.

    A query is then passed to the attention mechanism, which is used
    to compute the weights that are assigned to each key. The weights
    are then used to compute the output.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        linear_bias: bool = False,
        norm_bias: bool = True,
        elementwise_affine: bool = True,
        scale_by_num_heads: bool = False,
    ) -> None:
        """
        :param dim: The dimension of the query tensor.
        :param num_heads: The number of attention heads (parallel attention mechanisms).
        :param linear_bias: Whether to include a bias term in the linear transformation.
        :param norm_bias: Whether to include a bias term in the normalization layer.
        :param elementwise_affine: Whether to include a learnable affine transformation.
        :param scale_by_num_heads: Whether to scale the output by the number of heads.
        """
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // self.num_heads
        self.inner_dim = self.head_dim * self.num_heads
        self.scale_by_num_heads = scale_by_num_heads

        self.queries = nn.Linear(self.dim, self.inner_dim, bias=linear_bias)
        self.keys = nn.Linear(self.dim, self.inner_dim, bias=linear_bias)
        self.values = nn.Linear(self.dim, self.inner_dim, bias=linear_bias)
        self.output = nn.Linear(self.inner_dim, self.dim, bias=linear_bias)

        self.query_norm = FloatLayerNorm(self.inner_dim, bias=norm_bias, elementwise_affine=elementwise_affine)
        self.key_norm = FloatLayerNorm(self.inner_dim, bias=norm_bias, elementwise_affine=elementwise_affine)

    def scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        The scaled dot product attention mechanism.
        """
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward method of the attention mechanism.

        :param x: The input tensor.
        :return: The output tensor.
        """
        b, s = x.shape[:2]

        x_q, x_k, x_v = self.queries(x), self.keys(x), self.values(x)

        x_q = self.query_norm(x_q)
        x_k = self.key_norm(x_k)

        x_q = x_q.view(b, s, self.num_heads, self.head_dim)
        x_k = x_k.view(b, s, self.num_heads, self.head_dim)
        x_v = x_v.view(b, s, self.num_heads, self.head_dim)

        output = self.scaled_dot_product_attention(
            x_q.permute(0, 2, 1, 3),
            x_k.permute(0, 2, 1, 3),
            x_v.permute(0, 2, 1, 3),
            dropout_p=0.0,
            is_causal=False,
            scale=1 / self.head_dim ** 0.5 if self.scale_by_num_heads else 1.0,
        )
        output = output.permute(0, 2, 1, 3)
        output = output.flatten(-2)
        output = self.output(output)

        return output
