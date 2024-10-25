import torch
import torch.nn as nn

from typing import Optional

from heybuddy.modules.base import Module
from heybuddy.modules.attention import Attention
from heybuddy.modules.multi_layer_perceptron import GatedMultiLayerPerceptron

__all__ = ["TransformerBlock"]

class TransformerBlock(Module):
    """
    Transformer block with attention and feed-forward layers.
    
    :see: https://arxiv.org/abs/1706.03762
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        multiple_of: int,
        norm_epsilon: float = 1e-5,
        hidden_dim_multiplier: int = 4,
        layer_id: Optional[int] = None,
    ) -> None:
        """
        :param dim: Dimension of the input tensor.
        :param num_heads: Number of attention heads.
        :param multiple_of: Multiple of the input tensor dimension.
        :param norm_epsilon: Epsilon value for normalization layers.
        :param hidden_dim_multiplier: Multiplier for the hidden dimension in the feed-forward layer.
        :param layer_id: Identifier of the layer. Optional.
        """
        super(TransformerBlock, self).__init__()

        self.layer_id = layer_id
        self.norm_epsilon = norm_epsilon

        self.attention = Attention(dim, num_heads)
        self.feed_forward = GatedMultiLayerPerceptron(
            input_dim=dim,
            hidden_dim=dim * hidden_dim_multiplier,
            multiple_of=multiple_of,
        )

        self.attention_norm = nn.LayerNorm(dim, eps=norm_epsilon)
        self.feed_forward_norm = nn.LayerNorm(dim, eps=norm_epsilon)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the transformer block.

        :param x: Input tensor.
        :return: Output tensor.
        """
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x
