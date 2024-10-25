import torch

from typing import Optional, Tuple
from torch.nn import functional as F

def init_weights(
    m: torch.nn.Module,
    mean: float=0.0,
    std: float=0.01
) -> None:
    """
    Initialize the weights of a module.

    :param m: The module to initialize.
    :param mean: The mean of the normal distribution.
    :param std: The standard deviation of the normal distribution.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size: int, dilation: int) -> int:
    """
    Calculate the padding required for a convolutional layer.

    :param kernel_size: The size of the kernel.
    :param dilation: The dilation of the kernel.
    :return: The padding required.
    """
    return int((kernel_size * dilation - dilation) / 2)

def subsequent_mask(length: int) -> torch.Tensor:
    """
    Generate a mask to prevent attention to subsequent elements.

    :param length: The length of the mask.
    :return: The mask.
    """
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
    return mask

def sequence_mask(
    length: torch.Tensor,
    max_length: Optional[int] = None
) -> torch.Tensor:
    """
    Generate a boolean mask for the length of sequences.
    
    :param length: The length of the sequences.
    :param max_length: The maximum length of the sequences.
    :return: The mask.
    """
    if max_length is None:
        max_length = int(length.max())
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Generate the path for the attention.

    :param duration: The duration of the attention. (b, 1, t_x)
    :param mask: The mask for the attention. (b, 1, t_y, t_x)
    :return: The path for the attention. (b, 1, t_y, t_x)
    """
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).type_as(mask)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, (0, 0, 1, 0, 0, 0))[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path

def slice_segments(
    x: torch.Tensor,
    ids_str: torch.Tensor,
    segment_size: int=4
) -> torch.Tensor:
    """
    Slice segments from a tensor.

    :param x: The tensor to slice.
    :param ids_str: The starting indices.
    :param segment_size: The size of the segments.
    :return: The sliced tensor.
    """
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = max(0, int(ids_str[i]))
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret

def rand_slice_segments(
    x: torch.Tensor,
    x_lengths: Optional[int]=None,
    segment_size: int=4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly slice segments from a tensor.

    :param x: The tensor to slice.
    :param x_lengths: The lengths of the sequences.
    :param segment_size: The size of the segments.
    :return: The sliced tensor and the starting indices.
    """
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str

@torch.jit.script
def fused_add_tanh_sigmoid_multiply(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    n_channels: torch.Tensor
) -> torch.Tensor:
    """
    Fused operation of adding the tanh and sigmoid activations and multiplying the results.

    :param input_a: The first input tensor.
    :param input_b: The second input tensor.
    :param n_channels: The number of channels for the tanh and sigmoid activations.
    :return: The result of the fused operation.
    """
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts
