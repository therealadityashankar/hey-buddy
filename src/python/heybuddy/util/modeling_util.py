from __future__ import annotations

from typing import Optional, Any, TYPE_CHECKING
from typing_extensions import Literal

if TYPE_CHECKING:
    import torch.nn as nn

__all__ = [
    "ActivationFunctionLiteral",
    "get_activation",
    "find_nearest_multiple",
    "get_normalized_dim"
]

ActivationFunctionLiteral = Literal["relu", "gelu", "silu", "swish", "mish", "tanh", "sigmoid", "identity"]

def find_nearest_multiple(
    input_number: int,
    multiple: int,
    direction: Literal["up", "down"] = "up",
) -> int:
    """
    Find the nearest multiple of a number in a given direction

    >>> find_nearest_multiple(95, 8)
    96
    >>> find_nearest_multiple(100, 8)
    104

    :param input_number: The number for which the nearest multiple is to be found
    :param multiple: The number whose multiple is to be found
    :param direction: The direction in which the nearest multiple is to be found. Default is "up"
    :return: The nearest multiple of the input number
    """
    if input_number % multiple == 0:
        return input_number
    if direction == "down":
        return input_number - (input_number % multiple)
    return input_number + multiple - (input_number % multiple)

def get_normalized_dim(
    dim: int,
    multiple_of: int=8,
    down_ratio: float=2/3,
) -> int:
    """
    Gets a normalized dimension by first multiplying by a down ratio,
    then rounding up to the nearest multiple of a number.

    The default of 2/3 does a good job in keeping a balance of rounding
    down in the first step and up in the second step.

    >>> get_normalized_dim(76, 32)
    64
    >>> get_normalized_dim(100, 32)
    96
    >>> get_normalized_dim(106, 32)
    96
    >>> get_normalized_dim(146, 32)
    128

    :param dim: The dimension to be normalized
    :param down_ratio: The ratio by which the dimension is first multiplied by. Default is 2/3
    :param multiple_of: The number to which the dimension is rounded to. Default is 8
    :return: The normalized dimension
    """
    return find_nearest_multiple(
        int(dim * down_ratio),
        multiple_of,
        "up"
    )

def get_activation(
    activation: Optional[ActivationFunctionLiteral],
    *args: Any,
    **kwargs: Any
) -> nn.Module:
    """
    Returns an activation function module based on the provided name.

    The supported activation functions are:

    - "relu": Rectified Linear Unit, commonly used in many neural networks.
    - "gelu": Gaussian Error Linear Unit, often used in transformer architectures.
    - "silu": Sigmoid Linear Unit, also known as "swish", a smooth non-linear activation.
    - "swish": Alias for "silu" for consistency with other naming conventions.
    - "mish": Another smooth activation function similar to "swish".
    - "tanh": Hyperbolic tangent, used in tasks requiring a bounded output in the range [-1, 1].
    - "sigmoid": Produces outputs in the range [0, 1], useful for binary classification tasks.
    - "identity": No-op activation, returns the input unchanged. This can be useful when you want
      to effectively "turn off" activation without modifying the model structure.

    :param activation: A string representing the name of the desired activation function.
                       If None is provided, or if "identity" is selected, no activation will be applied.
    :param args: Additional positional arguments to be passed to the activation function, if any.
    :param kwargs: Additional keyword arguments to be passed to the activation function, if any.
    :return: A PyTorch activation function module (`torch.nn.Module`).
    :raises ValueError: If an unknown activation function is passed.
    """
    import torch.nn as nn
    activation_map = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "mish": nn.Mish,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "identity": nn.Identity,
        None: nn.Identity
    }.get(activation, None)
    if activation_map is None:
        raise ValueError(f"Activation function '{activation}' not found.")
    return activation_map(*args, **kwargs) # type: ignore[no-any-return]
