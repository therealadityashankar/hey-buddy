# Adapted from https://github.com/dscripka/piper-sample-generator/blob/master/piper_train/vits/transforms.py
import numpy as np
import torch

from torch.nn import functional as F

from typing import Optional, Tuple

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3

def piecewise_rational_quadratic_transform(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    inverse: bool = False,
    tails: Optional[str]=None,
    tail_bound: float=1.0,
    min_bin_width: float=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative: float=DEFAULT_MIN_DERIVATIVE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Piecewise rational quadratic spline bijector.

    :param inputs: Tensor, shape (batch_size, features)
    :param unnormalized_widths: Tensor, shape (batch_size, num_bins)
    :param unnormalized_heights: Tensor, shape (batch_size, num_bins)
    :param unnormalized_derivatives: Tensor, shape (batch_size, num_bins - 1)
    :param inverse: bool, whether to apply the inverse of the transform
    :param tails: str, type of tail bound to use. One of "linear" or None.
    :param tail_bound: float, bound of the tail
    :param min_bin_width: float, minimal bin width
    :param min_bin_height: float, minimal bin height
    :param min_derivative: float, minimal derivative
    :return: Tuple of two Tensors, shape (batch_size, features)
    """
    if tails is None:
        spline_fn = rational_quadratic_spline
        spline_kwargs = {}
    else:
        spline_fn = unconstrained_rational_quadratic_spline # type: ignore[assignment]
        spline_kwargs = {"tails": tails, "tail_bound": tail_bound}

    outputs, logabsdet = spline_fn(
        inputs=inputs,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        inverse=inverse,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
        **spline_kwargs # type: ignore[arg-type]
    )
    return outputs, logabsdet

def search_sorted(
    bin_locations: torch.Tensor,
    inputs: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Find the indices of the bins to which each value in input belongs.

    :param bin_locations: Tensor, shape (batch_size, num_bins)
    :param inputs: Tensor, shape (batch_size, features)
    :param eps: float, small number to prevent numerical errors
    :return: Tensor, shape (batch_size, features)
    """
    bin_locations[..., bin_locations.size(-1) - 1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1

def unconstrained_rational_quadratic_spline(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    inverse: bool=False,
    tails: Optional[str]="linear",
    tail_bound: float=1.0,
    min_bin_width: float=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative: float=DEFAULT_MIN_DERIVATIVE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unconstrained rational quadratic spline bijector.

    :param inputs: Tensor, shape (batch_size, features)
    :param unnormalized_widths: Tensor, shape (batch_size, num_bins)
    :param unnormalized_heights: Tensor, shape (batch_size, num_bins)
    :param unnormalized_derivatives: Tensor, shape (batch_size, num_bins - 1)
    :param inverse: bool, whether to apply the inverse of the transform
    :param tails: str, type of tail bound to use. One of "linear" or None.
    :param tail_bound: float, bound of the tail
    :param min_bin_width: float, minimal bin width
    :param min_bin_height: float, minimal bin height
    :param min_derivative: float, minimal derivative
    :return: Tuple of two Tensors, shape (batch_size, features)
    """
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == "linear":
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        # unnormalized_derivatives[..., -1] = constant
        unnormalized_derivatives[..., unnormalized_derivatives.size(-1) - 1] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError("{} tails are not implemented.".format(tails))

    (
        outputs[inside_interval_mask],
        logabsdet[inside_interval_mask],
    ) = rational_quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )

    return outputs, logabsdet

def rational_quadratic_spline(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    inverse: bool=False,
    left: float=0.0,
    right: float=1.0,
    bottom: float=0.0,
    top: float=1.0,
    min_bin_width: float=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative: float=DEFAULT_MIN_DERIVATIVE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rational quadratic spline bijector.

    :param inputs: Tensor, shape (batch_size, features)
    :param unnormalized_widths: Tensor, shape (batch_size, num_bins)
    :param unnormalized_heights: Tensor, shape (batch_size, num_bins)
    :param unnormalized_derivatives: Tensor, shape (batch_size, num_bins - 1)
    :param inverse: bool, whether to apply the inverse of the transform
    :param left: float, left boundary of the spline
    :param right: float, right boundary of the spline
    :param bottom: float, bottom boundary of the spline
    :param top: float, top boundary of the spline
    :param min_bin_width: float, minimal bin width
    :param min_bin_height: float, minimal bin height
    :param min_derivative: float, minimal derivative
    :return: Tuple of two Tensors, shape (batch_size, features)
    """
    num_bins = unnormalized_widths.shape[-1]

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., cumwidths.size(-1) - 1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., cumheights.size(-1) - 1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = search_sorted(cumheights, inputs)[..., None]
    else:
        bin_idx = search_sorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all(), discriminant

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, -logabsdet

    theta = (inputs - input_cumwidths) / input_bin_widths
    theta_one_minus_theta = theta * (1 - theta)

    numerator = input_heights * (
        input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
    )
    denominator = input_delta + (
        (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
        * theta_one_minus_theta
    )
    outputs = input_cumheights + numerator / denominator

    derivative_numerator = input_delta.pow(2) * (
        input_derivatives_plus_one * theta.pow(2)
        + 2 * input_delta * theta_one_minus_theta
        + input_derivatives * (1 - theta).pow(2)
    )
    logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

    return outputs, logabsdet
