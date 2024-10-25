import numpy as np
import torch
from typing import Sequence, Union, Any

__all__ = ['SingleAudioType', 'AudioType']

SingleAudioType = Union[str, bytes, bytearray, np.ndarray[Any, Any], torch.Tensor]
AudioType = Union[SingleAudioType, Sequence[SingleAudioType]]
