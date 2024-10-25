from __future__ import annotations

import os

from typing import Type, Tuple, Dict, Any, Optional, TYPE_CHECKING
from heybuddy.util.pretrained_util import PretrainedModelMixin

if TYPE_CHECKING:
    import torch

__all__ = [
    "load_state_dict",
    "PretrainedTorchModel"
]

def load_state_dict(path: str) -> Dict[str, Any]:
    """
    Load a state dict from a file.

    This is only a small extension on `torch.load` to allow safetensors files.
    It is not suitable for loading very large models, see taproot for that.

    :param path: The path to the file to load.
    :return: The state dict. Usually a dictionary of tensors, but torch.load can return other things too.
    """
    name, ext = os.path.splitext(os.path.basename(path))
    if ext == ".safetensors":
        try:
            from safetensors import safe_open
            checkpoint: Dict[str, Any] = {}
            with safe_open(path, framework="pt", device="cpu") as f: # type: ignore[attr-defined,no-untyped-call]
                for key in f.keys():
                    checkpoint[key] = f.get_tensor(key)
            return checkpoint
        except ImportError:
            raise ImportError("You need to install the safetensors package to load safetensors files. Run `pip install safetensors`.")
    else:
        import torch
        return torch.load(path, map_location="cpu") # type: ignore[no-any-return]

class PretrainedTorchModel(PretrainedModelMixin):
    """
    A wrapper around an Torch model that can be used to run inference on new data.
    """
    model_args: Optional[Tuple[Any]] = None
    model_kwargs: Optional[Dict[str, Any]] = None
    model_path: Optional[str] = None
    model_eval: bool = True
    module: Optional[torch.nn.Module] = None

    def __init__(
        self,
        device_id: Optional[int]=None,
        load: bool=False
    ) -> None:
        self.loaded = False
        self.device_id = device_id
        if load:
            self.load()

    @property
    def device(self) -> torch.device:
        """
        Get the device that the model is loaded on.
        """
        import torch
        if not hasattr(self, "_device"):
            if self.device_id is None:
                self._device = torch.device("cpu")
            else:
                self._device = torch.device(f"cuda:{self.device_id}")
        return self._device

    @property
    def model(self) -> torch.nn.Module:
        """
        Get the model that is loaded.
        """
        model = self.module
        if self.model_path is not None:
            for model_path_part in self.model_path.split("."):
                model = getattr(model, model_path_part)
        if model is None:
            raise ValueError("Model path not found.")
        return model

    @property
    def dtype(self) -> torch.dtype:
        """
        Get the dtype that the model is loaded with.
        """
        import torch
        if not self.loaded:
            return torch.float32
        return next(self.model.parameters()).dtype

    @classmethod
    def get_model_class(cls) -> Optional[Type[torch.nn.Module]]:
        """
        Get the model class that this model uses.
        """
        return None

    def load(self) -> None:
        """
        Load the Torch model into memory.
        """
        import torch
        if self.loaded:
            return

        model_class = self.get_model_class()
        if model_class is not None:
            model_args = self.model_args or ()
            model_kwargs = self.model_kwargs or {}
            self.module = model_class(*model_args, **model_kwargs)
            self.module.load_state_dict(load_state_dict(self.pretrained_model_path))
            self.module.to(self.device)
        else:
            self.module = torch.load(self.pretrained_model_path)

        if self.model_eval and self.module:
            self.module.eval()

    def unload(self) -> None:
        """
        Unload the Torch model from memory.
        """
        if not self.loaded:
            return
        self.module = None
        self.loaded = False

    def eval(self) -> None:
        """
        Set the model to evaluation mode.
        """
        if not self.loaded:
            self.load()
        assert self.module is not None
        self.module.eval()

    def to(
        self,
        device: Optional[torch.device]=None,
        dtype: Optional[torch.dtype]=None
    ) -> None:
        """
        Move the model to a new device and/or dtype.
        """
        if not self.loaded:
            self.load()
        assert self.module is not None
        self.module.to(device=device, dtype=dtype)

    def prepare_tensors(self, arg: Any) -> Any:
        """
        Prepare a tensor for running through the model.

        :param arg: The argument to prepare. This can be anything, and will be recursively processed.
        :return: The prepared argument.
        """
        import torch
        if isinstance(arg, torch.Tensor):
            return arg.to(device=self.device, dtype=self.dtype)
        elif isinstance(arg, list):
            return [self.prepare_tensors(a) for a in arg]
        elif isinstance(arg, tuple):
            return tuple(self.prepare_tensors(a) for a in arg)
        elif isinstance(arg, dict):
            return {k: self.prepare_tensors(v) for k, v in arg.items()}
        else:
            return arg

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Run the model on the given input.
        """
        if not self.loaded:
            self.load()
        return self.model(
            *self.prepare_tensors(args),
            **self.prepare_tensors(kwargs)
        )
