from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union, Dict, TYPE_CHECKING

from heybuddy.util.pretrained_util import PretrainedModelMixin

if TYPE_CHECKING:
    import torch

__all__ = ["PretrainedONNXModel"]

class PretrainedONNXModel(PretrainedModelMixin):
    """
    A wrapper around an ONNX model that can be used to run inference on new data.
    """
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
        The device on which the model is running.
        """
        import torch
        if self.device_id is None:
            return torch.device("cpu")
        return torch.device(f"cuda:{self.device_id}")

    @classmethod
    def from_file(
        cls,
        pretrained_model_path: str,
        device_id: Optional[int]=None,
        load: bool=False
    ) -> PretrainedONNXModel:
        """
        Create a new instance of this class from a file path.
        """
        instance = cls(
            device_id=device_id,
            load=False
        )
        instance._pretrained_model_path = pretrained_model_path
        if load:
            instance.load()
        return instance

    def unload(self) -> None:
        """
        Reload the ONNX model into memory.
        """
        self.loaded = False
        if hasattr(self, "session"):
            del self.session

    def load(self) -> None:
        """
        Load the ONNX model into memory.
        """
        if self.loaded:
            return
        from onnxruntime import InferenceSession, SessionOptions # type: ignore[import-untyped]
        self.session_options = SessionOptions()
        providers: Sequence[Union[str, Tuple[str, Dict[str, str]]]] = []
        if self.device_id is None:
            providers = ["CPUExecutionProvider"]
        else:
            providers = [("CUDAExecutionProvider", {"device_id": str(self.device_id)})]
        self.session = InferenceSession(
            self.pretrained_model_path,
            providers=providers,
            session_options=self.session_options
        )
        self.loaded = True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Run the model on the given input.
        """
        if not self.loaded:
            self.load()
        retry = kwargs.pop("retry", True)
        try:
            return self.session.run(None, kwargs)
        except Exception as e:
            if retry:
                self.unload()
                return self(*args, retry=False, **kwargs)
            raise
