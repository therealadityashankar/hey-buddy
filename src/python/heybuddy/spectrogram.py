import numpy as np

from typing import Any, Dict, Optional

from heybuddy.util import PretrainedONNXModel

__all__ = [
    "MelSpectrogramModel"
    "get_mel_spectrogram_model"
]

class MelSpectrogramModel(PretrainedONNXModel):
    """
    Compute the mel spectrogram of an audio signal.

    This is an ONNX version of the PyTorch model from the `torchaudio` library.
    It itself is not the same spectrogram extraction used to train google's speech embedding model,
    however it is close enough to be used as a preprocessor for the embeddings.
    """
    pretrained_model_url = "https://huggingface.co/benjamin-paine/hey-buddy/resolve/main/pretrained/mel-spectrogram.onnx"
    pretrained_model_sha256_sum = "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f"

    def __call__(self, audio: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        Compute the mel spectrogram of an audio signal.
        """
        assert isinstance(audio, np.ndarray)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        assert audio.ndim == 2, f"Audio must be a 1D or 2D array, got {audio.ndim}D"
        prediction = super().__call__(input=audio.astype(np.float32))
        return np.squeeze(prediction[0])/10 + 2 # type: ignore[no-any-return]

GLOBAL_MEL_MODELS: Dict[Optional[int], MelSpectrogramModel] = {}
def get_mel_spectrogram_model(device_id: Optional[int]=None) -> MelSpectrogramModel:
    """
    Get the global MEL model.
    """
    if device_id not in GLOBAL_MEL_MODELS:
        GLOBAL_MEL_MODELS[device_id] = MelSpectrogramModel(device_id=device_id, load=True)
    return GLOBAL_MEL_MODELS[device_id]
