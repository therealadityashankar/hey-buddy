import numpy as np

from typing import Any, Optional, Tuple, Union, Dict

from heybuddy.util import PretrainedONNXModel

__all__ = ["SileroVADModel"]

class SileroVADModel(PretrainedONNXModel):
    """
    Voice activity detection model based on the Silero VAD model.
    """
    pretrained_model_url = "https://huggingface.co/benjamin-paine/hey-buddy/resolve/main/pretrained/silero-vad.onnx"
    pretrained_model_sha256_sum = None

    def __init__(
        self,
        device_id: Optional[int]=None,
        load: bool=False
    ) -> None:
        super().__init__(device_id=device_id, load=load)
        self.h = np.zeros((2, 1, 64)).astype(np.float32)
        self.c = np.zeros((2, 1, 64)).astype(np.float32)

    def trim(
        self,
        audio: np.ndarray[Any, Any],
        sample_rate: int=16000,
        frame_duration: float=0.03,
        min_start: int=2000,
        threshold: float=0.15,
        pad_s: Optional[Union[float,Tuple[float,float]]]=None,
    ) -> np.ndarray[Any, Any]:
        """
        Trims audio based on voice activity detection.
        """
        return_first = False
        if audio.ndim == 1:
            return_first = True
            audio = audio[np.newaxis, :]

        audio_len = audio.shape[1]
        frame_size = int(sample_rate * frame_duration)

        # Find first voice frame
        start = min_start
        for i in range(min_start, audio_len, frame_size):
            if self(audio[:, i:i+frame_size], sample_rate) > threshold:
                start = i
                break

        # Find last voice frame
        end = len(audio)
        for i in range(audio_len-frame_size, min_start, -frame_size):
            if self(audio[:, i:i+frame_size], sample_rate) > threshold:
                end = i
                break

        # Trim audio
        audio = np.hstack([
            audio[:, :min_start],
            audio[:, start:end]
        ])

        # Pad audio
        if isinstance(pad_s, tuple):
            pad_start, pad_end = pad_s
        elif isinstance(pad_s, float):
            pad_start = pad_end = pad_s
        else:
            pad_start = pad_end = 0

        if pad_start > 0:
            if pad_end > 0:
                audio = np.pad(audio, ((0, 0), (int(pad_start*sample_rate), int(pad_end*sample_rate))), mode="constant")
            else:
                audio = np.pad(audio, ((0, 0), (int(pad_start*sample_rate), 0)), mode="constant")
        elif pad_end > 0:
            audio = np.pad(audio, ((0, 0), (0, int(pad_end*sample_rate))), mode="constant")

        if return_first:
            return audio[0] # type: ignore[no-any-return]
        return audio

    def __call__(
        self,
        audio: np.ndarray[Any, Any],
        sample_rate: int=16000,
        retry: bool=True
    ) -> float:
        """
        Determine likelihood of voice activity in audio.
        """
        assert isinstance(audio, np.ndarray)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        assert audio.ndim == 2, f"Audio must be a 1D or 2D array, got {audio.ndim}D"
        try:
            out, self.h, self.c = super().__call__(
                input=audio.astype(np.float32),
                h=self.h,
                c=self.c,
                sr=np.array([sample_rate], dtype=np.int64),
                retry=False
            )
            return out[0][0] # type: ignore[no-any-return]
        except Exception as e:
            if retry:
                self.unload()
                return self(audio, sample_rate, retry=False)
            raise

GLOBAL_VAD_MODELS: Dict[Optional[int], SileroVADModel] = {}
def get_vad_model(device_id: Optional[int]=None) -> SileroVADModel:
    """
    Get the global VAD model.
    """
    if device_id not in GLOBAL_VAD_MODELS:
        GLOBAL_VAD_MODELS[device_id] = SileroVADModel(device_id=device_id, load=True)
    return GLOBAL_VAD_MODELS[device_id]
