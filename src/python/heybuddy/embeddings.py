from __future__ import annotations

import numpy as np

from typing import Union, List, Optional, Tuple, Callable, Dict, Any, TYPE_CHECKING

from heybuddy.spectrogram import MelSpectrogramModel
from heybuddy.util import (
    PretrainedONNXModel,
    audio_to_bct_tensor,
    logger
)

if TYPE_CHECKING:
    import torch
    from heybuddy.util.typing_util import AudioType

__all__ = [
    "SpeechEmbeddings",
    "get_speech_embeddings",
]

class SpeechEmbeddingModel(PretrainedONNXModel):
    """
    Compute speech embeddings from spectrograms.

    This is an ONNX model based on Google's Tensorflow implementation of the same model.
    """
    pretrained_model_url = "https://huggingface.co/benjamin-paine/hey-buddy/resolve/main/pretrained/speech-embedding.onnx"
    pretrained_model_sha256_sum = "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f"

    def __call__(self, spectrograms: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        Compute embeddings from spectrograms.

        Args:
            spectrograms: A numpy array of mel spectrograms.

        Returns:
            A numpy array of embeddings.
        """
        return super().__call__(input_1=spectrograms)[0].squeeze() # type: ignore[no-any-return]

class SpeechEmbeddings:
    """
    A class to compute embeddings from audio files.
    """
    def __init__(
        self,
        device_id: Optional[int]=None,
        load: bool=False,
    ) -> None:
        self.spectrogram = MelSpectrogramModel(device_id=device_id, load=load)
        self.embeddings = SpeechEmbeddingModel(device_id=device_id, load=load)

    def audio_to_spectrograms(
        self,
        audio: torch.Tensor,
        batch_size: int=128,
        mel_bins: int=32,
        on_progress: Optional[Callable[[int, int], None]]=None,
    ) -> np.ndarray[Any, Any]:
        """
        Compute mel spectrograms from audio.
        """
        b, t = audio.shape # batch, time
        n_frames = int(np.ceil(t / 160 - 3)) # 160 = 10ms * 16kHz
        n_total = b * n_frames

        # Buffer for spectrograms
        spectrograms = np.empty((b, n_frames, mel_bins), dtype=np.float32)

        # Make batches
        for i in range(0, max(b, batch_size), batch_size):
            batch = audio[i:i+batch_size]
            mel = self.spectrogram(batch.detach().cpu().numpy())
            spectrograms[i:i+batch_size, :, :] = mel.squeeze()
            if on_progress is not None:
                on_progress(min(i+batch_size, b), n_total)

        if on_progress is not None:
            on_progress(n_total, n_total)

        return spectrograms

    def spectrograms_to_embeddings(
        self,
        spectrograms: np.ndarray[Any, Any],
        batch_size: int=128,
        embedding_dim: int=96,
        window_size: int=76,
        window_stride: int=8,
        on_progress: Optional[Callable[[int, int], None]]=None,
    ) -> np.ndarray[Any, Any]:
        """
        Compute embeddings from spectrograms.
        """
        b, t, m = spectrograms.shape # batch, time, mel
        spectrograms = spectrograms[:, :, :, np.newaxis] # re-add channel dimension
        assert t >= window_size, f"Time dimension {t} must be at least {window_size}"
        n_frames = (t - window_size) // window_stride + 1
        n_total = b * n_frames

        # Buffer for embeddings
        embeddings = np.empty((b, n_frames, embedding_dim), dtype=np.float32)

        # Buffer for batch of windows
        batch: List[Tuple[int, int, np.ndarray[Any, Any]]] = [] # (i, j, spectrogram)
        n_processed: int = 0

        def process_batch() -> None:
            """
            Process the current batch of windows.
            """
            nonlocal n_processed
            if not batch:
                return
            window = np.array([task[2] for task in batch])
            result = self.embeddings(window)
            for x, (i, j, _) in enumerate(batch):
                embeddings[i, j // window_stride] = result[x]
            n_processed += len(batch)
            if on_progress is not None:
                on_progress(n_processed, n_total)
            batch.clear()

        def push_batch(i: int, j: int, window: np.ndarray[Any, Any]) -> None:
            """
            Add a window to the batch, processing if full.
            """
            batch.append((i, j, window))
            if len(batch) >= batch_size:
                process_batch()

        # Iterate over batches
        for i, spectrogram in enumerate(spectrograms):
            # Iterate over frames
            for j in range(0, t, window_stride):
                window = spectrogram[j:j+window_size]
                if window.shape[0] < window_size:
                    break
                # Will trigger process_batch if batch is full
                push_batch(i, j, window)

        # Process any remaining frames
        process_batch()

        if on_progress is not None:
            on_progress(n_total, n_total)

        return embeddings

    def __call__(
        self,
        audio: AudioType,
        spectrogram_batch_size: int=32,
        mel_bins: int=32,
        embedding_batch_size: int=32,
        embedding_dim: int=96,
        window_size: int=76,
        window_stride: int=8,
        audio_window_size: int=17280,
        audio_window_stride: int=1920,
        on_spectrogram_progress: Optional[Callable[[int, int], None]]=None,
        on_embedding_progress: Optional[Callable[[int, int], None]]=None,
        remove_nan: bool=True,
        return_spectrograms: bool=False,
    ) -> Union[
        np.ndarray[Any, Any],
        Tuple[
            np.ndarray[Any, Any],
            np.ndarray[Any, Any],
        ]
    ]:
        """
        Compute embeddings from audio.
        """
        audio_tensor, sample_rate = audio_to_bct_tensor(
            audio,
            sample_rate=16000,
        )
        audio_tensor *= 32767.0 # Return to int16 range values, but as float32
        if audio_tensor.shape[1] > 1:
            audio_tensor = audio_tensor.mean(dim=1, keepdim=True)

        audio_tensor = audio_tensor[:, 0, :] # batch, time
        embeddings_list = []
        spectrograms_list = []

        for i in range(0, audio_tensor.shape[1] - audio_window_size + 1, audio_window_stride):
            spectrograms = self.audio_to_spectrograms(
                audio_tensor[:, i:i+audio_window_size],
                batch_size=spectrogram_batch_size,
                mel_bins=mel_bins,
                on_progress=on_spectrogram_progress,
            )
            embeddings = self.spectrograms_to_embeddings(
                spectrograms,
                batch_size=embedding_batch_size,
                embedding_dim=embedding_dim,
                window_size=window_size,
                window_stride=window_stride,
                on_progress=on_embedding_progress,
            )
            embeddings_list.append(embeddings)
            if return_spectrograms:
                spectrograms_list.append(spectrograms)

        embeddings = np.concatenate(embeddings_list, axis=1)
        if return_spectrograms:
            spectrograms = np.concatenate(spectrograms_list, axis=1)

        if remove_nan:
            remove_nan_indices = []
            for i, embedding in enumerate(embeddings):
                if np.isnan(embedding).any():
                    remove_nan_indices.append(i)

            if remove_nan_indices:
                logger.warning(f"Replacing {len(remove_nan_indices)} NaN embeddings with random embeddings.")
                # replace nans with a different random embedding if possible
                keep_indices = np.setdiff1d(np.arange(len(embeddings)), remove_nan_indices)
                if keep_indices.size == 0:
                    logger.warning("All embeddings are NaN, returning zero embeddings.")
                    return np.zeros(embeddings.shape, dtype=np.float32)
                for i in remove_nan_indices:
                    embeddings[i] = embeddings[np.random.choice(keep_indices)]

        if return_spectrograms:
            b, t, m = spectrograms.shape # batch, time, mel
            truncated_t = t - ((t - window_size) % window_stride)
            return embeddings, spectrograms[:, :truncated_t]

        return embeddings

GLOBAL_EMBEDDINGS: Dict[Optional[int], SpeechEmbeddings] = {}
def get_speech_embeddings(device_id: Optional[int]=None) -> SpeechEmbeddings:
    """
    Get a SpeechEmbeddings instance for a given device_id.
    """
    if device_id not in GLOBAL_EMBEDDINGS:
        GLOBAL_EMBEDDINGS[device_id] = SpeechEmbeddings(device_id=device_id)
    return GLOBAL_EMBEDDINGS[device_id]
