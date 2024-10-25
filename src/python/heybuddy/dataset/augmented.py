from __future__ import annotations

import numpy as np

from typing import List, Optional, Union, Any, Dict, Callable, Iterator, TYPE_CHECKING
from threading import Lock

from heybuddy.util import logger
from heybuddy.constants import *
from heybuddy.dataset.generator import AudioDatasetGenerator

if TYPE_CHECKING:
    import torch
    from datasets import Dataset # type: ignore[import-untyped]

class AugmentedAudioGenerator(AudioDatasetGenerator):
    """
    A generator that yields augmented audio samples.

    Augmentation is performed on-the-fly, so the generator does not store
    augmented samples in memory.
    """
    _resample: Dict[int, Callable[[torch.Tensor], torch.Tensor]]

    def __init__(
        self,
        source_dataset: Dataset,
        device_id: Optional[int]=None,
        augmentation_dataset: Optional[Dataset]=None,
        impulse_response_dataset: Optional[Dataset]=None,
        target_length: float=1.44, # seconds
        sample_rate: int=16000,
        batch_size: int=128,
        seven_band_aug_prob: float=DEFAULT_AUGMENT_SEVEN_BAND_PROB,
        seven_band_aug_gain_db: float=DEFAULT_AUGMENT_SEVEN_BAND_GAIN_DB,
        tanh_distortion_prob: float=DEFAULT_AUGMENT_TANH_DISTORTION_PROB,
        tanh_min_distortion: float=DEFAULT_AUGMENT_TANH_MIN_DISTORTION,
        tanh_max_distortion: float=DEFAULT_AUGMENT_TANH_MAX_DISTORTION,
        pitch_shift_prob: float=DEFAULT_AUGMENT_PITCH_SHIFT_PROB,
        pitch_shift_semitones: int=DEFAULT_AUGMENT_PITCH_SHIFT_SEMITONES,
        band_stop_prob: float=DEFAULT_AUGMENT_BAND_STOP_PROB,
        colored_noise_prob: float=DEFAULT_AUGMENT_COLORED_NOISE_PROB,
        colored_noise_min_snr_db: float=DEFAULT_AUGMENT_COLORED_NOISE_MIN_SNR_DB,
        colored_noise_max_snr_db: float=DEFAULT_AUGMENT_COLORED_NOISE_MAX_SNR_DB,
        colored_noise_min_f_decay: float=DEFAULT_AUGMENT_COLORED_NOISE_MIN_F_DECAY,
        colored_noise_max_f_decay: float=DEFAULT_AUGMENT_COLORED_NOISE_MAX_F_DECAY,
        background_noise_prob: float=DEFAULT_AUGMENT_BACKGROUND_NOISE_PROB,
        background_noise_min_snr_db: float=DEFAULT_AUGMENT_BACKGROUND_NOISE_MIN_SNR_DB,
        background_noise_max_snr_db: float=DEFAULT_AUGMENT_BACKGROUND_NOISE_MAX_SNR_DB,
        gain_prob: float=DEFAULT_AUGMENT_GAIN_PROB,
        reverb_prob: float=DEFAULT_AUGMENT_REVERB_PROB,
    ) -> None:
        """
        Initializes the augmented audio generator.
        """
        import audiomentations # type: ignore[import-untyped]
        import torch_audiomentations # type: ignore[import-untyped]
        super().__init__(device_id=device_id)
        self.source_dataset = source_dataset
        self.impulse_response_dataset = impulse_response_dataset
        self.augmentation_dataset = augmentation_dataset
        self.target_length = target_length
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.reverb_prob = reverb_prob

        self.background_noise_prob = background_noise_prob
        self.background_noise_min_snr_db = background_noise_min_snr_db
        self.background_noise_max_snr_db = background_noise_max_snr_db
        self.lock = Lock()

        # Check datasets
        if self.background_noise_prob > 0 and not self.augmentation_dataset:
            raise ValueError(f"Background noise is enabled but no augmentation dataset is provided")
        if self.reverb_prob > 0 and not self.impulse_response_dataset:
            raise ValueError(f"Reverb is enabled but no impulse response dataset is provided")

        # Non-batchable
        self.augment = audiomentations.Compose([
            audiomentations.SevenBandParametricEQ(
                p=seven_band_aug_prob,
                min_gain_db=-seven_band_aug_gain_db,
                max_gain_db=seven_band_aug_gain_db,
            ),
            audiomentations.TanhDistortion(
                p=tanh_distortion_prob,
                min_distortion=tanh_min_distortion,
                max_distortion=tanh_max_distortion,
            ),
        ])

        # Batchable
        self.augment_batch = torch_audiomentations.Compose([
            torch_audiomentations.PitchShift(
                min_transpose_semitones=-pitch_shift_semitones,
                max_transpose_semitones=pitch_shift_semitones,
                p=pitch_shift_prob,
                sample_rate=sample_rate,
                mode="per_batch",
                output_type="tensor"
            ),
            torch_audiomentations.BandStopFilter(
                p=band_stop_prob,
                mode="per_batch",
                output_type="tensor"
            ),
            torch_audiomentations.AddColoredNoise(
                min_snr_in_db=colored_noise_min_snr_db,
                max_snr_in_db=colored_noise_max_snr_db,
                min_f_decay=colored_noise_min_f_decay,
                max_f_decay=colored_noise_max_f_decay,
                p=colored_noise_prob,
                mode="per_batch",
                output_type="tensor"
            ),
            torch_audiomentations.Gain(
                p=gain_prob,
                mode="per_batch",
                output_type="tensor"
            ),
        ], output_type="tensor")

    @property
    def target_num_samples(self) -> int:
        """
        Returns the target number of samples based on the target length and sample rate.
        """
        return int(self.target_length * self.sample_rate)

    def get_resample(self, sample_rate: int) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Returns a resampling function that resamples audio to the target sample rate.
        """
        import torchaudio # type: ignore[import-untyped]
        if not hasattr(self, "_resample"):
            self._resample = {}

        if sample_rate not in self._resample:
            if sample_rate == self.sample_rate:
                self._resample[sample_rate] = lambda x: x
            else:
                self._resample[sample_rate] = torchaudio.transforms.Resample(
                    sample_rate,
                    self.sample_rate,
                ).to(self.device)
        return self._resample[sample_rate]

    def get_next_dataset_value(self, name: str, shuffle_first: bool=True) -> Any:
        """
        Returns the next value in a dataset that is an attribute of the generator.
        """
        dataset = getattr(self, name, None)
        assert dataset is not None, f"Dataset {name} is not defined"

        iterator_name = f"{name}_iterator"
        if not hasattr(self, iterator_name):
            setattr(self, iterator_name, iter(dataset.shuffle()) if shuffle_first else iter(dataset))
        try:
            return next(getattr(self, iterator_name))
        except StopIteration:
            setattr(self, iterator_name, iter(dataset.shuffle()))
            return next(getattr(self, iterator_name))

    def get_next_audio_dataset_waveform(self, name: str, shuffle_first: bool=False) -> torch.Tensor:
        """
        Returns the next audio waveform in a dataset.
        """
        import torch
        next_dataset_dict = self.get_next_dataset_value(name, shuffle_first)
        assert "audio" in next_dataset_dict, "Dataset must have an 'audio' key"

        data = torch.from_numpy(next_dataset_dict["audio"]["array"]).to(self.device, dtype=torch.float32)
        sample_rate = int(next_dataset_dict["audio"]["sampling_rate"])
        return self.get_resample(sample_rate)(data)

    def get_next_audio_sample(self) -> torch.Tensor:
        """
        Returns the next audio sample in the dataset.
        """
        return self.get_next_audio_dataset_waveform("source_dataset", False)

    def get_next_audio_sample_dict(self) -> Dict[str, Any]:
        """
        Returns the next audio sample in the dataset as a dictionary.
        """
        return self.get_next_dataset_value("source_dataset", False) # type: ignore[no-any-return]

    def get_next_impulse_response(self) -> torch.Tensor:
        """
        Returns the next impulse response sample.
        """
        return self.get_next_audio_dataset_waveform("impulse_response_dataset")

    def get_next_background_noise(self) -> torch.Tensor:
        """
        Returns the next background noise sample.
        """
        return self.get_next_audio_dataset_waveform("augmentation_dataset")

    def to_target_length(
        self,
        audio: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """
        Clips the audio to the target length.
        """
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        num_samples = audio.shape[0]
        target_num_samples = self.target_num_samples
        if num_samples >= target_num_samples:
            return audio[:target_num_samples]

        total_silence = target_num_samples - num_samples
        if total_silence == 1:
            # Extremely rare but possible, the calculation below will fail
            # if this is the case, just add a single frame to the end
            silence_before = 0
            silence_after = 1
        else:
            silence_before = np.random.randint(
                int(total_silence / 4),
                int(3 * total_silence / 4)
            )
            silence_after = total_silence - silence_before

        return np.concatenate([ # type: ignore[no-any-return]
            np.zeros(silence_before),
            audio,
            np.zeros(silence_after)
        ]).astype(np.float32)

    def add_background_noise_to_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Adds background noise to a batch of audio samples.

        The background noise is randomly selected from the augmentation dataset.
        """
        import torch
        import torchaudio
        batch_length = batch.shape[0] * batch.shape[1] # Total number of samples
        batch_noises: List[torch.Tensor] = []
        batch_noise_length = 0

        while batch_noise_length < batch_length:
            noise = self.get_next_background_noise()
            noise_length = noise.shape[0]
            batch_noises.append(noise)
            batch_noise_length += noise_length

        background_tensor = torch.split(
            torch.cat(batch_noises)[:batch_length],
            batch.shape[1]
        )

        background_tensor = torch.stack([ # type: ignore[assignment]
            torch.cat([
                tensor,
                torch.zeros(
                    (batch.shape[1] - tensor.shape[0],),
                    dtype=tensor.dtype,
                    device=tensor.device
                )
            ])
            for tensor in background_tensor
        ])

        background_noise_snr_db_range = self.background_noise_max_snr_db - self.background_noise_min_snr_db
        snr = torch.rand(batch.shape[0], dtype=batch.dtype, device=self.device) * background_noise_snr_db_range + self.background_noise_min_snr_db

        return torchaudio.functional.add_noise( # type: ignore[no-any-return]
            waveform=batch,
            noise=background_tensor,
            snr=snr.clone().detach(),
        )

    def to_audio_array(
        self,
        audio: Union[np.ndarray[Any, Any], List[Any], List[List[Any]]]
    ) -> np.ndarray[Any, Any]:
        """
        Ensures that the audio is a numpy array.
        """
        if isinstance(audio, list):
            if not audio:
                raise ValueError("Audio list is empty")
            if isinstance(audio[0], list):
                is_float = isinstance(audio[0][0], float)
            else:
                is_float = isinstance(audio[0], float)
            if is_float:
                return np.array(audio, dtype=np.float32)
            return np.array(audio, dtype=np.int16)
        return audio

    def execute_augment_batch(self, batch: Dataset) -> torch.Tensor:
        """
        Augments a batch of audio samples.
        """
        import torch
        import torchaudio
        augmented_clips: Dict[int, torch.Tensor] = {}

        if self.target_length is None:
            # Set target length to median length of the batch
            self.target_length = sum([
                len(audio["array"]) / audio["sampling_rate"]
                for audio in batch
            ]) / len(batch)

        # Gather clips that need to be resampled
        resample_dicts: Dict[int, Dict[int, np.ndarray[Any, Any]]] = {}
        for i, audio in enumerate(batch):
            samples = self.to_audio_array(audio["array"])
            sample_rate = audio["sampling_rate"]

            if sample_rate != self.sample_rate:
                # Batch clips that need to be resampled
                if sample_rate not in resample_dicts:
                    resample_dicts[sample_rate] = {}
                resample_dicts[sample_rate][i] = samples
            else:
                # Otherwise, augment the clip now
                augmented_clips[i] = self.augment(
                    self.to_target_length(samples),
                    sample_rate=sample_rate
                )

        # If any clips need to be resampled, do so, then augment
        for sample_rate, resample_clips in resample_dicts.items():
            logger.info(f"Resampling {len(resample_clips)} clips to {self.sample_rate} Hz from {sample_rate} Hz")
            resample = torchaudio.transforms.Resample(
                sample_rate,
                self.sample_rate
            ).to(self.device)
            # To batch this, all the clips need to be the same length
            max_clip_len = max([
                resample_clips[i].shape[0]
                for i in resample_clips.keys()
            ])
            # Stack and pad the clips
            stacked_clips = torch.vstack([
                torch.concat([
                    torch.from_numpy(resample_clips[i]),
                    torch.zeros(
                        (max_clip_len - resample_clips[i].shape[0],),
                        dtype=torch.float32
                    )
                ])
                for i in resample_clips.keys()
            ]).unsqueeze(1).to(self.device, dtype=torch.float32)
            # Now resample the batch
            stacked_clips = resample(stacked_clips).squeeze(axis=1)
            for i, resampled_clip in zip(resample_clips.keys(), stacked_clips):
                # Finally, augment the resampled clip
                augmented_clips[i] = self.augment(
                    self.to_target_length(resampled_clip.numpy()),
                    sample_rate=self.sample_rate
                )

        # Perform batchable augmentations
        augmented_batch = torch.vstack([
            torch.from_numpy(augmented_clips[i])
            for i in range(len(augmented_clips))
        ]).unsqueeze(1).to(self.device, dtype=torch.float32)

        try:
            augmented_batch = self.augment_batch(
                augmented_batch,
                sample_rate=self.sample_rate,
            ).squeeze(axis=1)
        except KeyError:
            # For some reason, torch-audiomentations occasionally doesn't
            # randomize the `snr_in_db` parameter before this gets called.
            # This is a workaround to try again.
            augmented_batch = self.augment_batch(
                augmented_batch,
                sample_rate=self.sample_rate,
            ).squeeze(axis=1)

        # Perform background noise augmentation
        if np.random.rand() < self.background_noise_prob and self.augmentation_dataset:
            augmented_batch = self.add_background_noise_to_batch(augmented_batch)

        # Perform reverb
        if np.random.rand() < self.reverb_prob and self.impulse_response_dataset:
            from speechbrain.processing.signal_processing import reverberate # type: ignore[import-untyped]
            augmented_batch = reverberate(
                augmented_batch,
                self.get_next_impulse_response(),
            )

        return augmented_batch

    def __call__(
        self,
        num_samples: int,
        **kwargs: Any
    ) -> Iterator[Dict[str, Any]]:
        """
        Generates audio samples.
        """
        import torch
        total_batches = int(np.ceil(num_samples / self.batch_size))
        with torch.no_grad():
            for i in range(total_batches):
                batch_samples = min(self.batch_size, num_samples - i * self.batch_size)
                batch_items = [
                    self.get_next_audio_sample_dict()
                    for _ in range(batch_samples)
                ]
                batch_results = self.execute_augment_batch([
                    batch_item["audio"] for batch_item in batch_items
                ])
                for audio, batch_item in zip(batch_results, batch_items):
                    yield {
                        "audio": {
                            "array": audio.cpu().numpy(),
                            "sampling_rate": self.sample_rate
                        },
                        **{
                            k: v
                            for k, v in batch_item.items()
                            if k != "audio"
                        }
                    }
