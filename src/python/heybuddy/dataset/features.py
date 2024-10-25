from __future__ import annotations

import os
import re
import gc
import math
import psutil
import scipy # type: ignore[import-untyped]
import tqdm
import random
import numpy as np

from typing import Optional, Union, Tuple, List, Any, TYPE_CHECKING
from concurrent.futures import ProcessPoolExecutor

from heybuddy.util import safe_name, logger
from heybuddy.constants import *
from heybuddy.embeddings import SpeechEmbeddings, get_speech_embeddings
from heybuddy.dataset.piper import PiperSpeechGenerator
from heybuddy.dataset.augmented import AugmentedAudioGenerator
from heybuddy.dataset.precalculated import PrecalculatedDatasetIterator

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
SupplementalDatasetType = Optional[Union[str, List[str], Tuple[str, ...]]]

if TYPE_CHECKING:
    import torch
    from datasets import Dataset # type: ignore[import-untyped]

class TrainingFeaturesGenerator:
    """
    Generate a dataset of features.
    """
    def __init__(
        self,
        # Common parameters
        device_id: Optional[int]=None,
        use_tqdm: bool=True,
        use_autoconfigure: bool=True,
        sample_rate: int=16000,
        sample_batch_size: int=DEFAULT_FEATURE_BATCH_SIZE, # Maximum number of samples to generate at once, about 20 gb of memory
        # TTS parameters
        tts_text: str="Hello, world!",
        tts_additional_texts: List[str]=[],
        tts_adversarial: bool=False,
        tts_adversarial_num_phrases: int=DEFAULT_ADVERSARIAL_PHRASES,
        tts_adversarial_custom_phrases: List[str]=[],
        tts_batch_size: int=DEFAULT_TTS_BATCH_SIZE,
        tts_phrase_augment_prob: float=DEFAULT_AUGMENT_PHRASE_PROB,
        tts_phrase_augment_words: List[str]=DEFAULT_AUGMENT_PHRASE_WORDS,
        # Augmentation parameters
        augment_target_length: float=1.44,
        augment_batch_size: int=DEFAULT_AUGMENT_BATCH_SIZE,
        augment_sample_ratio: float=DEFAULT_AUGMENT_SAMPLE_RATIO, # ratio of samples to augmented samples, only used if tts_num_samples is None
        augment_dataset_streaming: bool=False,
        augment_background_dataset: SupplementalDatasetType=None,
        augment_impulse_dataset: SupplementalDatasetType=None,
        augment_seven_band_prob: float=DEFAULT_AUGMENT_SEVEN_BAND_PROB,
        augment_seven_band_gain_db: float=DEFAULT_AUGMENT_SEVEN_BAND_GAIN_DB,
        augment_tanh_distortion_prob: float=DEFAULT_AUGMENT_TANH_DISTORTION_PROB,
        augment_tanh_min_distortion: float=DEFAULT_AUGMENT_TANH_MIN_DISTORTION,
        augment_tanh_max_distortion: float=DEFAULT_AUGMENT_TANH_MAX_DISTORTION,
        augment_pitch_shift_prob: float=DEFAULT_AUGMENT_PITCH_SHIFT_PROB,
        augment_pitch_shift_semitones: int=DEFAULT_AUGMENT_PITCH_SHIFT_SEMITONES,
        augment_band_stop_prob: float=DEFAULT_AUGMENT_BAND_STOP_PROB,
        augment_colored_noise_prob: float=DEFAULT_AUGMENT_COLORED_NOISE_PROB,
        augment_colored_noise_min_snr_db: float=DEFAULT_AUGMENT_COLORED_NOISE_MIN_SNR_DB,
        augment_colored_noise_max_snr_db: float=DEFAULT_AUGMENT_COLORED_NOISE_MAX_SNR_DB,
        augment_colored_noise_min_f_decay: float=DEFAULT_AUGMENT_COLORED_NOISE_MIN_F_DECAY,
        augment_colored_noise_max_f_decay: float=DEFAULT_AUGMENT_COLORED_NOISE_MAX_F_DECAY,
        augment_background_noise_prob: float=DEFAULT_AUGMENT_BACKGROUND_NOISE_PROB,
        augment_background_noise_min_snr_db: float=DEFAULT_AUGMENT_BACKGROUND_NOISE_MIN_SNR_DB,
        augment_background_noise_max_snr_db: float=DEFAULT_AUGMENT_BACKGROUND_NOISE_MAX_SNR_DB,
        augment_gain_prob: float=DEFAULT_AUGMENT_GAIN_PROB,
        augment_reverb_prob: float=DEFAULT_AUGMENT_REVERB_PROB,
        # Embedding parameters
        embedding_spectrogram_batch_size: int=DEFAULT_EMBEDDING_SPECTROGRAM_BATCH_SIZE,
        embedding_batch_size: int=DEFAULT_EMBEDDING_BATCH_SIZE,
    ) -> None:
        """
        :param device_id: Optional, the device id to use for processing
        :param use_tqdm: Optional, whether to use tqdm for progress bars
        :param use_autoconfigure: Optional, whether to automatically configure the generator based on available resources
        :param sample_rate: Optional, the sample rate to use for audio processing
        :param sample_batch_size: Optional, the maximum number of samples to generate at once
        :param tts_text: Optional, the wake phrase to use for TTS
        :param tts_additional_texts: Optional, additional wake phrases to use for TTS
        :param tts_adversarial: Optional, whether to generate adversarial samples
        :param tts_adversarial_num_phrases: Optional, the number of adversarial phrases to generate
        :param tts_adversarial_custom_phrases: Optional, custom adversarial phrases to use for TTS
        :param tts_batch_size: Optional, the batch size to use for TTS
        :param tts_phrase_augment_prob: Optional, the probability to augment phrases
        :param tts_phrase_augment_words: Optional, the words to use for phrase augmentation
        :param augment_target_length: Optional, the target length for augmented samples
        :param augment_batch_size: Optional, the batch size to use for augmentation
        :param augment_sample_ratio: Optional, the ratio of samples to augmented samples
        :param augment_dataset_streaming: Optional, whether to stream the dataset
        :param augment_background_dataset: Optional, the background dataset to use for augmentation
        :param augment_impulse_dataset: Optional, the impulse dataset to use for augmentation
        :param augment_seven_band_prob: Optional, the probability to apply seven band augmentation
        :param augment_seven_band_gain_db: Optional, the gain to apply for seven band augmentation
        :param augment_tanh_distortion_prob: Optional, the probability to apply tanh distortion
        :param augment_tanh_min_distortion: Optional, the minimum distortion to apply for tanh distortion
        :param augment_tanh_max_distortion: Optional, the maximum distortion to apply for tanh distortion
        :param augment_pitch_shift_prob: Optional, the probability to apply pitch shift
        :param augment_pitch_shift_semitones: Optional, the semitones to apply for pitch shift
        :param augment_band_stop_prob: Optional, the probability to apply band stop
        :param augment_colored_noise_prob: Optional, the probability to apply colored noise
        :param augment_colored_noise_min_snr_db: Optional, the minimum SNR to apply for colored noise
        :param augment_colored_noise_max_snr_db: Optional, the maximum SNR to apply for colored noise
        :param augment_colored_noise_min_f_decay: Optional, the minimum frequency decay to apply for colored noise
        :param augment_colored_noise_max_f_decay: Optional, the maximum frequency decay to apply for colored noise
        :param augment_background_noise_prob: Optional, the probability to apply background noise
        :param augment_background_noise_min_snr_db: Optional, the minimum SNR to apply for background noise
        :param augment_background_noise_max_snr_db: Optional, the maximum SNR to apply for background noise
        :param augment_gain_prob: Optional, the probability to apply gain
        :param augment_reverb_prob: Optional, the probability to apply reverb
        :param embedding_spectrogram_batch_size: Optional, the batch size to use for embedding spectrograms
        :param embedding_batch_size: Optional, the batch size to use for embeddings
        """
        self.device_id = device_id
        self.use_autoconfigure = use_autoconfigure
        self.use_tqdm = use_tqdm
        self.sample_rate = sample_rate
        self.sample_batch_size = sample_batch_size
        self.tts_text = tts_text
        self.tts_additional_texts = tts_additional_texts
        self.tts_adversarial = tts_adversarial
        self.tts_adversarial_num_phrases = tts_adversarial_num_phrases
        self.tts_adversarial_custom_phrases = tts_adversarial_custom_phrases
        self.tts_batch_size = tts_batch_size
        self.tts_phrase_augment_prob = tts_phrase_augment_prob
        self.tts_phrase_augment_words = tts_phrase_augment_words
        self.augment_target_length = augment_target_length
        self.augment_sample_ratio = augment_sample_ratio
        self.augment_batch_size = augment_batch_size
        self.augment_dataset_streaming = augment_dataset_streaming
        self.augment_background_dataset = augment_background_dataset
        self.augment_impulse_dataset = augment_impulse_dataset
        self.augment_seven_band_prob = augment_seven_band_prob
        self.augment_seven_band_gain_db = augment_seven_band_gain_db
        self.augment_tanh_distortion_prob = augment_tanh_distortion_prob
        self.augment_tanh_min_distortion = augment_tanh_min_distortion
        self.augment_tanh_max_distortion = augment_tanh_max_distortion
        self.augment_pitch_shift_prob = augment_pitch_shift_prob
        self.augment_pitch_shift_semitones = augment_pitch_shift_semitones
        self.augment_band_stop_prob = augment_band_stop_prob
        self.augment_colored_noise_prob = augment_colored_noise_prob
        self.augment_colored_noise_min_snr_db = augment_colored_noise_min_snr_db
        self.augment_colored_noise_max_snr_db = augment_colored_noise_max_snr_db
        self.augment_colored_noise_min_f_decay = augment_colored_noise_min_f_decay
        self.augment_colored_noise_max_f_decay = augment_colored_noise_max_f_decay
        self.augment_background_noise_prob = augment_background_noise_prob
        self.augment_background_noise_min_snr_db = augment_background_noise_min_snr_db
        self.augment_background_noise_max_snr_db = augment_background_noise_max_snr_db
        self.augment_gain_prob = augment_gain_prob
        self.augment_reverb_prob = augment_reverb_prob
        self.embedding_spectrogram_batch_size = embedding_spectrogram_batch_size
        self.embedding_batch_size = embedding_batch_size

    @property
    def device(self) -> torch.device:
        """
        Returns the device.
        """
        import torch
        if self.device_id is None:
            return torch.device("cpu")
        return torch.device(f"cuda:{self.device_id}")

    def autoconfigure(self) -> None:
        """
        Autoconfigures the generator based on available resources.
        """
        import torch
        if torch.cuda.is_available():
            self.device_id = torch.cuda.current_device()
            total_memory_gib = torch.cuda.get_device_properties(self.device).total_memory / (2<<29)

            if total_memory_gib >= 8:
                self.tts_batch_size = 64
                self.augment_batch_size = 128
                self.embedding_spectrogram_batch_size = 128
                self.embedding_batch_size = 128
            elif total_memory_gib >= 4:
                self.tts_batch_size = 32
                self.augment_batch_size = 64
                self.embedding_spectrogram_batch_size = 64
                self.embedding_batch_size = 64
            elif total_memory_gib >= 2:
                self.tts_batch_size = 8
                self.augment_batch_size = 16
                self.embedding_spectrogram_batch_size = 32
                self.embedding_batch_size = 32
            else:
                self.tts_batch_size = 2
                self.augment_batch_size = 4
                self.embedding_spectrogram_batch_size = 16
                self.embedding_batch_size = 16
        else:
            self.device_id = None
            total_memory_gib = psutil.virtual_memory().total / (2<<29)

            if total_memory_gib >= 16:
                self.tts_batch_size = 32
                self.augment_batch_size = 64
                self.embedding_spectrogram_batch_size = 64
                self.embedding_batch_size = 64
            elif total_memory_gib >= 8:
                self.tts_batch_size = 16
                self.augment_batch_size = 32
                self.embedding_spectrogram_batch_size = 32
                self.embedding_batch_size = 32
            else:
                self.tts_batch_size = 8
                self.augment_batch_size = 16
                self.embedding_spectrogram_batch_size = 16
                self.embedding_batch_size = 16

    def get_speech_embeddings_model(self) -> SpeechEmbeddings:
        """
        Returns the speech embeddings model.
        """
        return get_speech_embeddings(device_id=self.device_id)

    def get_tts_generator(self) -> PiperSpeechGenerator:
        """
        Returns a generator for TTS.
        """
        return PiperSpeechGenerator(
            phrase=self.tts_text,
            adversarial=self.tts_adversarial,
            num_adversarial_texts=self.tts_adversarial_num_phrases,
            additional_phrases=self.tts_additional_texts,
            custom_adversarial_texts=self.tts_adversarial_custom_phrases,
            phrase_augment_prob=self.tts_phrase_augment_prob,
            phrase_augment_words=self.tts_phrase_augment_words,
            batch_size=self.tts_batch_size,
            target_sample_rate=self.sample_rate,
            device_id=self.device_id,
        )

    def get_audio_dataset_from_path(
        self,
        name: str,
        split: Optional[str]=None,
    ) -> Dataset:
        """
        Returns a dataset from a path.

        If the path is a huggingface repository (user/dataset), it will
        use `load_dataset`. If the path is a folder in the local filesystem,
        will use an audio folder dataset.

        :param name: name of the dataset
        :param split: optional split
        :return: the dataset
        """
        from datasets import load_dataset, Dataset, Audio
        if re.match(r"^[a-z0-9\-]+/[a-z0-9\-]+$", name):
            try:
                ds = load_dataset(name, split=split, streaming=self.augment_dataset_streaming)
            except ValueError:
                if split is not None:
                    logger.warning(f"Failed to load dataset '{name}' with split '{split}', trying with 'train'")
                    ds = load_dataset(name, split="train", streaming=self.augment_dataset_streaming)
                else:
                    raise
            ds = ds.remove_columns([c for c in ds.column_names if c != "audio"])
            return ds
        if os.path.isdir(name):
            # Find all audio files recursively in directory
            audio_files = []
            for root, _, files in os.walk(name):
                for file in files:
                    path, ext = os.path.splitext(file)
                    if ext in SUPPORTED_AUDIO_EXTENSIONS:
                        audio_files.append(os.path.join(root, file))
            return Dataset.from_dict({"audio": audio_files}).cast_column("audio", Audio())
        raise ValueError(f"Invalid dataset path: {name}")

    def get_impulse_dataset(self, testing: bool=False) -> Optional[Dataset]:
        """
        Returns the impulse dataset.
        """
        from datasets import interleave_datasets
        if self.augment_impulse_dataset is None:
            return None
        split = "test" if testing else "train"
        if isinstance(self.augment_impulse_dataset, list) or isinstance(self.augment_impulse_dataset, tuple):
            dataset_list = [
                self.get_audio_dataset_from_path(i, split)
                for i in self.augment_impulse_dataset
            ]
            if len(dataset_list) == 0:
                return None
            elif len(dataset_list) == 1:
                return dataset_list[0]
            else:
                return interleave_datasets(dataset_list)
        return self.get_audio_dataset_from_path(self.augment_impulse_dataset, split)

    def get_background_dataset(self, testing: bool=False) -> Optional[Dataset]:
        """
        Returns the background dataset.
        """
        from datasets import interleave_datasets
        if self.augment_background_dataset is None:
            return None
        split = "test" if testing else "train"
        if isinstance(self.augment_background_dataset, list) or isinstance(self.augment_background_dataset, tuple):
            dataset_list = [
                self.get_audio_dataset_from_path(i, split)
                for i in self.augment_background_dataset
            ]
            if len(dataset_list) == 0:
                return None
            elif len(dataset_list) == 1:
                return dataset_list[0]
            else:
                return interleave_datasets(dataset_list)
        return self.get_audio_dataset_from_path(self.augment_background_dataset, split)

    def get_augmented_generator(
        self,
        dataset: Dataset,
        testing: bool=False,
    ) -> AugmentedAudioGenerator:
        """
        Returns a generator for augmentation.
        """
        return AugmentedAudioGenerator(
            dataset,
            batch_size=self.augment_batch_size,
            target_length=self.augment_target_length,
            sample_rate=self.sample_rate,
            augmentation_dataset=self.get_background_dataset(testing),
            impulse_response_dataset=self.get_impulse_dataset(testing),
            seven_band_aug_prob=self.augment_seven_band_prob,
            seven_band_aug_gain_db=self.augment_seven_band_gain_db,
            tanh_distortion_prob=self.augment_tanh_distortion_prob,
            tanh_min_distortion=self.augment_tanh_min_distortion,
            tanh_max_distortion=self.augment_tanh_max_distortion,
            pitch_shift_prob=self.augment_pitch_shift_prob,
            pitch_shift_semitones=self.augment_pitch_shift_semitones,
            band_stop_prob=self.augment_band_stop_prob,
            colored_noise_prob=self.augment_colored_noise_prob,
            colored_noise_min_snr_db=self.augment_colored_noise_min_snr_db,
            colored_noise_max_snr_db=self.augment_colored_noise_max_snr_db,
            colored_noise_min_f_decay=self.augment_colored_noise_min_f_decay,
            colored_noise_max_f_decay=self.augment_colored_noise_max_f_decay,
            background_noise_prob=self.augment_background_noise_prob,
            background_noise_min_snr_db=self.augment_background_noise_min_snr_db,
            background_noise_max_snr_db=self.augment_background_noise_max_snr_db,
            gain_prob=self.augment_gain_prob,
            reverb_prob=self.augment_reverb_prob,
            device_id=self.device_id,
        )

    def generate(
        self,
        num_samples: int,
        sample_save_path: Optional[str]=None,
        augmented_sample_save_path: Optional[str]=None,
        testing: bool=False,
        validation: bool=False,
    ) -> np.ndarray[Any, Any]:
        """
        Generates samples and calculates embeddings.
        """
        from datasets import Dataset
        import torch

        with torch.no_grad():
            if self.use_autoconfigure:
                self.autoconfigure()

            if validation:
                tts_num_samples = num_samples
            else:
                tts_num_samples = int(num_samples // self.augment_sample_ratio)
                tts_num_samples = max(1, min(num_samples, tts_num_samples))

            type_label = f"{'adversarial ' if self.tts_adversarial else ''}{'validation ' if validation else 'testing ' if testing else ''}"
            tts_generator = self.get_tts_generator()(tts_num_samples)

            if self.use_tqdm:
                tts_iterator = tqdm.tqdm(
                    tts_generator,
                    total=tts_num_samples,
                    mininterval=1.0,
                    desc=f"Generating {type_label}samples"
                )
            else:
                tts_iterator = tts_generator # type: ignore[assignment]

            logger.debug(f"Generating {tts_num_samples} {type_label}samples for wake phrase '{self.tts_text}'")
            samples = [
                i["audio"] for i in tts_iterator
            ]
            if sample_save_path is not None:
                scipy.io.wavfile.write(
                    sample_save_path,
                    self.sample_rate,
                    random.choice(samples)["array"],
                )

            # Recover memory
            del tts_generator
            gc.collect()
            torch.cuda.empty_cache()

            if validation:
                # Just pad to target length
                for i, sample in enumerate(samples):
                    pad_frames = int(self.sample_rate * self.augment_target_length) - len(sample["array"])
                    if pad_frames > 0:
                        pad_start = int(pad_frames // 2)
                        pad_end = pad_frames - pad_start
                        samples[i] = np.pad(
                            sample["array"],
                            (pad_start, pad_end),
                            "constant",
                            constant_values=0
                        )
                    else:
                        samples[i] = sample["array"]
            else:
                # Generate augmented samples for embedding, just keep np arrays
                dataset = Dataset.from_dict({"audio": samples})
                augmented_generator = self.get_augmented_generator(
                    dataset,
                    testing=testing
                )(num_samples)
                if self.use_tqdm:
                    augmented_iterator = tqdm.tqdm(
                        augmented_generator,
                        total=num_samples,
                        mininterval=1.0,
                        desc=f"Generating augmented {type_label}samples"
                    )
                else:
                    augmented_iterator = augmented_generator # type: ignore[assignment]

                samples = [
                    i["audio"]["array"] for i in augmented_iterator
                ]
                if augmented_sample_save_path is not None:
                    scipy.io.wavfile.write(
                        augmented_sample_save_path,
                        self.sample_rate,
                        random.choice(samples),
                    )

                # Recover memory
                del augmented_generator
                gc.collect()
                torch.cuda.empty_cache()

            # Calculate embeddings
            if self.use_tqdm:
                num_embeddings_per_window = num_samples * 4
                num_windows = 4

                embedding_pbar = tqdm.tqdm(
                    desc=f"Generating {type_label}embeddings",
                    total=num_embeddings_per_window * num_windows,
                    mininterval=1.0,
                )

                num_windows_complete = 0
                last_step = 0

                def on_embedding_progress(i: int, total: int) -> None:
                    nonlocal num_windows_complete, last_step
                    if i < last_step:
                        num_windows_complete += 1

                    embedding_pbar.n = (i + num_windows_complete * num_embeddings_per_window)
                    embedding_pbar.refresh()
                    last_step = i
            else:
                on_embedding_progress = None # type: ignore[assignment]

            return self.get_speech_embeddings_model()( # type: ignore[return-value]
                samples,
                spectrogram_batch_size=self.embedding_spectrogram_batch_size,
                embedding_batch_size=self.embedding_batch_size,
                on_embedding_progress=on_embedding_progress
            )

    def __call__(
        self,
        num_samples: int,
        sample_save_path: Optional[str]=None,
        augmented_sample_save_path: Optional[str]=None,
        testing: bool=False,
        validation: bool=False,
    ) -> np.ndarray[Any, Any]:
        """
        Generates samples and calculates embeddings.

        :param num_samples: number of samples to generate
        :param sample_save_path: Optional, a path to save a representative sample for debugging
        :param augmented_sample_save_path: Optional, a path to save a representative augmented sample for debugging
        """
        sample_batches = math.ceil(num_samples / self.sample_batch_size)
        sample_batch_remaining = num_samples % self.sample_batch_size
        sample_batch_sizes = [self.sample_batch_size] * sample_batches
        if sample_batch_remaining > 0:
            sample_batch_sizes[-1] = sample_batch_remaining

        sample_feature_batches = []
        batch_iterator = tqdm.tqdm(sample_batch_sizes, desc=f"Generating {'testing ' if testing else 'validation ' if validation else ''}features", unit="batch") if self.use_tqdm else sample_batch_sizes

        # STOP! Before you swap this `for` and `with` thinking you will make it more efficient, know that
        # it is purposefully kept this way to let the subprocess completely exit before proceeding to the next
        # batch. This is due to one thing; `pt_main_thread`. This thread will happily gobble up all the memory
        # on the machine, and eventually crash the system. We have not imported PyTorch yet at this point in
        # runtime, so the parent process is not associated with the main thread of the PyTorch process. This
        # is purposeful so the pytorch process can exit without taking the parent process with it.
        for sample_batch_size in batch_iterator: # type: ignore[attr-defined]
            with ProcessPoolExecutor() as executor:
                future = executor.submit(
                    self.generate,
                    sample_batch_size,
                    sample_save_path,
                    augmented_sample_save_path,
                    testing,
                    validation,
                )
                sample_feature_batches.append(future.result())
            if sample_batches == 1:
                return sample_feature_batches[0]
        return np.concatenate(sample_feature_batches)

    @classmethod
    def default(
        cls,
        wake_phrase: str,
        adversarial: bool=False,
        num_adversarial_phrases: int=10,
        additional_wake_phrases: List[str]=[],
        custom_adversarial_phrases: List[str]=[],
        dataset_streaming: bool=False,
        tts_batch_size: int=DEFAULT_TTS_BATCH_SIZE,
        phrase_augment_prob: float=DEFAULT_AUGMENT_PHRASE_PROB,
        phrase_augment_words: List[str]=DEFAULT_AUGMENT_PHRASE_WORDS,
        # Augmentation parameters
        augment_target_length: float=1.44,
        augment_batch_size: int=DEFAULT_AUGMENT_BATCH_SIZE,
        augment_sample_ratio: float=DEFAULT_AUGMENT_SAMPLE_RATIO, # ratio of samples to augmented samples, only used if tts_num_samples is None
        augment_dataset_streaming: bool=False,
        augment_background_dataset: SupplementalDatasetType=DEFAULT_BACKGROUND_DATASET,
        augment_impulse_dataset: SupplementalDatasetType=DEFAULT_IMPULSE_DATASET,
        augment_seven_band_prob: float=DEFAULT_AUGMENT_SEVEN_BAND_PROB,
        augment_seven_band_gain_db: float=DEFAULT_AUGMENT_SEVEN_BAND_GAIN_DB,
        augment_tanh_distortion_prob: float=DEFAULT_AUGMENT_TANH_DISTORTION_PROB,
        augment_tanh_min_distortion: float=DEFAULT_AUGMENT_TANH_MIN_DISTORTION,
        augment_tanh_max_distortion: float=DEFAULT_AUGMENT_TANH_MAX_DISTORTION,
        augment_pitch_shift_prob: float=DEFAULT_AUGMENT_PITCH_SHIFT_PROB,
        augment_pitch_shift_semitones: int=DEFAULT_AUGMENT_PITCH_SHIFT_SEMITONES,
        augment_band_stop_prob: float=DEFAULT_AUGMENT_BAND_STOP_PROB,
        augment_colored_noise_prob: float=DEFAULT_AUGMENT_COLORED_NOISE_PROB,
        augment_colored_noise_min_snr_db: float=DEFAULT_AUGMENT_COLORED_NOISE_MIN_SNR_DB,
        augment_colored_noise_max_snr_db: float=DEFAULT_AUGMENT_COLORED_NOISE_MAX_SNR_DB,
        augment_colored_noise_min_f_decay: float=DEFAULT_AUGMENT_COLORED_NOISE_MIN_F_DECAY,
        augment_colored_noise_max_f_decay: float=DEFAULT_AUGMENT_COLORED_NOISE_MAX_F_DECAY,
        augment_background_noise_prob: float=DEFAULT_AUGMENT_BACKGROUND_NOISE_PROB,
        augment_background_noise_min_snr_db: float=DEFAULT_AUGMENT_BACKGROUND_NOISE_MIN_SNR_DB,
        augment_background_noise_max_snr_db: float=DEFAULT_AUGMENT_BACKGROUND_NOISE_MAX_SNR_DB,
        augment_gain_prob: float=DEFAULT_AUGMENT_GAIN_PROB,
        augment_reverb_prob: float=DEFAULT_AUGMENT_REVERB_PROB,
        # Embedding parameters
        embedding_spectrogram_batch_size: int=DEFAULT_EMBEDDING_SPECTROGRAM_BATCH_SIZE,
        embedding_batch_size: int=DEFAULT_EMBEDDING_BATCH_SIZE,
    ) -> TrainingFeaturesGenerator:
        """
        Returns a default dataset generator.
        """
        return cls(
            use_autoconfigure=True,
            tts_text=wake_phrase,
            tts_adversarial=adversarial,
            tts_batch_size=tts_batch_size,
            tts_adversarial_num_phrases=num_adversarial_phrases,
            tts_adversarial_custom_phrases=custom_adversarial_phrases,
            tts_additional_texts=additional_wake_phrases,
            tts_phrase_augment_prob=phrase_augment_prob,
            tts_phrase_augment_words=phrase_augment_words,
            augment_background_dataset=augment_background_dataset,
            augment_impulse_dataset=augment_impulse_dataset,
            augment_dataset_streaming=augment_dataset_streaming,
            augment_target_length=augment_target_length,
            augment_batch_size=augment_batch_size,
            augment_sample_ratio=augment_sample_ratio,
            augment_seven_band_prob=augment_seven_band_prob,
            augment_seven_band_gain_db=augment_seven_band_gain_db,
            augment_tanh_distortion_prob=augment_tanh_distortion_prob,
            augment_tanh_min_distortion=augment_tanh_min_distortion,
            augment_tanh_max_distortion=augment_tanh_max_distortion,
            augment_pitch_shift_prob=augment_pitch_shift_prob,
            augment_pitch_shift_semitones=augment_pitch_shift_semitones,
            augment_band_stop_prob=augment_band_stop_prob,
            augment_colored_noise_prob=augment_colored_noise_prob,
            augment_colored_noise_min_snr_db=augment_colored_noise_min_snr_db,
            augment_colored_noise_max_snr_db=augment_colored_noise_max_snr_db,
            augment_colored_noise_min_f_decay=augment_colored_noise_min_f_decay,
            augment_colored_noise_max_f_decay=augment_colored_noise_max_f_decay,
            augment_background_noise_prob=augment_background_noise_prob,
            augment_background_noise_min_snr_db=augment_background_noise_min_snr_db,
            augment_background_noise_max_snr_db=augment_background_noise_max_snr_db,
            augment_gain_prob=augment_gain_prob,
            augment_reverb_prob=augment_reverb_prob,
            embedding_spectrogram_batch_size=embedding_spectrogram_batch_size,
            embedding_batch_size=embedding_batch_size,
        )

    @classmethod
    def get_wake_phrase_file_name(cls, wake_phrase: str, testing: bool=False) -> str:
        """
        Returns the file name for a given wake phrase without extension.
        """
        wake_phrase = safe_name(wake_phrase)
        return wake_phrase.strip("_") + ("_tst" if testing else "")

    @classmethod
    def get_training_features(
        cls,
        wake_phrase: str,
        num_positive_samples: int,
        num_adversarial_samples: int,
        num_adversarial_phrases: int=10,
        additional_wake_phrases: List[str]=[],
        custom_adversarial_phrases: List[str]=[],
        testing: bool=False,
        use_cache: bool=True,
        save_samples: bool=True,
        keep_in_memory: bool=False,
        dataset_streaming: bool=False,
        tts_batch_size: int=DEFAULT_TTS_BATCH_SIZE,
        phrase_augment_prob: float=DEFAULT_AUGMENT_PHRASE_PROB,
        phrase_augment_words: List[str]=DEFAULT_AUGMENT_PHRASE_WORDS,
        # Augmentation parameters
        augment_target_length: float=1.44,
        augment_batch_size: int=DEFAULT_AUGMENT_BATCH_SIZE,
        augment_sample_ratio: float=DEFAULT_AUGMENT_SAMPLE_RATIO, # ratio of samples to augmented samples, only used if tts_num_samples is None
        augment_dataset_streaming: bool=False,
        augment_background_dataset: SupplementalDatasetType=DEFAULT_BACKGROUND_DATASET,
        augment_impulse_dataset: SupplementalDatasetType=DEFAULT_IMPULSE_DATASET,
        augment_seven_band_prob: float=DEFAULT_AUGMENT_SEVEN_BAND_PROB,
        augment_seven_band_gain_db: float=DEFAULT_AUGMENT_SEVEN_BAND_GAIN_DB,
        augment_tanh_distortion_prob: float=DEFAULT_AUGMENT_TANH_DISTORTION_PROB,
        augment_tanh_min_distortion: float=DEFAULT_AUGMENT_TANH_MIN_DISTORTION,
        augment_tanh_max_distortion: float=DEFAULT_AUGMENT_TANH_MAX_DISTORTION,
        augment_pitch_shift_prob: float=DEFAULT_AUGMENT_PITCH_SHIFT_PROB,
        augment_pitch_shift_semitones: int=DEFAULT_AUGMENT_PITCH_SHIFT_SEMITONES,
        augment_band_stop_prob: float=DEFAULT_AUGMENT_BAND_STOP_PROB,
        augment_colored_noise_prob: float=DEFAULT_AUGMENT_COLORED_NOISE_PROB,
        augment_colored_noise_min_snr_db: float=DEFAULT_AUGMENT_COLORED_NOISE_MIN_SNR_DB,
        augment_colored_noise_max_snr_db: float=DEFAULT_AUGMENT_COLORED_NOISE_MAX_SNR_DB,
        augment_colored_noise_min_f_decay: float=DEFAULT_AUGMENT_COLORED_NOISE_MIN_F_DECAY,
        augment_colored_noise_max_f_decay: float=DEFAULT_AUGMENT_COLORED_NOISE_MAX_F_DECAY,
        augment_background_noise_prob: float=DEFAULT_AUGMENT_BACKGROUND_NOISE_PROB,
        augment_background_noise_min_snr_db: float=DEFAULT_AUGMENT_BACKGROUND_NOISE_MIN_SNR_DB,
        augment_background_noise_max_snr_db: float=DEFAULT_AUGMENT_BACKGROUND_NOISE_MAX_SNR_DB,
        augment_gain_prob: float=DEFAULT_AUGMENT_GAIN_PROB,
        augment_reverb_prob: float=DEFAULT_AUGMENT_REVERB_PROB,
        # Embedding parameters
        embedding_spectrogram_batch_size: int=DEFAULT_EMBEDDING_SPECTROGRAM_BATCH_SIZE,
        embedding_batch_size: int=DEFAULT_EMBEDDING_BATCH_SIZE,
    ) -> Tuple[
        PrecalculatedDatasetIterator,
        PrecalculatedDatasetIterator
    ]:
        """
        Generates optimally configured training features for a given wake phrase and number of steps.
        Leverages caching to reduce redundant computation.
        """
        name = cls.get_wake_phrase_file_name(wake_phrase, testing=testing)
        adversarial_name = f"{name}_adv"

        existing_embeddings = 0
        existing_adversarial_embeddings = 0

        if use_cache:
            try:
                existing_dataset = PrecalculatedDatasetIterator(name)
                existing_embeddings = len(existing_dataset)
            except FileNotFoundError:
                pass

            try:
                existing_adversarial_dataset = PrecalculatedDatasetIterator(adversarial_name)
                existing_adversarial_embeddings = len(existing_adversarial_dataset)
            except FileNotFoundError:
                pass

            if existing_embeddings >= num_positive_samples and existing_adversarial_embeddings >= num_adversarial_samples:
                # No need to generate new embeddings
                return existing_dataset, existing_adversarial_dataset

        # Generate positive if needed
        if existing_embeddings < num_positive_samples:
            logger.debug(f"Generating {num_positive_samples - existing_embeddings} positive samples for wake phrase '{wake_phrase}'")
            positive = cls.default(
                wake_phrase,
                additional_wake_phrases=additional_wake_phrases,
                dataset_streaming=dataset_streaming,
                phrase_augment_prob=phrase_augment_prob,
                phrase_augment_words=phrase_augment_words,
                tts_batch_size=tts_batch_size,
                augment_background_dataset=augment_background_dataset,
                augment_impulse_dataset=augment_impulse_dataset,
                augment_dataset_streaming=augment_dataset_streaming,
                augment_target_length=augment_target_length,
                augment_batch_size=augment_batch_size,
                augment_sample_ratio=augment_sample_ratio,
                augment_seven_band_prob=augment_seven_band_prob,
                augment_seven_band_gain_db=augment_seven_band_gain_db,
                augment_tanh_distortion_prob=augment_tanh_distortion_prob,
                augment_tanh_min_distortion=augment_tanh_min_distortion,
                augment_tanh_max_distortion=augment_tanh_max_distortion,
                augment_pitch_shift_prob=augment_pitch_shift_prob,
                augment_pitch_shift_semitones=augment_pitch_shift_semitones,
                augment_band_stop_prob=augment_band_stop_prob,
                augment_colored_noise_prob=augment_colored_noise_prob,
                augment_colored_noise_min_snr_db=augment_colored_noise_min_snr_db,
                augment_colored_noise_max_snr_db=augment_colored_noise_max_snr_db,
                augment_colored_noise_min_f_decay=augment_colored_noise_min_f_decay,
                augment_colored_noise_max_f_decay=augment_colored_noise_max_f_decay,
                augment_background_noise_prob=augment_background_noise_prob,
                augment_background_noise_min_snr_db=augment_background_noise_min_snr_db,
                augment_background_noise_max_snr_db=augment_background_noise_max_snr_db,
                augment_gain_prob=augment_gain_prob,
                augment_reverb_prob=augment_reverb_prob,
                embedding_spectrogram_batch_size=embedding_spectrogram_batch_size,
                embedding_batch_size=embedding_batch_size,
            )
            if existing_embeddings > 0:
                positive_features = np.concatenate([
                    existing_dataset.precalculated,
                    positive(
                        num_positive_samples - existing_embeddings,
                        testing=testing
                    )
                ])
            else:
                positive_features = positive(
                    num_positive_samples,
                    sample_save_path=f"{name}.wav" if save_samples else None,
                    augmented_sample_save_path=f"{name}_augmented.wav" if save_samples else None,
                    testing=testing,
                )
            del positive
            positive_iterator = PrecalculatedDatasetIterator.from_array(
                positive_features,
                name=name,
                keep_in_memory=keep_in_memory
            )
            if not keep_in_memory:
                del positive_features
                gc.collect()
        else:
            positive_iterator = existing_dataset

        # Clear generator
        gc.collect()

        # Generate adversarial if needed
        if existing_adversarial_embeddings < num_adversarial_samples:
            adversarial = cls.default(
                wake_phrase,
                adversarial=True,
                additional_wake_phrases=additional_wake_phrases,
                num_adversarial_phrases=num_adversarial_phrases,
                custom_adversarial_phrases=custom_adversarial_phrases,
                dataset_streaming=dataset_streaming,
                phrase_augment_prob=phrase_augment_prob,
                phrase_augment_words=phrase_augment_words,
                tts_batch_size=tts_batch_size,
                augment_background_dataset=augment_background_dataset,
                augment_impulse_dataset=augment_impulse_dataset,
                augment_dataset_streaming=augment_dataset_streaming,
                augment_target_length=augment_target_length,
                augment_batch_size=augment_batch_size,
                augment_sample_ratio=augment_sample_ratio,
                augment_seven_band_prob=augment_seven_band_prob,
                augment_seven_band_gain_db=augment_seven_band_gain_db,
                augment_tanh_distortion_prob=augment_tanh_distortion_prob,
                augment_tanh_min_distortion=augment_tanh_min_distortion,
                augment_tanh_max_distortion=augment_tanh_max_distortion,
                augment_pitch_shift_prob=augment_pitch_shift_prob,
                augment_pitch_shift_semitones=augment_pitch_shift_semitones,
                augment_band_stop_prob=augment_band_stop_prob,
                augment_colored_noise_prob=augment_colored_noise_prob,
                augment_colored_noise_min_snr_db=augment_colored_noise_min_snr_db,
                augment_colored_noise_max_snr_db=augment_colored_noise_max_snr_db,
                augment_colored_noise_min_f_decay=augment_colored_noise_min_f_decay,
                augment_colored_noise_max_f_decay=augment_colored_noise_max_f_decay,
                augment_background_noise_prob=augment_background_noise_prob,
                augment_background_noise_min_snr_db=augment_background_noise_min_snr_db,
                augment_background_noise_max_snr_db=augment_background_noise_max_snr_db,
                augment_gain_prob=augment_gain_prob,
                augment_reverb_prob=augment_reverb_prob,
                embedding_spectrogram_batch_size=embedding_spectrogram_batch_size,
                embedding_batch_size=embedding_batch_size,
            )
            if existing_adversarial_embeddings > 0:
                adversarial_features = np.concatenate([
                    existing_adversarial_dataset.precalculated,
                    adversarial(
                        num_adversarial_samples - existing_adversarial_embeddings,
                        testing=testing
                    )
                ])
            else:
                adversarial_features = adversarial(
                    num_adversarial_samples,
                    sample_save_path=f"{adversarial_name}.wav" if save_samples else None,
                    augmented_sample_save_path=f"{adversarial_name}_augmented.wav" if save_samples else None,
                    testing=testing,
                )
            del adversarial
            adversarial_iterator = PrecalculatedDatasetIterator.from_array(
                adversarial_features,
                name=adversarial_name,
                keep_in_memory=keep_in_memory
            )
            if not keep_in_memory:
                del adversarial_features
                gc.collect()
        else:
            adversarial_iterator = existing_adversarial_dataset

        gc.collect()
        return positive_iterator, adversarial_iterator

    @classmethod
    def get_validation_features(
        cls,
        wake_phrase: str,
        num_positive_samples: int,
        tts_batch_size: int=DEFAULT_TTS_BATCH_SIZE,
        phrase_augment_prob: float=DEFAULT_AUGMENT_PHRASE_PROB,
        phrase_augment_words: List[str]=DEFAULT_AUGMENT_PHRASE_WORDS,
        use_cache: bool=True,
        save_samples: bool=True,
        keep_in_memory: bool=False,
        augment_target_length: float=1.44,
    ) -> PrecalculatedDatasetIterator:
        """
        Generates optimally configured validation features for a given wake phrase and number of steps.
        """
        name = cls.get_wake_phrase_file_name(wake_phrase, testing=False)
        name = f"{name}_val"

        existing_embeddings = 0

        if use_cache:
            try:
                existing_dataset = PrecalculatedDatasetIterator(name)
                existing_embeddings = len(existing_dataset)
            except FileNotFoundError:
                pass

            if existing_embeddings >= num_positive_samples:
                # No need to generate new embeddings
                return existing_dataset

        # Generate positive if needed
        if existing_embeddings < num_positive_samples:
            positive = cls.default(
                wake_phrase,
                phrase_augment_prob=phrase_augment_prob,
                phrase_augment_words=phrase_augment_words,
                tts_batch_size=tts_batch_size,
                augment_target_length=augment_target_length,
            )
            if existing_embeddings > 0:
                positive_features = np.concatenate([
                    existing_dataset.precalculated,
                    positive(
                        num_positive_samples - existing_embeddings,
                        validation=True
                    )
                ])
            else:
                positive_features = positive(
                    num_positive_samples,
                    sample_save_path=f"{name}.wav" if save_samples else None,
                    validation=True,
                )
            del positive
            positive_iterator = PrecalculatedDatasetIterator.from_array(
                positive_features,
                name=name,
                keep_in_memory=keep_in_memory
            )
            if not keep_in_memory:
                del positive_features
                gc.collect()
        else:
            positive_iterator = existing_dataset

        # Clear generator
        gc.collect()
        return positive_iterator
