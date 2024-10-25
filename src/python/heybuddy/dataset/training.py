from __future__ import annotations

import os
import numpy as np

from typing import Iterator, List, Tuple, Optional, Any, Dict, Union, TYPE_CHECKING
from queue import Queue, Empty
from threading import Thread, Event

from heybuddy.constants import *
from heybuddy.dataset.precalculated import (
    PrecalculatedDatasetIterator,
    PrecalculatedTrainingDatasetLarge,
    PrecalculatedTrainingDatasetMedium,
    PrecalculatedValidationDataset,
)
from heybuddy.dataset.features import TrainingFeaturesGenerator
from heybuddy.util import logger

if TYPE_CHECKING:
    import torch

__all__ = [
    "TrainingDatasetIterator",
    "WakeWordTrainingDatasetIterator",
]

SupplementalDatasetType = Optional[Union[str, List[str], Tuple[str, ...]]]
class TrainingDatasetIterator:
    """
    A generator for training datasets.

    Takes a number of positive and negative datasets,
    then batches them together to create a training dataset.
    """
    threads: List[Tuple[Thread, Event]]
    queue: Queue[Tuple[torch.Tensor, torch.Tensor]]

    def __init__(
        self,
        max_samples: Optional[int] = None,
        num_batch_threads: int=2,
        max_queued_batches: int=100,
        start: bool=True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the generator with the given datasets.

        :param max_samples: The maximum number of samples to yield.
        :param num_batch_threads: The number of threads to use for batch generation.
        :param max_queued_batches: The maximum number of batches to queue.
        :param start: Whether to start the batch generation threads.
        """
        self.total_yielded_samples = 0
        self.max_samples = max_samples
        self.num_batch_threads = num_batch_threads
        self.threads = []
        self.queue = Queue(max_queued_batches)
        self.started = False
        if start:
            self.start()

    def metadata(self) -> Dict[str, Any]:
        """
        Get metadata for the training dataset.
        """
        return {
            "max_samples": self.max_samples,
            "num_batch_threads": self.num_batch_threads,
        }

    def start(self) -> None:
        """
        Start the batch generation threads.
        """
        if self.started:
            return
        self.started = True
        logger.info(f"Starting batch generation with {self.num_batch_threads} threads")
        for _ in range(self.num_batch_threads):
            stop_event = Event()
            thread = Thread(target=self._generate_batches, args=(stop_event,))
            thread.daemon = True
            thread.start()
            self.threads.append((thread, stop_event))

    def check_restart(self) -> None:
        """
        Restart the batch generation threads if they have stopped.
        """
        if self.started:
            for i, (thread, event) in enumerate(self.threads):
                if not thread.is_alive():
                    logger.warning(f"Batch generation thread {i} has stopped, restarting")
                    event.set()
                    thread.join()
                    event.clear()
                    self.threads[i] = (Thread(target=self._generate_batches, args=(event,)), event)
                    self.threads[i][0].daemon = True
                    self.threads[i][0].start()
        else:
            self.start()

    def stop(self) -> None:
        """
        Stop the batch generation threads.
        """
        for _, stop_event in self.threads:
            stop_event.set()
        for thread, _ in self.threads:
            thread.join()
        self.threads.clear()
        with self.queue.mutex:
            self.queue.queue.clear()
        self.started = False

    def iterate(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        """
        Iterate over the training datasets.
        """
        yielded_samples = 0
        while True:
            try:
                yield self.queue.get(timeout=1)
                yielded_samples += 1
                self.total_yielded_samples += 1
                if self.max_samples is not None and yielded_samples >= self.max_samples:
                    break
                if self.total_yielded_samples % 10 == 0:
                    self.check_restart()
            except Empty:
                self.check_restart()

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        """
        Iterate over the training datasets.
        """
        return self.iterate()

    def _generate_batches(self, stop_event: Event) -> None:
        """
        Generate batches and add them to the queue.
        """
        raise NotImplementedError("Subclasses must implement this method")

class WakeWordTrainingDatasetIterator(TrainingDatasetIterator):
    """
    A generator for wake word training datasets.
    """
    def __init__(
        self,
        max_samples: Optional[int] = None,
        num_batch_threads: int=2,
        max_queued_batches: int=100,
        start: bool=True,
        positive: List[Tuple[PrecalculatedDatasetIterator, int]] = [],
        negative: List[Tuple[PrecalculatedDatasetIterator, int]] = [],
    ) -> None:
        """
        Initialize the generator with the given datasets.

        :param max_samples: The maximum number of samples to yield.
        :param num_batch_threads: The number of threads to use for batch generation.
        :param max_queued_batches: The maximum number of batches to queue.
        :param start: Whether to start the batch generation threads.
        :param positive: A list of positive datasets and the number of samples to take from each.
        :param negative: A list of negative datasets and the number of samples to take from each.
        """
        super().__init__(
            max_samples=max_samples,
            num_batch_threads=num_batch_threads,
            max_queued_batches=max_queued_batches,
            start=start,
        )
        assert positive or negative, "At least one positive or negative dataset is required"
        self.positive = positive
        self.negative = negative

    def metadata(self) -> Dict[str, Any]:
        """
        Get metadata for the training dataset.
        """
        return {
            **super().metadata(),
            "positive": [{
                "length": len(dataset),
                "batch_size": batch_size,
                "metadata": None if not isinstance(dataset, PrecalculatedDatasetIterator) else dataset.metadata()
            } for dataset, batch_size in self.positive],
            "negative": [{
                "length": len(dataset),
                "batch_size": batch_size,
                "metadata": None if not isinstance(dataset, PrecalculatedDatasetIterator) else dataset.metadata()
            } for dataset, batch_size in self.negative],
        }

    def summary(self) -> str:
        """
        Summarizes total samples yielded.
        """
        lines = [f"Total batches yielded: {self.total_yielded_samples}"]
        for i, (dataset, batch_size) in enumerate(self.positive):
            samples_taken = dataset.total_taken
            samples_unique = len(dataset)
            seen_rate = samples_taken / samples_unique
            lines.append(f"Positive dataset {i+1}: {samples_taken} samples taken out of {samples_unique} unique samples ({batch_size} per batch, {seen_rate:.2%} seen)")
        for i, (dataset, batch_size) in enumerate(self.negative):
            samples_taken = dataset.total_taken
            samples_unique = len(dataset)
            seen_rate = samples_taken / samples_unique
            lines.append(f"Negative dataset {i+1}: {samples_taken} samples taken out of {samples_unique} unique samples ({batch_size} per batch, {seen_rate:.2%} seen)")
        return "\n".join(lines)

    def multiply_batch_size(self, ratio: float) -> None:
        """
        Multiply the number of samples in each dataset by the given ratio.
        """
        restart = self.started
        if self.started:
            self.stop()
        self.positive = [
            (dataset, max(1, int(num_samples * ratio)))
            for dataset, num_samples in self.positive
        ]
        self.negative = [
            (dataset, max(1, int(num_samples * ratio)))
            for dataset, num_samples in self.negative
        ]
        if restart:
            self.start()

    def half_batch_size(self) -> None:
        """
        Halve the number of samples in each dataset.
        """
        self.multiply_batch_size(0.5)

    def double_batch_size(self) -> None:
        """
        Double the number of samples in each dataset.
        """
        self.multiply_batch_size(2)

    def _generate_batches(self, stop_event: Event) -> None:
        """
        Generate batches and add them to the queue.
        """
        import torch
        samples_batch = []
        labels_batch = []

        while not stop_event.is_set():
            for positive_dataset, num_samples in self.positive:
                samples_batch.append(positive_dataset.take(num_samples))
                labels_batch.append(np.ones(num_samples))
            for negative_dataset, num_samples in self.negative:
                samples_batch.append(negative_dataset.take(num_samples))
                labels_batch.append(np.zeros(num_samples))

            x = torch.from_numpy(np.concatenate(samples_batch))
            y = torch.from_numpy(np.concatenate(labels_batch).astype(np.int64))

            if x.shape[0] != y.shape[0]:
                logger.warning(f"Shapes of x and y do not match: {x.shape} != {y.shape}")
                min_len = min(x.shape[0], y.shape[0])
                x = x[:min_len]
                y = y[:min_len]

            while self.queue.full():
                if stop_event.is_set():
                    return
                stop_event.wait(0.1)

            self.queue.put((x, y))
            samples_batch.clear()
            labels_batch.clear()

    @classmethod
    def default(
        cls,
        wake_phrase: str,
        additional_wake_phrases: List[str]=[],
        num_positive_samples: int=DEFAULT_POSITIVE_SAMPLES,
        num_adversarial_phrases: int=DEFAULT_ADVERSARIAL_PHRASES,
        custom_adversarial_phrases: List[str]=[],
        num_adversarial_samples: int=DEFAULT_ADVERSARIAL_SAMPLES,
        positive_per_batch: int=DEFAULT_POSITIVE_BATCH_SIZE,
        negative_per_batch: int=DEFAULT_NEGATIVE_BATCH_SIZE,
        adversarial_per_batch: int=DEFAULT_ADVERSARIAL_BATCH_SIZE,
        use_cache: bool=True,
        dataset_streaming: bool=False,
        num_batch_threads: int=DEFAULT_BATCH_THREADS,
        start: bool=True,
        large_training: bool=True,
        medium_training: bool=True,
        custom_training: Optional[str]=None,
        phrase_augment_prob: float=DEFAULT_AUGMENT_PHRASE_PROB,
        phrase_augment_words: List[str]=DEFAULT_AUGMENT_PHRASE_WORDS,
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
    ) -> TrainingDatasetIterator:
        """
        Return the default training dataset generator.

        :param wake_phrase: The wake phrase to use for training.
        :param num_positive_samples: The number of positive samples to generate for training.
        :param num_adversarial_samples: The number of negative samples to generate for training.
        :param num_adversarial_phrases: The number of adversarial phrases to generate for training.
        :param custom_adversarial_phrases: A list of custom adversarial phrases to use for training. Optional.
        :param positive_per_batch: The number of positive samples to include in each batch.
        :param negative_per_batch: The number of negative samples to include in each batch.
        :param adversarial_per_batch: The number of adversarial negative samples to include in each batch.
        :param use_cache: Whether to use the cache for training features.
        :param dataset_streaming: Whether to use dataset streaming for training features.
        :param num_batch_threads: The number of threads to use for batch generation.
        :param start: Whether to start the batch generation threads.
        :param large_training: Whether to include the default large training dataset. Not mutually exclusive with other options.
        :param medium_training: Whether to include the default medium training dataset. Not mutually exclusive with other options.
        :param custom_training: A path to a custom training dataset. Not mutually exclusive with other options.
        :param phrase_augment_prob: The probability of augmenting the wake phrase.
        :param phrase_augment_words: A list of words to use for augmenting the wake phrase.
        :param augment_dataset_streaming: Whether to use dataset streaming for augmentation.
        :param augment_background_dataset: The background dataset to use for augmentation.
        :param augment_impulse_dataset: The impulse dataset to use for augmentation.
        :param augment_seven_band_prob: The probability of applying seven band equalization.
        :param augment_seven_band_gain_db: The gain to apply to the seven band equalization.
        :param augment_tanh_distortion_prob: The probability of applying tanh distortion.
        :param augment_tanh_min_distortion: The minimum distortion to apply with tanh distortion.
        :param augment_tanh_max_distortion: The maximum distortion to apply with tanh distortion.
        :param augment_pitch_shift_prob: The probability of applying pitch shifting.
        :param augment_pitch_shift_semitones: The number of semitones to shift the pitch.
        :param augment_band_stop_prob: The probability of applying band stop filtering.
        :param augment_colored_noise_prob: The probability of applying colored noise.
        :param augment_colored_noise_min_snr_db: The minimum SNR of the colored noise.
        :param augment_colored_noise_max_snr_db: The maximum SNR of the colored noise.
        :param augment_colored_noise_min_f_decay: The minimum frequency decay of the colored noise.
        :param augment_colored_noise_max_f_decay: The maximum frequency decay of the colored noise.
        :param augment_background_noise_prob: The probability of applying background noise.
        :param augment_background_noise_min_snr_db: The minimum SNR of the background noise.
        :param augment_background_noise_max_snr_db: The maximum SNR of the background noise.
        :param augment_gain_prob: The probability of applying gain.
        :param augment_reverb_prob: The probability of applying reverb.
        """
        positive_features, adversarial_features = TrainingFeaturesGenerator.get_training_features(
            wake_phrase=wake_phrase,
            additional_wake_phrases=additional_wake_phrases,
            num_positive_samples=num_positive_samples,
            num_adversarial_samples=num_adversarial_samples,
            num_adversarial_phrases=num_adversarial_phrases,
            custom_adversarial_phrases=custom_adversarial_phrases,
            use_cache=use_cache,
            dataset_streaming=dataset_streaming,
            phrase_augment_prob=phrase_augment_prob,
            phrase_augment_words=phrase_augment_words,
            augment_dataset_streaming=augment_dataset_streaming,
            augment_background_dataset=augment_background_dataset,
            augment_impulse_dataset=augment_impulse_dataset,
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
        )
        positive_list = [(positive_features, positive_per_batch)]
        negative_list = [(adversarial_features, adversarial_per_batch)]

        if additional_wake_phrases:
            for i, additional_wake_phrase in enumerate(additional_wake_phrases):
                additional_positive_features, additional_adversarial_features = TrainingFeaturesGenerator.get_training_features(
                    wake_phrase=additional_wake_phrase,
                    num_positive_samples=num_positive_samples,
                    num_adversarial_samples=num_adversarial_samples,
                    num_adversarial_phrases=num_adversarial_phrases,
                    use_cache=use_cache,
                    dataset_streaming=dataset_streaming,
                    phrase_augment_prob=phrase_augment_prob,
                    phrase_augment_words=phrase_augment_words,
                    augment_dataset_streaming=augment_dataset_streaming,
                    augment_background_dataset=augment_background_dataset,
                    augment_impulse_dataset=augment_impulse_dataset,
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
                )
                positive_list.append((additional_positive_features, positive_per_batch))
                negative_list.append((additional_adversarial_features, adversarial_per_batch))

        if large_training:
            this_per_batch = negative_per_batch if not medium_training else int(negative_per_batch * 2 / 3)
            negative_list.append(
                (
                    PrecalculatedTrainingDatasetLarge(exclude_phrase=wake_phrase),
                    this_per_batch
                )
            )
        if medium_training:
            this_per_batch = negative_per_batch if not large_training else (negative_per_batch - int(negative_per_batch * 2 / 3))
            negative_list.append(
                (
                    PrecalculatedTrainingDatasetMedium(exclude_phrase=wake_phrase),
                    this_per_batch
                )
            )
        if custom_training:
            negative_list.append(
                (
                    PrecalculatedDatasetIterator(
                        name=os.path.splitext(os.path.basename(custom_training))[0],
                        directory=os.path.dirname(custom_training),
                        exclude_phrase=wake_phrase,
                        labeled=True
                    ),
                    negative_per_batch
                )
            )
        return cls(
            positive=positive_list,
            negative=negative_list,
            num_batch_threads=num_batch_threads,
            start=start,
        )

    @classmethod
    def testing(
        cls,
        wake_phrase: str,
        additional_wake_phrases: List[str]=[],
        num_positive_samples: int=1000,
        num_adversarial_samples: int=1000,
        num_adversarial_phrases: int=10,
        custom_adversarial_phrases: List[str]=[],
        positive_per_batch: int=50,
        adversarial_per_batch: int=50,
        use_cache: bool=True,
        dataset_streaming: bool=False,
        num_batch_threads: int=1,
        start: bool=True,
        phrase_augment_prob: float=DEFAULT_AUGMENT_PHRASE_PROB,
        phrase_augment_words: List[str]=DEFAULT_AUGMENT_PHRASE_WORDS,
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
    ) -> TrainingDatasetIterator:
        """
        Return the testing dataset generator.

        :param wake_phrase: The wake phrase to use for training.
        :param num_positive_samples: The number of positive samples to generate for testing.
        :param num_adversarial_samples: The number of negative samples to generate for testing.
        :param num_adversarial_phrases: The number of adversarial phrases to generate for testing.
        :param custom_adversarial_phrases: A list of custom adversarial phrases to use for testing. Optional.
        :param positive_per_batch: The number of positive samples to include in each batch.
        :param adversarial_per_batch: The number of adversarial negative samples to include in each batch.
        :param use_cache: Whether to use the cache for training features.
        :param dataset_streaming: Whether to use dataset streaming for training features.
        :param num_batch_threads: The number of threads to use for batch generation.
        :param start: Whether to start the batch generation threads.
        :param phrase_augment_prob: The probability of augmenting the wake phrase.
        :param phrase_augment_words: A list of words to use for augmenting the wake phrase.
        :param augment_dataset_streaming: Whether to use dataset streaming for augmentation.
        :param augment_background_dataset: The background dataset to use for augmentation.
        :param augment_impulse_dataset: The impulse dataset to use for augmentation.
        :param augment_seven_band_prob: The probability of applying seven band equalization.
        :param augment_seven_band_gain_db: The gain to apply to the seven band equalization.
        :param augment_tanh_distortion_prob: The probability of applying tanh distortion.
        :param augment_tanh_min_distortion: The minimum distortion to apply with tanh distortion.
        :param augment_tanh_max_distortion: The maximum distortion to apply with tanh distortion.
        :param augment_pitch_shift_prob: The probability of applying pitch shifting.
        :param augment_pitch_shift_semitones: The number of semitones to shift the pitch.
        :param augment_band_stop_prob: The probability of applying band stop filtering.
        :param augment_colored_noise_prob: The probability of applying colored noise.
        :param augment_colored_noise_min_snr_db: The minimum SNR of the colored noise.
        :param augment_colored_noise_max_snr_db: The maximum SNR of the colored noise.
        :param augment_colored_noise_min_f_decay: The minimum frequency decay of the colored noise.
        :param augment_colored_noise_max_f_decay: The maximum frequency decay of the colored noise.
        :param augment_background_noise_prob: The probability of applying background noise.
        :param augment_background_noise_min_snr_db: The minimum SNR of the background noise.
        :param augment_background_noise_max_snr_db: The maximum SNR of the background noise.
        :param augment_gain_prob: The probability of applying gain.
        :param augment_reverb_prob: The probability of applying reverb.
        """
        positive_features, adversarial_features = TrainingFeaturesGenerator.get_training_features(
            wake_phrase=wake_phrase,
            num_positive_samples=num_positive_samples,
            num_adversarial_samples=num_adversarial_samples,
            num_adversarial_phrases=num_adversarial_phrases,
            custom_adversarial_phrases=custom_adversarial_phrases,
            use_cache=use_cache,
            dataset_streaming=dataset_streaming,
            testing=True,
            phrase_augment_prob=phrase_augment_prob,
            phrase_augment_words=phrase_augment_words,
            augment_dataset_streaming=augment_dataset_streaming,
            augment_background_dataset=augment_background_dataset,
            augment_impulse_dataset=augment_impulse_dataset,
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
        )
        positive_list = [(positive_features, positive_per_batch)]
        negative_list = [(adversarial_features, adversarial_per_batch)]

        if additional_wake_phrases:
            for i, additional_wake_phrase in enumerate(additional_wake_phrases):
                additional_positive_features, additional_adversarial_features = TrainingFeaturesGenerator.get_training_features(
                    wake_phrase=additional_wake_phrase,
                    num_positive_samples=num_positive_samples,
                    num_adversarial_samples=num_adversarial_samples,
                    num_adversarial_phrases=num_adversarial_phrases,
                    use_cache=use_cache,
                    dataset_streaming=dataset_streaming,
                    testing=True,
                    phrase_augment_prob=phrase_augment_prob,
                    phrase_augment_words=phrase_augment_words,
                    augment_dataset_streaming=augment_dataset_streaming,
                    augment_background_dataset=augment_background_dataset,
                    augment_impulse_dataset=augment_impulse_dataset,
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
                )
                positive_list.append((additional_positive_features, positive_per_batch))
                negative_list.append((additional_adversarial_features, adversarial_per_batch))

        return cls(
            positive=positive_list,
            negative=negative_list,
            max_samples=max([
                num_positive_samples // positive_per_batch,
                num_adversarial_samples // adversarial_per_batch
            ]),
            num_batch_threads=num_batch_threads,
            start=start,
        )

    @classmethod
    def validation(
        cls,
        wake_phrase: str,
        additional_wake_phrases: List[str]=[],
        positive_batch_size: int=50,
        negative_batch_size: int=1000,
        num_samples: int=1000,
        num_batch_threads: int=1,
        start: bool=True,
        precalculated_validation: bool=True,
        custom_validation: Optional[str]=None,
        phrase_augment_prob: float=DEFAULT_AUGMENT_PHRASE_PROB,
        phrase_augment_words: List[str]=DEFAULT_AUGMENT_PHRASE_WORDS,
    ) -> TrainingDatasetIterator:
        """
        Return the validation dataset generator.

        :param wake_phrase: The wake phrase to use for training.
        :param positive_batch_size: The number of positive samples to include in each batch.
        :param negative_batch_size: The number of negative samples to include in each batch.
        :param num_samples: The number of positive samples to generate for validation.
        :param num_batch_threads: The number of threads to use for batch generation.
        :param start: Whether to start the batch generation threads.
        :param precalculated_validation: Whether to include the default precalculated validation dataset.
        :param custom_validation: A path to a custom validation dataset.
        :param phrase_augment_prob: The probability of augmenting the wake phrase.
        :param phrase_augment_words: A list of words to use for augmenting the wake phrase.
        """
        negative_list: List[Tuple[PrecalculatedDatasetIterator, int]] = []
        if precalculated_validation:
            negative_list.append((PrecalculatedValidationDataset(exclude_phrase=wake_phrase), negative_batch_size))
        if custom_validation:
            negative_list.append(
                (
                    PrecalculatedDatasetIterator(
                        name=os.path.splitext(os.path.basename(custom_validation))[0],
                        directory=os.path.dirname(custom_validation),
                        exclude_phrase=wake_phrase,
                        labeled=True
                    ),
                    negative_batch_size
                )
            )
        validation_negative_feature_count = sum([len(dataset) for dataset, _ in negative_list])
        validation_positive_features = TrainingFeaturesGenerator.get_validation_features(
            wake_phrase=wake_phrase,
            num_positive_samples=num_samples,
        )
        positive_list = [(validation_positive_features, positive_batch_size)]

        if additional_wake_phrases:
            for i, additional_wake_phrase in enumerate(additional_wake_phrases):
                additional_positive_features = TrainingFeaturesGenerator.get_validation_features(
                    wake_phrase=additional_wake_phrase,
                    num_positive_samples=num_samples,
                    phrase_augment_prob=phrase_augment_prob,
                    phrase_augment_words=phrase_augment_words,
                )
                positive_list.append((additional_positive_features, positive_batch_size))

        return cls(
            positive=positive_list,
            negative=negative_list,
            num_batch_threads=num_batch_threads,
            max_samples=max([
                validation_negative_feature_count // negative_batch_size,
                len(validation_positive_features) // positive_batch_size,
            ]),
            start=start,
        )

    @classmethod
    def all(
        cls,
        wake_phrase: str,
        additional_wake_phrases: List[str]=[],
        num_positive_samples: int=100000,
        num_adversarial_samples: int=50000,
        num_adversarial_phrases: int=10,
        custom_adversarial_phrases: List[str]=[],
        positive_per_batch: int=50,
        negative_per_batch: int=1000,
        adversarial_per_batch: int=50,
        num_batch_threads: int=2,
        large_training: bool=True,
        medium_training: bool=True,
        custom_training: Optional[str]=None,
        validation_positive_batch_size: int=50,
        validation_negative_batch_size: int=1000,
        validation_num_positive_samples: int=1000,
        validation_num_batch_threads: int=1,
        validation_include_precalculated: bool=True,
        validation_custom: Optional[str]=None,
        testing_num_positive_samples: int=1000,
        testing_num_adversarial_samples: int=1000,
        testing_num_adversarial_phrases: int=10,
        testing_custom_adversarial_phrases: List[str]=[],
        testing_positive_per_batch: Optional[int]=None, # none = match positive_per_batch
        testing_adversarial_per_batch: Optional[int]=None, # none = match adversarial_per_batch,
        testing_num_batch_threads: int=1,
        dataset_streaming: bool=False,
        start: bool=True,
        phrase_augment_prob: float=DEFAULT_AUGMENT_PHRASE_PROB,
        phrase_augment_words: List[str]=DEFAULT_AUGMENT_PHRASE_WORDS,
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
    ) -> Tuple[TrainingDatasetIterator, TrainingDatasetIterator, TrainingDatasetIterator]:
        """
        Return the training, validation, and testing dataset generators.

        :param wake_phrase: The wake phrase to use for training.
        :param num_positive_samples: The number of positive samples to generate for training.
        :param num_adversarial_samples: The number of negative samples to generate for training.
        :param num_adversarial_phrases: The number of adversarial phrases to generate for training.
        :param custom_adversarial_phrases: A list of custom adversarial phrases to use for training. Optional.
        :param positive_per_batch: The number of positive samples to include in each batch.
        :param negative_per_batch: The number of negative samples to include in each batch.
        :param adversarial_per_batch: The number of adversarial negative samples to include in each batch.
        :param num_batch_threads: The number of threads to use for batch generation.
        :param large_training: Whether to include the default large training dataset. Not mutually exclusive with other options.
        :param medium_training: Whether to include the default medium training dataset. Not mutually exclusive with other options.
        :param custom_training: A path to a custom training dataset. Not mutually exclusive with other options.
        :param validation_positive_batch_size: The number of positive samples to include in each validation batch.
        :param validation_negative_batch_size: The number of negative samples to include in each validation batch.
        :param validation_num_positive_samples: The number of positive samples to generate for validation.
        :param validation_num_batch_threads: The number of threads to use for validation batch generation.
        :param validation_include_precalculated: Whether to include the default precalculated validation dataset.
        :param validation_custom: A path to a custom validation dataset.
        :param testing_num_positive_samples: The number of positive samples to generate for testing.
        :param testing_num_adversarial_samples: The number of negative samples to generate for testing.
        :param testing_num_adversarial_phrases: The number of adversarial phrases to generate for testing.
        :param testing_custom_adversarial_phrases: A list of custom adversarial phrases to use for testing. Optional.
        :param testing_positive_per_batch: The number of positive samples to include in each testing batch. If None, this
                                           defaults to the value of positive_per_batch.
        :param testing_adversarial_per_batch: The number of adversarial negative samples to include in each testing batch.
                                              If None, this defaults to the value of adversarial_per_batch.
        :param testing_num_batch_threads: The number of threads to use for testing batch generation.
        :param phrase_augment_prob: The probability of augmenting the wake phrase.
        :param phrase_augment_words: A list of words to use for augmenting the wake phrase.
        :param augment_dataset_streaming: Whether to use dataset streaming for augmentation.
        :param augment_background_dataset: The background dataset to use for augmentation.
        :param augment_impulse_dataset: The impulse dataset to use for augmentation.
        :param augment_seven_band_prob: The probability of applying seven band equalization.
        :param augment_seven_band_gain_db: The gain to apply to the seven band equalization.
        :param augment_tanh_distortion_prob: The probability of applying tanh distortion.
        :param augment_tanh_min_distortion: The minimum distortion to apply with tanh distortion.
        :param augment_tanh_max_distortion: The maximum distortion to apply with tanh distortion.
        :param augment_pitch_shift_prob: The probability of applying pitch shifting.
        :param augment_pitch_shift_semitones: The number of semitones to shift the pitch.
        :param augment_band_stop_prob: The probability of applying band stop filtering.
        :param augment_colored_noise_prob: The probability of applying colored noise.
        :param augment_colored_noise_min_snr_db: The minimum SNR of the colored noise.
        :param augment_colored_noise_max_snr_db: The maximum SNR of the colored noise.
        :param augment_colored_noise_min_f_decay: The minimum frequency decay of the colored noise.
        :param augment_colored_noise_max_f_decay: The maximum frequency decay of the colored noise.
        :param augment_background_noise_prob: The probability of applying background noise.
        :param augment_background_noise_min_snr_db: The minimum SNR of the background noise.
        :param augment_background_noise_max_snr_db: The maximum SNR of the background noise.
        :param augment_gain_prob: The probability of applying gain.
        :param augment_reverb_prob: The probability of applying reverb.
        :return: A tuple of training, validation, and testing dataset generators.
        """
        training = cls.default(
            wake_phrase=wake_phrase,
            additional_wake_phrases=additional_wake_phrases,
            num_positive_samples=num_positive_samples,
            num_adversarial_samples=num_adversarial_samples,
            num_adversarial_phrases=num_adversarial_phrases,
            custom_adversarial_phrases=custom_adversarial_phrases,
            positive_per_batch=positive_per_batch,
            negative_per_batch=negative_per_batch,
            adversarial_per_batch=adversarial_per_batch,
            dataset_streaming=dataset_streaming,
            num_batch_threads=num_batch_threads,
            start=False,
            large_training=large_training,
            medium_training=medium_training,
            custom_training=custom_training,
            phrase_augment_prob=phrase_augment_prob,
            phrase_augment_words=phrase_augment_words,
            augment_dataset_streaming=augment_dataset_streaming,
            augment_background_dataset=augment_background_dataset,
            augment_impulse_dataset=augment_impulse_dataset,
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
        )
        testing = cls.testing(
            wake_phrase=wake_phrase,
            additional_wake_phrases=additional_wake_phrases,
            num_positive_samples=testing_num_positive_samples,
            num_adversarial_samples=testing_num_adversarial_samples,
            num_adversarial_phrases=testing_num_adversarial_phrases,
            custom_adversarial_phrases=testing_custom_adversarial_phrases,
            positive_per_batch=testing_positive_per_batch or positive_per_batch,
            adversarial_per_batch=testing_adversarial_per_batch or adversarial_per_batch,
            dataset_streaming=dataset_streaming,
            num_batch_threads=testing_num_batch_threads,
            start=False,
            phrase_augment_prob=phrase_augment_prob,
            phrase_augment_words=phrase_augment_words,
            augment_dataset_streaming=augment_dataset_streaming,
            augment_background_dataset=augment_background_dataset,
            augment_impulse_dataset=augment_impulse_dataset,
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
        )
        validation = cls.validation(
            wake_phrase=wake_phrase,
            additional_wake_phrases=additional_wake_phrases,
            positive_batch_size=validation_positive_batch_size,
            negative_batch_size=validation_negative_batch_size,
            num_batch_threads=validation_num_batch_threads,
            num_samples=validation_num_positive_samples,
            start=False,
            precalculated_validation=validation_include_precalculated,
            custom_validation=validation_custom,
        )
        if start:
            training.start()
            validation.start()
            testing.start()
        return training, validation, testing
