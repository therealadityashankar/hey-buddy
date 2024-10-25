from __future__ import annotations

import os
import gc
import re
import numpy as np

from typing import Any, Iterator, Optional, Dict, List, Tuple, Union, TYPE_CHECKING
from typing_extensions import Self
from threading import Lock
from tqdm import tqdm
from math import ceil, log10

from heybuddy.constants import *
from heybuddy.util import (
    logger,
    check_download_file_to_dir,
    get_file_name_from_url,
)

if TYPE_CHECKING:
    from heybuddy.embeddings import SpeechEmbeddings
    from heybuddy.tokens import BERTTokenizer

__all__ = [
    "PrecalculatedLabeledTrainingDatasetGenerator",
    "PrecalculatedTrainingDatasetGenerator",
    "PrecalculatedDatasetIterator",
    "HostedPrecalculatedDatasetIterator",
    "PrecalculatedTrainingDatasetLarge",
    "PrecalculatedTrainingDatasetMedium",
    "PrecalculatedValidationDataset",
]

LOCAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "precalculated"))
os.makedirs(LOCAL_DIR, exist_ok=True)

SupplementalDatasetType = Optional[Union[str, List[str], Tuple[str, ...]]]

class PrecalculatedTrainingDatasetGenerator:
    """
    A generator for creating ltraining datasets.
    This should be used to take existing in-the-wild data and create packed datasets of features.
    """
    _last_text: str
    _last_tokens: np.ndarray[Any, Any]

    def __init__(
        self,
        dataset_path: str,
        config_name: Optional[str]=None,
        split: str="train",
        audio_key: str="audio",
        audio_array_key: Optional[str]="array",
        audio_sample_rate_key: Optional[str]="sampling_rate",
        device_id: Optional[int]=None,
        sample_rate: int=16000,
        seconds_per_batch: float=1.44,
        process_batch_size: int=128,
        embedding_batch_size: int=32,
    ) -> None:
        """
        :param dataset_path: The path to the dataset.
        :param config_name: The configuration name.
        :param split: The dataset split.
        :param audio_key: The key for the audio.
        :param audio_array_key: The key for the audio array.
        :param audio_sample_rate_key: The key for the audio sample rate.
        :param device_id: The device ID.
        :param sample_rate: The sample rate.
        :param seconds_per_batch: The number of seconds per batch.
        :param process_batch_size: The number of samples per batch.
        """
        self.dataset_path = dataset_path
        self.split = split
        self.config_name = config_name
        self.audio_key = audio_key
        self.audio_array_key = audio_array_key
        self.audio_sample_rate_key = audio_sample_rate_key
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.seconds_per_batch = seconds_per_batch
        self.process_batch_size = process_batch_size
        self.embedding_batch_size = embedding_batch_size

    @property
    def samples_per_batch(self) -> int:
        """
        Returns the number of samples per batch.
        """
        return int(self.sample_rate * self.seconds_per_batch)

    @property
    def speech_embeddings(self) -> SpeechEmbeddings:
        """
        Returns the speech embeddings model.
        """
        if not hasattr(self, "_speech_embeddings"):
            from heybuddy.embeddings import get_speech_embeddings
            self._speech_embeddings = get_speech_embeddings(device_id=self.device_id)
        return self._speech_embeddings

    def label_embeddings(
        self,
        embeddings: np.ndarray[Any, Any],
        batch: List[Tuple[np.ndarray[Any, Any], Dict[str, Any]]],
    ) -> np.ndarray[Any, Any]:
        """
        Adds any additional embeddings to the batch.
        Base class returns the embeddings as-is.
        """
        return embeddings

    def __call__(
        self,
        name: str,
        output_dir: str=LOCAL_DIR,
        max_hours: float=1000.0,
        dataset_streaming: bool=True,
        trust_remote_code: bool=False,
        samples_per_file: int=10000,
    ) -> None:
        """
        Creates a labeled training dataset.
        """
        from datasets import load_dataset # type: ignore[import-untyped,unused-ignore]
        import torch
        import torchaudio # type: ignore[import-untyped]
        import numpy as np

        output_dir = os.path.join(output_dir, name)
        os.makedirs(output_dir, exist_ok=True)

        dataset = load_dataset(
            self.dataset_path,
            self.config_name,
            split=self.split,
            streaming=dataset_streaming,
            trust_remote_code=trust_remote_code,
        )

        num_batches = 0
        max_batches = int(max_hours * 3600 / self.seconds_per_batch / self.process_batch_size)
        num_files = ceil((max_batches * self.process_batch_size) / samples_per_file)
        num_file_digits = int(log10(num_files)) + 1

        total_num_samples = max_batches * self.process_batch_size
        logger.info(f"Will generate up to {total_num_samples} samples from {max_hours} hours of data in \"{self.dataset_path}:{self.split}\". Writing {num_files} files to \"{output_dir}\".")

        progress = tqdm(total=max_batches, unit="batch", desc="Generating dataset")
        batch: List[Tuple[np.ndarray[Any, Any], Dict[str, Any]]] = []
        data_files: List[str] = []
        buffer: Optional[np.ndarray[Any, Any]] = None

        resamplers: Dict[int, torchaudio.transforms.Resample] = {}

        def resample_audio(audio: np.ndarray[Any, Any], sample_rate: int) -> np.ndarray[Any, Any]:
            """
            Resamples the audio to the desired sample rate.
            """
            if sample_rate not in resamplers:
                resamplers[sample_rate] = torchaudio.transforms.Resample(sample_rate, self.sample_rate).to(dtype=torch.float32)
            resampler = resamplers[sample_rate]
            return resampler(torch.tensor(audio).to(dtype=torch.float32)).numpy().astype(np.float32) # type: ignore[no-any-return]

        def flush_buffer() -> None:
            """
            Flushes the buffer to disk.
            """
            nonlocal buffer
            this_file_num = len(data_files)
            chunk_output_path = os.path.join(output_dir, f"{this_file_num:0{num_file_digits}d}.npy")
            np.save(chunk_output_path, buffer) # type: ignore[arg-type]
            data_files.append(chunk_output_path)
            buffer = None

        def process_batch() -> None:
            """
            Processes the batch.
            """
            nonlocal buffer, num_batches
            # Get embeddings from audio
            batch_embeddings = self.speech_embeddings(
                [a for (a, s) in batch],
                spectrogram_batch_size=self.embedding_batch_size,
                embedding_batch_size=self.embedding_batch_size,
                remove_nan=False
            ) # n, 16, 96
            assert isinstance(batch_embeddings, np.ndarray) # type guard

            # Get other embeddings
            batch_labeled_embeddings = self.label_embeddings(
                embeddings=batch_embeddings,
                batch=batch,
            )

            # Remove NaN's
            batch_labeled_embeddings = batch_labeled_embeddings[~np.isnan(batch_labeled_embeddings).any(axis=(1, 2))]

            if len(batch_labeled_embeddings) != len(batch):
                removed_samples = len(batch) - len(batch_labeled_embeddings)
                logger.warning(f"Removed {removed_samples} samples due to NaN values in embeddings.")

            # Append to buffer
            if buffer is None:
                buffer = batch_labeled_embeddings
            else:
                buffer = np.concatenate([buffer, batch_labeled_embeddings])

            # Clear the batch
            batch.clear()
            num_batches += 1
            progress.update(1)
            # Flush the buffer if needed
            if buffer is not None and buffer.shape[0] >= samples_per_file:
                flush_buffer()
            gc.collect()

        for sample in dataset:
            # Get the audio array and sample rate
            audio = sample.pop(self.audio_key)
            if self.audio_sample_rate_key is not None:
                try:
                    sample_rate = audio[self.audio_sample_rate_key]
                except KeyError:
                    try:
                        sample_rate = sample[self.audio_sample_rate_key]
                    except KeyError:
                        sample_rate = None
            else:
                sample_rate = None

            if self.audio_array_key is not None:
                audio = audio[self.audio_array_key]

            # Re-sample if needed
            if sample_rate is not None and sample_rate != self.sample_rate:
                audio = resample_audio(audio, sample_rate)

            # Cast if needed
            audio = audio.astype(np.float32)

            # Iterate through the audio and create batches
            for i in range(0, len(audio), self.samples_per_batch): # Default is two seconds (31.440 samples)
                batch_audio = audio[i:i+self.samples_per_batch]
                if batch_audio.shape[0] < self.samples_per_batch: # Right pad with zeros if needed
                    batch_audio = np.concatenate([
                        batch_audio,
                        np.zeros(self.samples_per_batch - batch_audio.shape[0])
                    ]).astype(np.float32)

                batch.append((batch_audio, sample))
                # Process the batch if we have enough samples
                if len(batch) >= self.process_batch_size:
                    process_batch()

                # Check if we have reached the maximum number of batches
                if num_batches >= max_batches:
                    break
            # Check if we have reached the maximum number of batches
            if num_batches >= max_batches:
                break

        # Check if we have any remaining samples
        if len(batch) > 0 and num_batches < max_batches:
            process_batch()

        # Flush the remaining buffer
        if buffer is not None:
            flush_buffer()

class PrecalculatedLabeledTrainingDatasetGenerator(PrecalculatedTrainingDatasetGenerator):
    """
    A generator for creating labeled training datasets.
    This should be used to take existing in-the-wild data and create packed datasets of features and labels.

    The labels are the tokens of the text in the data, encoded with a BERT tokenizer.
    """
    def __init__(
        self,
        dataset_path: str,
        config_name: Optional[str]=None,
        split: str="train",
        audio_key: str="audio",
        audio_array_key: Optional[str]="array",
        audio_sample_rate_key: Optional[str]="sampling_rate",
        transcript_key: str="transcript",
        device_id: Optional[int]=None,
        sample_rate: int=16000,
        seconds_per_batch: float=1.44,
        process_batch_size: int=128,
        embedding_batch_size: int=32,
        tokenizer_max_length: int=96,
    ) -> None:
        """
        See the above class for more information.
        :param transcript_key: The key for the transcript.
        :param tokenizer_max_length: The maximum length of the tokenizer.
        """
        super().__init__(
            dataset_path=dataset_path,
            config_name=config_name,
            split=split,
            audio_key=audio_key,
            audio_array_key=audio_array_key,
            audio_sample_rate_key=audio_sample_rate_key,
            device_id=device_id,
            sample_rate=sample_rate,
            seconds_per_batch=seconds_per_batch,
            process_batch_size=process_batch_size,
        )
        self.transcript_key = transcript_key
        self.tokenizer_max_length = tokenizer_max_length

    @property
    def tokenizer(self) -> BERTTokenizer:
        """
        Returns the tokenizer.
        """
        if not hasattr(self, "_tokenizer"):
            from heybuddy.tokens import BERTTokenizer
            self._tokenizer = BERTTokenizer(length=self.tokenizer_max_length)
        return self._tokenizer

    def tokenize(self, text: str) -> np.ndarray[Any, Any]:
        """
        Tokenizes the text.
        """
        if getattr(self, "_last_text", None) == text:
            return self._last_tokens
        tokens = self.tokenizer(text).numpy()
        self._last_text = text
        self._last_tokens = tokens
        return tokens

    def label_embeddings(
        self,
        embeddings: np.ndarray[Any, Any],
        batch: List[Tuple[np.ndarray[Any, Any], Dict[str, Any]]],
    ) -> np.ndarray[Any, Any]:
        """
        Adds any additional embeddings to the batch.
        """
        tokens: Dict[str, np.ndarray[Any, Any]] = {}
        batch_tokens: List[np.ndarray[Any, Any]] = []
        for i, (audio, sample) in enumerate(batch):
            # Get the transcript
            transcript = sample[self.transcript_key]
            # Tokenize the transcript if not already done
            if transcript in tokens:
                tokenized = tokens[transcript]
            else:
                tokenized = self.tokenize(transcript)[np.newaxis, ...]
                tokens[transcript] = tokenized

            # Append the tokens to the batch
            batch_tokens.append(tokenized)

        # Concatenate
        return np.stack([ # type: ignore[no-any-return]
            np.concatenate([e, t], axis=0)
            for e, t in zip(embeddings, batch_tokens)
        ]).astype(np.float32)

class PrecalculatedDatasetIterator:
    """
    An extensible dataset generator for precalculated features.
    """
    def __init__(
        self,
        name: str,
        directory: str=LOCAL_DIR,
        exclude_phrase: Optional[str]=None,
        ordered: bool=False,
        labeled: bool=False,
        use_mem_map: bool=True,
        shuffle: bool=True,
        data: Optional[np.ndarray[Any, Any]]=None,
    ) -> None:
        """
        :param name: The name of the precalculated features. Must be unique.
        """
        self.lock = Lock()
        self.directory = directory
        self.name = name
        self.exclude_phrase = exclude_phrase
        self.index = 0
        self.total_taken = 0
        self.ordered = ordered
        self.labeled = labeled
        self.use_mem_map = use_mem_map
        if data is not None:
            self._precalculated = data
        if not os.path.exists(self.precalculated_path):
            raise FileNotFoundError(f"Could not find precalculated features at {self.precalculated_path}.")
        if shuffle and not ordered:
            self.shuffle() # Shuffle the indexes

    @property
    def exclude_text(self) -> str:
        """
        Gets the exclude phrase, with special tokens replaced with spaces,
        and superfluous whitespace removed.
        """
        if not hasattr(self, "_exclude_text"):
            if self.exclude_phrase is None:
                self._exclude_text = ""
            else:
                self._exclude_text = re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9]", " ", self.exclude_phrase.replace("'", ""))).strip()
        return self._exclude_text

    @property
    def tokenizer(self) -> BERTTokenizer:
        """
        Returns the tokenizer.
        """
        if not hasattr(self, "_tokenizer"):
            from heybuddy.tokens import BERTTokenizer
            self._tokenizer = BERTTokenizer()
        return self._tokenizer

    @property
    def exclude_tokens(self) -> set[int]:
        """
        Returns the tokens to exclude.
        """
        if not hasattr(self, "_exclude_tokens"):
            if self.exclude_phrase is None:
                self._exclude_tokens = set(np.array([]).flatten())
            else:
                self._exclude_tokens = set(self.tokenizer(self.exclude_text).numpy().flatten())
        return self._exclude_tokens

    @property
    def precalculated_path(self) -> str:
        """
        Returns the path to the precalculated data.
        """
        if not hasattr(self, "_precalculated_path"):
            self._precalculated_path = os.path.join(
                self.directory,
                f"{self.name}.npy",
            )
        return self._precalculated_path

    @property
    def precalculated(self) -> np.ndarray[Any, Any]:
        """
        Returns the precalculated features.
        """
        if not hasattr(self, "_precalculated"):
            load_kwargs: Dict[str, str] = {}
            if self.use_mem_map:
                load_kwargs["mmap_mode"] = "r"
            self._precalculated = np.load(
                self.precalculated_path,
                **load_kwargs, # type: ignore[arg-type]
            )
        return self._precalculated

    @property
    def indexes(self) -> np.ndarray[Any, Any]:
        """
        Returns the indexes of the precalculated features.
        """
        if not hasattr(self, "_indexes"):
            self._indexes = np.arange(len(self.precalculated))
        return self._indexes

    @classmethod
    def from_array(
        cls,
        array: np.ndarray[Any, Any],
        name: str,
        directory: str=LOCAL_DIR,
        ordered: bool=False,
        keep_in_memory: bool=False,
    ) -> PrecalculatedDatasetIterator:
        """
        Saves the precalculated features to disk.
        """
        precalculated_path = os.path.join(
            directory,
            f"{name}.npy",
        )
        np.save(precalculated_path, array)
        return PrecalculatedDatasetIterator(
            name,
            data=array if keep_in_memory else None,
            ordered=ordered,
        )

    def shuffle(self) -> Self:
        """
        Shuffles the precalculated features.
        """
        if not self.ordered:
            np.random.shuffle(self.indexes)
        return self

    def take(self, n: int) -> np.ndarray[Any, Any]:
        """
        Takes the first `n` samples.
        """
        # Dims can change depending on what the task is
        # For simple voice embeddings, the shape is [n, 16, 96]
        # For labeled voice embeddings, the shape is [n, 17, 96], with the last dimension being the tokens
        batch = self.precalculated[self.indexes[self.index:self.index+n]]
        if batch.shape[0] < n:
            # If we run out of samples, reset the index and shuffle.
            self.index = n - batch.shape[0]
            self.shuffle()
            batch = np.concatenate([
                batch,
                self.precalculated[self.indexes[:self.index]]
            ])
        else:
            self.index += n

        if self.labeled:
            if self.exclude_phrase is not None:
                # Filter
                mask = np.ones(n, dtype=bool)
                for i in range(n):
                    if not set(batch[i, -1, :].flatten()).isdisjoint(self.exclude_tokens):
                        mask[i] = False
                batch = batch[mask]
            batch = batch[:, :-1]
            if batch.shape[0] < n:
                batch = np.concatenate([
                    batch,
                    self.take(n - batch.shape[0])
                ])

        self.total_taken += n
        return batch # type: ignore[no-any-return]

    def iterate(self) -> Iterator[np.ndarray[Any, Any]]:
        """
        Iterates over the precalculated features.
        """
        while True:
            yield self.take(1)

    def metadata(self) -> Dict[str, Any]:
        """
        Gets the metadata for the precalculated features.
        """
        return {
            "name": self.name,
            "path": self.precalculated_path,
            "shape": self.precalculated.shape,
            "ordered": self.ordered,
            "labeled": self.labeled,
            "use_mem_map": self.use_mem_map,
        }

    def __len__(self) -> int:
        """
        Returns the number of samples in the precalculated features.
        """
        return self.precalculated.shape[0]

    def __iter__(self) -> Iterator[np.ndarray[Any, Any]]:
        """
        Iterates over the precalculated features.
        """
        return self.iterate()

    def __repr__(self) -> str:
        """
        Returns a string representation of the generator.
        """
        return f"{type(self).__name__}(num_samples={len(self)})"

class HostedPrecalculatedDatasetIterator(PrecalculatedDatasetIterator):
    """
    An extensible dataset generator for precalculated features.

    This class is for datasets that are hosted on the web.
    """
    precalculated_url: str
    precalculated_sha256_sum: Optional[str]=None
    precalculated_authorization: Optional[str]=None
    precalculated_ordered: bool=False
    precalculated_labeled: bool=True
    precalculated_use_mem_map: bool=True

    def __init__(
        self,
        directory: str=LOCAL_DIR,
        exclude_phrase: Optional[str]=None,
    ) -> None:
        """
        Infers the name of the precalculated features from the URL.
        """
        super().__init__(
            name=os.path.splitext(get_file_name_from_url(self.precalculated_url))[0],
            directory=directory,
            ordered=self.precalculated_ordered,
            labeled=self.precalculated_labeled,
            exclude_phrase=exclude_phrase,
            use_mem_map=self.precalculated_use_mem_map,
        )

    @property
    def precalculated_path(self) -> str:
        """
        Returns the path to the precalculated data.
        """
        if not hasattr(self, "_precalculated_path"):
            self._precalculated_path = check_download_file_to_dir(
                self.precalculated_url,
                LOCAL_DIR,
                use_tqdm=True,
                sha256_sum=self.precalculated_sha256_sum,
                authorization=self.precalculated_authorization,
            )
        return self._precalculated_path

class PrecalculatedTrainingDatasetLarge(HostedPrecalculatedDatasetIterator):
    """
    A set of precalculated features for training.

    This is in-the-wild audio data, so it should be interpreted
    as general negative examples. You will still need to supplement
    this set with adversarial examples to train a robust model.
    """
    precalculated_url: str = "https://huggingface.co/datasets/benjamin-paine/hey-buddy/resolve/main/precalculated/common/training-large.npy"

class PrecalculatedTrainingDatasetMedium(HostedPrecalculatedDatasetIterator):
    """
    A set of precalculated features for training.

    This is in-the-wild audio data, so it should be interpreted
    as general negative examples. You will still need to supplement
    this set with adversarial examples to train a robust model.
    """
    precalculated_url: str = "https://huggingface.co/datasets/benjamin-paine/hey-buddy/resolve/main/precalculated/common/training-medium.npy"

class PrecalculatedValidationDataset(HostedPrecalculatedDatasetIterator):
    """
    A set of precalculated features for validation (i.e., false-positive-rate estimation).

    This is in-the-wild speech data, so it should be interpreted
    as general negative examples. You will still need to supplement
    this set with adversarial examples to train a robust model.
    """
    precalculated_url: str = "https://huggingface.co/datasets/benjamin-paine/hey-buddy/resolve/main/precalculated/common/validation.npy"
