from __future__ import annotations

from typing import Any, Iterator, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from datasets import Features, Value, Dataset # type: ignore[import-untyped]

__all__ = [
    "DatasetGenerator",
    "AudioDatasetGenerator",
]

class DatasetGenerator:
    """
    A parent class for dataset generators.
    """
    def __init__(
        self,
        device_id: Optional[int]=None,
    ) -> None:
        """
        :param device_id: The device ID to use for generating data.
        """
        self.device_id = device_id

    @property
    def device(self) -> torch.device:
        """
        Returns the device to use for generating data.
        """
        import torch
        if self.device_id is not None:
            return torch.device(f"cuda:{self.device_id}")
        return torch.device("cpu")

    def get_features(self) -> Features:
        """
        Returns the features of the generated samples.
        """
        raise NotImplementedError

    def __call__(
        self,
        num_samples: int,
        **kwargs: Any
    ) -> Iterator[Dict[str, Any]]:
        """
        Generates samples.
        """
        raise NotImplementedError

    @classmethod
    def dataset(
        cls,
        num_samples: int=1000,
        device_id: Optional[int]=None,
        **kwargs: Any
    ) -> Dataset:
        """
        Builds a dataset from the generator.

        This somewhat defeats the purpose of the generator, but is useful for testing and
        for sharing datasets with others or to huggingface.

        :param num_threads: The number of threads to use for generating data.
        :param num_samples: The number of samples to generate.
        """
        from datasets import Dataset
        generator = cls(device_id=device_id, **kwargs)
        return Dataset.from_generator(
            generator.__call__,
            gen_kwargs={"num_samples": num_samples},
            features=generator.get_features(),
        )

class AudioDatasetGenerator(DatasetGenerator):
    """
    A small extension of the DatasetGenerator that returns
    the feature set for audio datasets, as well as formatting
    results in the iterate function.
    """
    def get_sample_rate(self) -> int:
        """
        Returns the sample rate of the generated samples.
        """
        return 16000

    def get_metadata_features(self) -> Dict[str, Value]:
        """
        Returns the metadata features of the generated samples.
        """
        return {}

    def get_features(self) -> Features:
        """
        Returns the features of the generated samples.
        """
        from datasets import Features, Audio
        return Features({
            "audio": Audio(),
            **self.get_metadata_features(),
        })
