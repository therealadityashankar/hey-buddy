from __future__ import annotations

from typing import List, Iterator, Dict, Tuple, Optional, Any, TYPE_CHECKING

import numpy as np

from heybuddy.constants import *
from heybuddy.util import get_adversarial_text_generator
from heybuddy.piper.pretrained import get_piper_tts_model
from heybuddy.dataset.generator import AudioDatasetGenerator

if TYPE_CHECKING:
    from heybuddy.util import AdversarialTextGenerator
    from datasets import Value # type: ignore[import-untyped]

class PiperSpeechGenerator(AudioDatasetGenerator):
    """
    Generates speech samples using tibritts.
    """
    def __init__(
        self,
        device_id: Optional[int]=None,
        phrase: str="Hello, world!",
        phrase_augment_prob: float=DEFAULT_AUGMENT_PHRASE_PROB,
        phrase_augment_words: List[str]=DEFAULT_AUGMENT_PHRASE_WORDS,
        additional_phrases: List[str]=[],
        adversarial: bool=False,
        num_adversarial_texts: int=10,
        custom_adversarial_texts: Optional[List[str]]=None,
        target_sample_rate: int=16000,
        resample_lowpass_filter_width: int=64,
        resample_rolloff: float=0.9475937167399596,
        resample_method: str="sinc_interp_kaiser",
        resample_beta: float=14.769656459379492,
        batch_size: int=1,
        slerp_weights: Tuple[float, ...]=DEFAULT_TTS_SLERP_WEIGHTS,
        length_scales: Tuple[float, ...]=DEFAULT_TTS_LENGTH_SCALES,
        noise_scales: Tuple[float, ...]=DEFAULT_TTS_NOISE_SCALES,
        noise_scale_ws: Tuple[float, ...]=DEFAULT_TTS_NOISE_SCALE_WEIGHTS,
        max_speakers: Optional[int]=None,
        min_phoneme_count: Optional[int]=None,
    ) -> None:
        """
        Initializes the speech generator.

        :param device_id: The device ID to use.
        :param phrase: The phrase to generate speech for.
        :param phrase_augment_prob: The probability of augmenting the phrase.
        :param phrase_augment_words: The words to augment the phrase with.
        :param additional_phrases: Additional phrases that are in this dataset; only used for ensuring we don't
                                   accidentally create adversarial samples for these phrases.
        :param adversarial: Whether to generate adversarial samples.
        :param num_adversarial_texts: The number of adversarial texts to generate.
        :param target_sample_rate: The target sample rate.
        :param resample_lowpass_filter_width: The lowpass filter width for resampling.
        :param resample_rolloff: The rolloff for resampling.
        :param resample_method: The resampling method.
        :param resample_beta: The beta for resampling.
        :param batch_size: The batch size.
        :param slerp_weights: The slerp weights.
        :param length_scales: The length scales.
        :param noise_scales: The noise scales.
        :param noise_scale_ws: The noise scale weights.
        :param max_speakers: The maximum number of speakers.
        :param min_phoneme_count: The minimum number of phonemes.
        """
        super().__init__(device_id=device_id)
        self.phrase = phrase
        self.phrase_augment_prob = phrase_augment_prob
        self.phrase_augment_words = phrase_augment_words
        self.additional_phrases = additional_phrases
        self.adversarial = adversarial
        self.slerp_weights = slerp_weights
        self.length_scales = length_scales
        self.noise_scales = noise_scales
        self.noise_scale_ws = noise_scale_ws
        self.max_speakers = max_speakers
        self.min_phoneme_count = min_phoneme_count
        self.num_adversarial_texts = num_adversarial_texts
        self.custom_adversarial_texts = custom_adversarial_texts
        self.batch_size = batch_size
        self.target_sample_rate = target_sample_rate
        self.model = get_piper_tts_model(device_id=device_id)

    def get_metadata_features(self) -> Dict[str, Value]:
        """
        Returns the metadata features.

        :return: The metadata features.
        """
        from datasets import Value
        return {
            "phrase": Value(dtype="string", id=None),
        }

    def get_adversarial_generator(self) -> AdversarialTextGenerator:
        """
        Returns the adversarial generator.

        :return: The adversarial generator.
        """
        if not hasattr(self, "_adversarial_generator"):
            from heybuddy.util import get_adversarial_text_generator
            generator = get_adversarial_text_generator(
                device_id=(self.device.index or 0) if self.device.type != "cpu" else None
            )
            self._adversarial_generator = generator
        return self._adversarial_generator
  
    def get_adversarial_texts(self) -> List[str]:
        """
        Returns a batch of adversarial texts.
        """
        if not hasattr(self, "_adversarial_texts"):
            if self.custom_adversarial_texts is not None:
                custom_texts = list(self.custom_adversarial_texts)
            else:
                custom_texts = []
            to_generate = self.num_adversarial_texts - len(custom_texts)
            self._adversarial_texts = custom_texts + list(
                self.get_adversarial_generator()(
                    self.phrase,
                    num_samples=to_generate,
                )
            )
            # remove any additional phrases
            self._adversarial_texts = [
                text for text in self._adversarial_texts
                if text not in self.additional_phrases
            ]
            assert len(self._adversarial_texts) > 0, "No adversarial texts generated"
        return self._adversarial_texts

    def get_texts(self, retry: bool=True) -> List[Tuple[str, float]]:
        """
        Returns the phrases to generate speech for.

        :return: The phrases to generate speech for.
        """
        if self.adversarial:
            try:
                unaugmented = self.get_adversarial_texts()
            except AttributeError:
                if retry:
                    if hasattr(self, "_adversarial_generator"):
                        del self._adversarial_generator
                    return self.get_texts(retry=False)
                raise
        else:
            unaugmented = [self.phrase]

        augmented = []
        if self.phrase_augment_prob > 0.0:
            augment_weight = self.phrase_augment_prob / (len(unaugmented) * len(self.phrase_augment_words))
            for phrase in unaugmented:
                for augmentation in self.phrase_augment_words:
                    augmented.append((f"{phrase}. {augmentation}", augment_weight))

        all_texts = [(u, 1.0) for u in unaugmented] + augmented
        return all_texts

    def __call__(
        self,
        num_samples: int,
        **kwargs: Any
    ) -> Iterator[Dict[str, Any]]:
        """
        Generates speech samples.
        """
        total_batches = int(np.ceil(num_samples / self.batch_size))
        for i in range(total_batches):
            batch_samples = min(num_samples - i * self.batch_size, self.batch_size)
            for text, audio in self.model(
                texts=self.get_texts(),
                num_samples=batch_samples,
                batch_size=self.batch_size,
                slerp_weights=self.slerp_weights,
                length_scales=self.length_scales,
                noise_scales=self.noise_scales,
                noise_scale_ws=self.noise_scale_ws,
                max_speakers=self.max_speakers,
                min_phoneme_count=self.min_phoneme_count,
                target_sample_rate=self.target_sample_rate,
            ):
                yield {
                    "audio": {
                        "array": audio,
                        "sampling_rate": self.target_sample_rate,
                    },
                    "phrase": text,
                }
