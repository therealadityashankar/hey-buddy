from __future__ import annotations

import re
import itertools
import numpy as np

from typing import List, Iterator, Optional, Dict, Optional, Union, TYPE_CHECKING
from heybuddy.util.log_util import logger

if TYPE_CHECKING: # This must be lazily imported during checking or runtime, otherwise it will cause a circular import
    from heybuddy.phonemizer import PretrainedPhonemizer, SimplePhonemizer

__all__ = [
    "AdversarialTextGenerator",
    "get_adversarial_text_generator",
]

def replace_phonemes(
    input_chars: List[str],
    max_replace: int,
    replace_char: str='"(.){1,3}"',
) -> List[str]:
    """
    Replace phonemes in a list of characters.
    """
    results = []
    num_chars = len(input_chars)

    # iterate over the number of characters to replace (1 to max_replace)
    for r in range(1, max_replace + 1):
        # get all combinations for a fixed r
        combinations = itertools.combinations(range(num_chars), r)
        for combination in combinations:
            chars = input_chars.copy()
            for index in combination:
                chars[index] = replace_char
            results.append(" ".join(chars))

    return results

class AdversarialTextGenerator:
    """
    Generate adversarial words and phrases based on phoneme overlap.

    >>> generator = AdversarialTextGenerator()("hello world", seed=42)
    >>> next(generator)
    'melone telecommuter'
    >>> next(generator)
    'coelho wilhelma'
    >>> next(generator)
    "strangled d'angelo's"
    """
    VOWEL_PHONEMES = ["AA", "AE", "AH", "AO", "AW", "AX", "AXR", "AY", "EH", "ER", "EY", "IH", "IX", "IY", "OW", "OY", "UH", "UW", "UX"]

    def __init__(
        self,
        device_id: Optional[int]=None,
        partial_phrase_ratio: float=0.10,
        input_words_ratio: float=0.33,
    ) -> None:
        """
        :param partial_phrase_ratio: The probability of returning a number of words less than the input
                                     text (but always between 1 and the number of input words)
        :param input_words_ratio: The probability of including individual input words in the adversarial
                                  texts when the input text consists of multiple words. For example,
                                  if the `input_text` was "ok google", then setting this value > 0.0
                                  will allow for adversarial texts like "ok noodle", versus the word "ok"
                                  never being present in the adversarial texts.
        """
        self.device_id = device_id
        self.partial_phrase_ratio = partial_phrase_ratio
        self.input_words_ratio = input_words_ratio

    @property
    def phonemizer(self) -> Union[PretrainedPhonemizer, SimplePhonemizer]:
        """
        :return: The phonemizer object used to predict phonemes for words.
        """
        if not hasattr(self, "_phonemizer"):
            from heybuddy.phonemizer import get_phonemizer
            self._phonemizer = get_phonemizer(device_id=self.device_id)
        return self._phonemizer

    def __call__(
        self,
        input_text: str,
        num_samples: Optional[int]=None,
        seed: Optional[int]=None,
    ) -> Iterator[str]:
        """
        Generate adversarial words and phrases based on phoneme overlap.

        :param input_text: The target text for adversarial phrases
        :param num_samples: The total number of adversarial texts to return. Uses sampling,
                            so not all possible combinations will be included and some duplicates
                            may be present. When absent, generates forever.
        :return: An iterable of strings corresponding to words and phrases that are phonetically
                 similar (but not identical) to the input text.
        """
        try:
            import pronouncing # type: ignore[import-untyped]
        except ImportError:
            raise ImportError("The 'pronouncing' library is required to generate adversarial texts! Run 'pip install pronouncing' to install.")

        if seed:
            np.random.seed(seed)

        word_phones = []
        input_text_phones = [pronouncing.phones_for_word(i) for i in input_text.split()]

        for phones, word in zip(input_text_phones, input_text.split()):
            if phones != []:
                word_phones.extend(phones)
            elif phones == []:
                logger.info(f"The word '{word}' was not found in the pronunciation dictionary, using Phonemizer.")
                phones = self.phonemizer(word)
                logger.info(f"Phones for '{word}': {phones}")
                word_phones.append(re.sub(r"[\]|\[]", "", re.sub(r"\]\[", " ", phones)))
            elif isinstance(phones[0], list):
                logger.info(f"There are multiple pronunciations for the word '{word}', using the first one.")
                word_phones.append(phones[0])

        # add all possible lexical stresses to vowels
        word_phones = [re.sub('|'.join(self.VOWEL_PHONEMES), lambda x: str(x.group(0)) + '[0|1|2]', re.sub(r'\d+', '', i)) for i in word_phones]

        adversarial_phrases = []
        for phones, word in zip(word_phones, input_text.split()):
            query_exps = []
            phones = phones.split()
            adversarial_words = []
            if len(phones) <= 2:
                query_exps.append(" ".join(phones))
            else:
                query_exps.extend(replace_phonemes(phones, max_replace=max(0, len(phones)-2), replace_char="(.){1,3}"))

            for query in query_exps:
                matches = pronouncing.search(query)
                matches_phones = [pronouncing.phones_for_word(i)[0] for i in matches]
                allowed_matches = [i for i, j in zip(matches, matches_phones) if j != phones]
                adversarial_words.extend([i for i in allowed_matches if word.lower() != i])

            if adversarial_words != []:
                adversarial_phrases.append(adversarial_words)

        # Build combinations for final output
        yielded_samples = 0
        while True:
            if num_samples and yielded_samples >= num_samples:
                break

            txts = []
            for j, k in zip(adversarial_phrases, input_text.split()):
                if np.random.random() > (1 - self.input_words_ratio):
                    txts.append(k)
                else:
                    txts.append(np.random.choice(j))

            if len(input_text.split()) > 1 and np.random.random() <= self.partial_phrase_ratio:
                n_words = np.random.randint(1, len(input_text.split())+1)
                adversarial_text = " ".join(np.random.choice(txts, size=n_words, replace=False))
            else:
                adversarial_text = " ".join(txts)

            if adversarial_text != input_text:
                yield adversarial_text
                yielded_samples += 1

ADVERSARIAL_TEXT_GENERATORS: Dict[Optional[int], AdversarialTextGenerator] = {}
def get_adversarial_text_generator(device_id: Optional[int]=None) -> AdversarialTextGenerator:
    """
    Get a cached adversarial text generator based on the device ID.

    :param device_id: The device ID to use for the adversarial text generator.
    :return: An adversarial text generator object.
    """
    if device_id not in ADVERSARIAL_TEXT_GENERATORS:
        ADVERSARIAL_TEXT_GENERATORS[device_id] = AdversarialTextGenerator(device_id=device_id)
    return ADVERSARIAL_TEXT_GENERATORS[device_id]
