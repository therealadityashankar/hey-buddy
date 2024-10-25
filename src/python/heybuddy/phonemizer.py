from __future__ import annotations

from heybuddy.util import PretrainedTorchModel

from typing import Dict, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from phonemizer.backend import EspeakBackend # type: ignore[import-untyped,import-not-found,unused-ignore]

__all__ = [
    "PretrainedPhonemizer",
    "get_phonemizer",
]

class PretrainedPhonemizer(PretrainedTorchModel):
    """
    A wrapper around the pretrained phonemizer model.

    >>> phonemizer = PretrainedPhonemizer()
    >>> phonemizer("hello world")
    '[HH][AH][L][OW] [W][ER][L][D]'
    """
    pretrained_model_url = "https://huggingface.co/benjamin-paine/hey-buddy/resolve/main/pretrained/deep-phonemizer.pt"
    model_path = "predictor" # The phonemizer class is itself a wrapper, so we point to the NN module inside it

    def load(self) -> None:
        """
        Load the pretrained model from the given path.

        Overrides the parent method to instead use DP's `from_checkpoint` method.
        """
        try:
            from dp.phonemizer import Phonemizer # type: ignore[import-untyped]
        except ImportError:
            raise ImportError("DeepPhonemizer package is required to use this model. Install it via `pip install deep-phonemizer`.")

        self.module = Phonemizer.from_checkpoint(
            self.pretrained_model_path,
            device=self.device,
        )
        self.loaded = True

    def __call__(self, text: str, lang: str="en_us") -> str:
        """
        Predict the phonemes for the given text.
        """
        if not self.loaded:
            self.load()
        assert self.module is not None
        return str(self.module(text, lang=lang))

class SimplePhonemizer:
    """
    A wrapper around the phonemizer package.
    """
    backends: Dict[str, EspeakBackend]
    phone_map: Dict[str, str] = {
        "ɑ": "AA",
        "æ": "AE",
        "ʌ": "AH",
        "ɔ": "AO",
        "aʊ": "AW",
        "ə": "AH", # schwa, mapped to AH in CMU but not arpabet
        "ɚ": "AXR",
        "aɪ": "AY",
        "ɛ": "EH",
        "ɝ": "ER",
        "ɜː": "ER",
        "eɪ": "EY",
        "ɪ": "IH",
        "ɨ": "IX",
        "iː": "IY",
        "i": "IY",
        "oʊ": "OW",
        "ɔɪ": "OY",
        "ʊ": "UH",
        "u": "UW",
        "ʉ": "UX",
        "b": "B",
        "tʃ": "CH",
        "d": "D",
        "ð": "DH",
        "ɾ": "DX",
        "l̩": "EL",
        "m̩": "EM",
        "n̩": "EN",
        "f": "F",
        "ɡ": "G",
        "h": "HH",
        "dʒ": "JH",
        "k": "K",
        "l": "L",
        "m": "M",
        "n": "N",
        "ŋ": "NG",
        "ɾ̃": "NX",
        "p": "P",
        "ʔ": "Q",
        "ɹ": "R",
        "s": "S",
        "ʃ": "SH",
        "t": "T",
        "θ": "TH",
        "v": "V",
        "w": "W",
        "ʍ": "WH",
        "j": "Y",
        "z": "Z",
        "ʒ": "ZH",
        ":": ":"
    }
    def __init__(self) -> None:
        self.backends = {}

    def get_backend(self, lang: str) -> EspeakBackend:
        """
        Get the backend for the given language.

        :param lang: The language code.
        :return: The backend for the given language.
        """
        if lang not in self.backends:
            try:
                from phonemizer.backend import EspeakBackend # type: ignore[import-untyped,import-not-found,unused-ignore]
            except ImportError:
                raise ImportError("Phonemizer package is required to use this model. Install it via `pip install phonemizer`.")
            self.backends[lang] = EspeakBackend(lang)
        return self.backends[lang]

    def ipa_to_arpabet(self, ipa: str) -> str:
        """
        Convert the given IPA to ARPAbet.

        :param ipa: The IPA string.
        :return: The ARPAbet string.
        """
        arpabet = ""
        ipa_len = len(ipa)
        i = 0
        while i < ipa_len:
            if i < ipa_len - 1 and ipa[i:i+2] in self.phone_map:
                this_phone = self.phone_map[ipa[i:i+2]]
                i += 1
            elif ipa[i] in self.phone_map:
                this_phone = self.phone_map[ipa[i]]
            else:
                this_phone = ipa[i]
            arpabet += f"[{this_phone}]"
            i += 1
        return arpabet

    def __call__(self, text: str, lang: str="en-us") -> str:
        """
        Predict the phonemes for the given text.
        """
        backend = self.get_backend(lang)
        return " ".join([
            self.ipa_to_arpabet(phone)
            for phone in backend.phonemize(text.strip().split(), strip=True)
        ])

GLOBAL_PHONEMIZERS: Dict[Optional[int], PretrainedPhonemizer] = {}
GLOBAL_SIMPLE_PHONEMIZER = SimplePhonemizer()
def get_phonemizer(
    device_id: Optional[int] = None,
    use_deep_phonemizer: bool = False,
) -> Union[PretrainedPhonemizer, SimplePhonemizer]:
    """
    Get the phonemizer model for the given device id.
    """
    if not use_deep_phonemizer:
        return GLOBAL_SIMPLE_PHONEMIZER
    if device_id not in GLOBAL_PHONEMIZERS:
        GLOBAL_PHONEMIZERS[device_id] = PretrainedPhonemizer(device_id=device_id, load=True)
    return GLOBAL_PHONEMIZERS[device_id]
