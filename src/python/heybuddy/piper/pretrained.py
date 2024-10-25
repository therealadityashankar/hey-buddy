from __future__ import annotations

import gc
import sys
import random
import numpy as np
import itertools

from typing import List, Dict, Optional, Dict, Any, Tuple, Union, Type, Callable, TYPE_CHECKING

from heybuddy.constants import *
from heybuddy.util import PretrainedTorchModel
from heybuddy.vad import get_vad_model

try:
    from piper_phonemize import phonemize_espeak # type: ignore[import-untyped]
except ImportError:
    sys.stderr.write("Could not import piper_phonemize. Please install it with `pip install piper-phonemize`.\n")
    sys.stderr.flush()
    raise

if TYPE_CHECKING:
    import torch
    from heybuddy.piper.models import Synthesizer

__all__ = [
    "PiperTTSModel",
    "get_piper_tts_model",
    "unload_piper_tts_model",
]

class PiperTTSModel(PretrainedTorchModel):
    """
    A wrapper around the pretrained PiperTTS model.
    """
    pretrained_model_url = "https://huggingface.co/benjamin-paine/hey-buddy/resolve/main/pretrained/piper-libritts-en-r-medium.safetensors"
    voice: str = "en-us"
    model_sample_rate: int = 22050
    resample_lowpass_filter_width: int = 64
    resample_rolloff: float = 0.9475937167399596
    resample_method: str = "sinc_interp_kaiser"
    resample_beta: float = 14.769656459379492
    num_speakers: int = 904
    model_kwargs = {
        "n_vocab": 256,
        "spec_channels": 513,
        "segment_size": 32,
        "inter_channels": 192,
        "hidden_channels": 192,
        "filter_channels": 768,
        "n_heads": 2,
        "n_layers": 6,
        "kernel_size": 3,
        "p_dropout": 0.1,
        "resblock": "2",
        "resblock_kernel_sizes": (3,5,7),
        "resblock_dilation_sizes": ((1,2),(2,6),(3,12)),
        "upsample_rates": (8,8,4),
        "upsample_initial_channel": 256,
        "upsample_kernel_sizes": (16,16,8),
        "n_speakers": 904,
        "gin_channels": 512,
        "use_sdp": True,
        "use_weight_norm": False,
        "encoder_use_weight_norm": True,
    }
    resample_kernels: Dict[int, torch.nn.Module] = {}
    text_phoneme_ids: Dict[str, Tuple[List[int], Optional[int]]] = {}

    @classmethod
    def get_model_class(cls) -> Type[Synthesizer]:
        """
        Gets the model class.
        """
        from heybuddy.piper.models import Synthesizer
        return Synthesizer

    def resample(
        self,
        audio: torch.Tensor,
        target_sample_rate: Optional[int]=None
    ) -> torch.Tensor:
        """
        Resamples the audio to the target sample rate.

        :param audio: The audio tensor.
        :param target_sample_rate: The target sample rate.
        :return: The resampled audio tensor.
        """
        import torchaudio # type: ignore[import-untyped]
        if target_sample_rate is None or target_sample_rate == self.model_sample_rate:
            return audio
        if not hasattr(self, "resample_kernels"):
            self.resample_kernels = {}
        if target_sample_rate not in self.resample_kernels:
            self.resample_kernels[target_sample_rate] = torchaudio.transforms.Resample(
                orig_freq=self.model_sample_rate,
                new_freq=target_sample_rate,
                lowpass_filter_width=self.resample_lowpass_filter_width,
                rolloff=self.resample_rolloff,
                resampling_method=self.resample_method,
                beta=self.resample_beta,
            ).to(self.device)
        return self.resample_kernels[target_sample_rate].to(audio.device)(audio) # type: ignore[no-any-return]

    def right_pad(self, lists: List[List[Any]], value: Any=1) -> None:
        """
        Right pads a list of lists in place.

        :param lists: The list of lists to pad.
        :param value: The value to pad with.
        """
        max_len = max(len(l) for l in lists)
        for l in lists:
            l.extend([value] * (max_len - len(l)))

    def phonemize(
        self,
        text: str,
        min_phoneme_count: Optional[int]=None,
    ) -> Tuple[List[int], Optional[int]]:
        """
        Phonemizes the given text.

        :param text: The text to phonemize.
        :param min_phoneme_count: The minimum phoneme count.
        :return: The phonemes and the clip index. When the clip index is None, use the whole audio, otherwise use the clip index.
        """
        clip_index: Optional[int] = None
        if text in self.text_phoneme_ids:
            phoneme_ids, clip_index = self.text_phoneme_ids[text]
            return phoneme_ids, clip_index

        from heybuddy.piper.phoneme_ids import phoneme_id_map
        phones = [
            p for sentence_phones in phonemize_espeak(text, self.voice)
            for p in sentence_phones
        ]
        text_phoneme_ids = []
        for phone in phones:
            if phone in phoneme_id_map:
                text_phoneme_ids.extend(
                    phoneme_id_map[phone] +
                    phoneme_id_map["_"]
                )

        if not text_phoneme_ids:
            raise ValueError(f"Could not phonemize text: {text}")

        phoneme_ids = phoneme_id_map["^"] + phoneme_id_map["_"] + text_phoneme_ids

        if min_phoneme_count is not None:
            while (len(phoneme_ids) - 1) < min_phoneme_count:
                clip_index = len(phoneme_ids) - 1
                phoneme_ids.extend(text_phoneme_ids)

        phoneme_ids.extend(phoneme_id_map["$"])
        self.text_phoneme_ids[text] = (phoneme_ids, clip_index)
        return phoneme_ids, clip_index

    def slerp(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        t: float,
        dot_threshold: float=0.9995,
        z_dim: int=-1
    ) -> torch.Tensor:
        """
        Spherical linear interpolation.

        :param a: The first vector.
        :param b: The second vector.
        :param t: The interpolation factor.
        :param dot_threshold: The threshold for the dot product.
        :param z_dim: The dimension to sum over.
        :return: The interpolated vector.
        """
        import torch
        a_norm = a / torch.norm(a, dim=z_dim, keepdim=True)
        b_norm = b / torch.norm(b, dim=z_dim, keepdim=True)
        dot = (a_norm * b_norm).sum(z_dim)

        # If the vectors are close enough, just return the linear interpolation
        if (torch.abs(dot) > dot_threshold).any():
            return (1 - t) * a + t * b

        theta = torch.acos(dot)
        theta_t = theta * t
        sin_theta = torch.sin(theta)
        sin_theta_t = torch.sin(theta_t)

        # Compute the sine scaling terms
        s_1 = torch.sin(theta - theta_t) / sin_theta
        s_2 = sin_theta_t / sin_theta

        # Interpolate
        return (s_1.unsqueeze(z_dim) * a) + (s_2.unsqueeze(z_dim) * b)

    def generate_batch(
        self,
        speaker_1: torch.Tensor,
        speaker_2: torch.Tensor,
        phoneme_id_batches: List[List[int]],
        slerp_weight: float,
        length_scale: float,
        noise_scale: float,
        noise_scale_w: float,
        max_length: Optional[int]=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a batch of speech samples.

        :param speaker_1: The first speaker.
        :param speaker_2: The second speaker.
        :param phoneme_id_batches: The phoneme ID batches.
        :param slerp_weight: The slerp weight.
        :param length_scale: The length scale.
        :param noise_scale: The noise scale.
        :param noise_scale_w: The noise scale for the weight.
        :param max_length: The maximum length of the audio.
        :return: The audio and the hop length.
        """
        import torch
        from heybuddy.piper.common import sequence_mask, generate_path
        input_tensor = torch.tensor(phoneme_id_batches, device=self.device).to(torch.int64)
        input_lengths_tensor = torch.tensor([len(p) for p in phoneme_id_batches], device=self.device).to(torch.int64)

        # Encoder
        input_tensor, input_prob_orig, logs_prob_orig, input_mask = self.model.enc_p(input_tensor, input_lengths_tensor)

        # Speaker embedding
        speaker_embedding = self.slerp(
            self.model.emb_g(speaker_1),
            self.model.emb_g(speaker_2),
            slerp_weight
        ).unsqueeze(-1)

        if self.model.use_sdp:
            log_weight = self.model.dp(input_tensor, input_mask, g=speaker_embedding, reverse=True, noise_scale=noise_scale_w)
        else:
            log_weight = self.model.dp(input_tensor, input_mask, g=speaker_embedding)

        weight = torch.exp(log_weight) * input_mask * length_scale
        weight_ceil = torch.ceil(weight)
        flow_lengths = torch.clamp_min(torch.sum(weight_ceil, [1,2]), 1).to(torch.int64)
        flow_mask = torch.unsqueeze(sequence_mask(flow_lengths, int(flow_lengths.max())), 1).to(dtype=input_mask.dtype)
        attention_mask = torch.unsqueeze(input_mask, 2) * torch.unsqueeze(flow_mask, -1)
        attention = generate_path(weight_ceil, attention_mask).squeeze(1)

        model_prob = torch.matmul(attention, input_prob_orig.transpose(1, 2)).transpose(1, 2)
        logs_prob = torch.matmul(attention, logs_prob_orig.transpose(1, 2)).transpose(1, 2)

        flow_prob = model_prob + torch.randn_like(model_prob) * torch.exp(logs_prob) * noise_scale
        flow = self.model.flow(flow_prob, flow_mask, g=speaker_embedding, reverse=True)
        audio = self.model.dec((flow * flow_mask)[:, :, :max_length], g=speaker_embedding)
        hop_length = weight_ceil * 256

        return audio, hop_length

    def trim_silence(
        self,
        sample: np.ndarray[Any, Any],
        frame_duration: float=0.030,
        sample_rate: int=16000,
        min_start: int=2000,
        threshold: float=0.05,
    ) -> np.ndarray[Any, Any]:
        """
        Uses SileroVAD to trim silence from the given sample.
        """
        vad = get_vad_model(
            device_id=(self.device.index or 0) if self.device.type != "cpu" else None
        )
        return vad.trim(
            sample,
            frame_duration=frame_duration,
            sample_rate=sample_rate,
            min_start=min_start,
            threshold=threshold,
        )

    def __call__(
        self,
        texts: Union[str, List[str], List[Tuple[str, float]]],
        num_samples: Optional[int]=None,
        batch_size: int=1,
        slerp_weights: Tuple[float, ...]=DEFAULT_TTS_SLERP_WEIGHTS,
        length_scales: Tuple[float, ...]=DEFAULT_TTS_LENGTH_SCALES,
        noise_scales: Tuple[float, ...]=DEFAULT_TTS_NOISE_SCALES,
        noise_scale_ws: Tuple[float, ...]=DEFAULT_TTS_NOISE_SCALE_WEIGHTS,
        max_speakers: Optional[int]=None,
        min_phoneme_count: Optional[int]=None,
        target_sample_rate: Optional[int]=None,
        trim_silence: bool=False,
        on_progress: Optional[Callable[[int, int], None]]=None,
    ) -> List[Tuple[str, np.ndarray[Any, Any]]]:
        """
        Generates speech samples.
        """
        import torch
        from heybuddy.piper.phoneme_ids import phoneme_id_map
        with torch.no_grad():
            num_speakers = self.num_speakers
            if max_speakers is not None:
                num_speakers = min(num_speakers, max_speakers)
            if num_samples is None:
                num_samples = len(texts)

            batch_size = max(batch_size, 1)
            num_batches = (num_samples + batch_size - 1) // batch_size

            # Cycle through all combinations of settings
            settings_iterator = itertools.cycle(
                itertools.product(
                    slerp_weights,
                    length_scales,
                    noise_scales,
                    noise_scale_ws,
                )
            )
            # Cycle through all combinations of speakers
            speakers_iterator = itertools.cycle(
                itertools.product(
                    range(num_speakers),
                    range(num_speakers),
                )
            )

            # Phonemize all texts
            if not isinstance(texts, list):
                texts = [texts]

            batch_phonemes: List[Tuple[str, float, List[int], Optional[int]]] = []
            for text_tuple in texts:
                if isinstance(text_tuple, tuple):
                    text, probability = text_tuple
                else:
                    text, probability = text_tuple, 1.0

                phoneme_ids, clip_index = self.phonemize(
                    text,
                    min_phoneme_count=min_phoneme_count
                )

                batch_phonemes.append((text, probability, phoneme_ids, clip_index))

            generated_samples = []
            num_generated_samples = 0

            for i in range(num_batches):
                this_batch_size = max(min(batch_size, num_samples - i * batch_size), 1)
                speakers = list(itertools.islice(speakers_iterator, 0, this_batch_size))
                slerp_weight, length_scale, noise_scale, noise_scale_w = next(settings_iterator)
                speaker_1 = torch.tensor([s[0] for s in speakers], device=self.device).to(torch.int64)
                speaker_2 = torch.tensor([s[1] for s in speakers], device=self.device).to(torch.int64)

                # Fill the batch
                batch_phoneme_ids = []
                batch_clip_indices = []
                batch_texts = []

                for _ in range(this_batch_size):
                    text, probability, phoneme_ids, clip_index = random.choices(
                        batch_phonemes,
                        weights=[p for _, p, _, _ in batch_phonemes],
                        k=1,
                    )[0]
                    batch_texts.append(text)
                    batch_phoneme_ids.append(phoneme_ids)
                    batch_clip_indices.append(clip_index)

                if not batch_phoneme_ids:
                    raise ValueError("No phoneme ID batches were generated.")

                # Pad the phoneme ID batches to the same length
                self.right_pad(
                    batch_phoneme_ids,
                    value=phoneme_id_map["^"][0]
                )

                # Generate the batch
                audio_tensor, hop_length = self.generate_batch(
                    speaker_1,
                    speaker_2,
                    batch_phoneme_ids,
                    slerp_weight,
                    length_scale,
                    noise_scale,
                    noise_scale_w,
                )

                # Clip the batch when needed; this is done first by overwriting the audio tensor
                # with 0.0 where the clip index is, then later clipping silence from the whole batch
                for i, clip_index in enumerate(batch_clip_indices):
                    if clip_index is not None:
                        first_sample_index = int(hop_length[i].flatten()[:clip_index-1].sum().item())
                        audio_tensor[i, 0, :first_sample_index] = 0.0
                    last_sample_index = int(hop_length[i].flatten().sum().item())
                    audio_tensor[i, 0, last_sample_index+1:] = 0.0

                # Resample the audio
                audio = self.resample(audio_tensor, target_sample_rate).cpu().numpy()
                del audio_tensor

                # Convert from float32 to int16
                audio = audio * (32767.0 / max(0.01, np.max(np.abs(audio))))
                audio = np.clip(audio, -32768, 32767).astype(np.int16)

                # Iterate through batch and trim silence
                for clip, text in zip(audio, batch_texts):
                    clip = np.trim_zeros(clip.flatten())
                    if trim_silence:
                        clip = self.trim_silence(clip.flatten())
                    generated_samples.append((text, clip))
                    num_generated_samples += 1
                    if on_progress is not None:
                        on_progress(num_generated_samples, num_samples)
            if on_progress is not None:
                on_progress(num_samples, num_samples)

            gc.collect()
            torch.cuda.empty_cache()
            return generated_samples

GLOBAL_PIPER_TTS: Dict[Optional[int], PiperTTSModel] = {}
def get_piper_tts_model(device_id: Optional[int] = None) -> PiperTTSModel:
    """
    Get the piper tts model for the given device id.
    """
    if device_id not in GLOBAL_PIPER_TTS:
        GLOBAL_PIPER_TTS[device_id] = PiperTTSModel(device_id=device_id, load=True)
    return GLOBAL_PIPER_TTS[device_id]

def unload_piper_tts_model(device_id: Optional[int] = None) -> None:
    """
    Unloads the piper tts model for the given device id.
    """
    import torch
    if device_id in GLOBAL_PIPER_TTS:
        GLOBAL_PIPER_TTS[device_id].unload()
        del GLOBAL_PIPER_TTS[device_id]
        torch.cuda.empty_cache()
