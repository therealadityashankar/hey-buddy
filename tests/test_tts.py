import os
import psutil
from heybuddy.util import debug_logger, audio_write
from heybuddy.dataset import PiperSpeechGenerator
from heybuddy.dataset import AugmentedAudioGenerator

def test_speech_generator() -> None:
    """
    Test the speech generator.
    """
    with debug_logger():
        cpu_generator = PiperSpeechGenerator(
            phrase="hello world",
        )(num_samples=1)
        result = next(iter(cpu_generator))
        assert "audio" in result
        audio_write(
            os.path.join(os.getcwd(), "test"),
            result["audio"]["array"],
            sample_rate=16000,
            format="wav"
        )

def test_augmented_speech_generator() -> None:
    """
    Test the speech generator.
    """
    with debug_logger():
        gpu_generator = PiperSpeechGenerator(
            phrase="hello world",
            device_id=0
        )(num_samples=1)
        augmented_generator = AugmentedAudioGenerator(
            source_dataset=gpu_generator,
            device_id=0
        )(num_samples=1)
        result = next(iter(augmented_generator))
        assert "audio" in result
        audio_write(
            os.path.join(os.getcwd(), "test-augmented"),
            result["audio"]["array"],
            sample_rate=16000,
            format="wav"
        )
