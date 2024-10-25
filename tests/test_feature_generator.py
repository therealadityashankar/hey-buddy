import psutil
from heybuddy.dataset import TrainingFeaturesGenerator

def test_features_generator() -> None:
    """
    Test the features generator function at a baseline.
    """
    generator = TrainingFeaturesGenerator()
    samples = generator(1)
    assert samples.shape == (1, 16, 96)

def test_features_generator_gpu_and_memory() -> None:
    """
    Test the features generator function's memory usage.
    """
    memory_at_start = psutil.virtual_memory().used
    generator = TrainingFeaturesGenerator(
        device="cuda",
        sample_batch_size=5000,
        tts_batch_size=64,
        tts_num_threads=2,
        augment_batch_size=128,
        augment_num_threads=2
    )
    samples = generator(10000)
    assert samples.shape == (10000, 16, 96)
    memory_at_end = psutil.virtual_memory().used
    memory_used = memory_at_end - memory_at_start
    assert memory_used < 1000000000 # 1 GB
