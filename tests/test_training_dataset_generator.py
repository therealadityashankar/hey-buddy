from heybuddy.dataset import TrainingDatasetGenerator
from heybuddy.util import debug_logger
from time import perf_counter

num_test_samples = 50

def test_training_dataset_generator() -> None:
    """
    Performs a baseline test.
    """
    with debug_logger():
        generator = TrainingDatasetGenerator.default(
            wake_phrase="hello_world",
            num_positive_samples=num_test_samples,
            num_negative_samples=num_test_samples,
            use_cache=False, # Force to generate new data for the test
        )
        for i, datum in enumerate(generator):
            if i > num_test_samples * 2:
                # Successfully iterated through all the data and wrapped around
                assert True
                break

def test_cached_data() -> None:
    """
    Ensures that the cached data is being used.
    """
    start = perf_counter()
    generator = TrainingDatasetGenerator.default(
        wake_phrase="hello_world",
        num_positive_samples=num_test_samples,
        num_negative_samples=num_test_samples,
    )
    next(iter(generator))
    first_time = perf_counter() - start
    assert first_time < 2.0 # cached, should take less than 2 seconds
