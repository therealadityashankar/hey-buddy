import os
import gc
import tqdm
import click
import logging
import numpy as np

from typing import Optional, List, Iterator, Any

from contextlib import contextmanager

from heybuddy.constants import *
from heybuddy.dataset import (
    PrecalculatedLabeledTrainingDatasetGenerator,
    WakeWordTrainingDatasetIterator,
)
from heybuddy.util import (
    AppendableNumpyArrayFile,
    WakeWordModelThread,
    safe_name,
    debug_logger,
    logger,
    human_duration
)

PRECALCULATED_DIR = os.path.join(os.path.dirname(__file__), "precalculated")

@contextmanager
def logging_context(debug: bool=False) -> Iterator[None]:
    """
    Context manager for setting up logging.
    """
    with debug_logger(logging.INFO if not debug else logging.DEBUG):
        yield

@click.group()
def main() -> None:
    pass

@main.command()
@click.argument("name", type=str, nargs=1)
@click.argument("repo_id", type=str, nargs=1)
@click.option("--directory", default=PRECALCULATED_DIR, help="Directory to save the embeddings to.", show_default=True)
@click.option("--config", type=str, default=None, help="The configuration name to create the dataset from (when multiple configs are supported.)", show_default=True)
@click.option("--split", type=str, default="train", help="Split to create the dataset from.", show_default=True)
@click.option("--audio-key", type=str, default="audio", help="Key in the dataset for the audio data.", show_default=True)
@click.option("--audio-array-key", type=str, default="array", help="Key in the audio data for the waveform.", show_default=True)
@click.option("--audio-sample-rate-key", type=str, default="sampling_rate", help="Key in the audio data for the sample rate.", show_default=True)
@click.option("--transcript-key", type=str, default="transcript", help="Key in the dataset for the transcript data.", show_default=True)
@click.option("--streaming/--no-streaming", default=True, is_flag=True, help="Stream the dataset, instead of downloading first.", show_default=True)
@click.option("--trust-remote-code/--no-trust-remote-code", default=False, is_flag=True, help="Trust remote code when downloading.", show_default=True)
@click.option("--hours", type=float, default=1000.0, help="Hours of audio to process.", show_default=True)
@click.option("--samples-per-file", type=int, default=10000, help="Number of samples per file.", show_default=True)
@click.option("--device-id", type=int, default=None, help="Device ID to use for processing. None uses CPU.", show_default=True)
@click.option("--sample-rate", type=int, default=16000, help="Sample rate to resample audio to.", show_default=True)
@click.option("--seconds-per-batch", type=float, default=1.44, help="Seconds of audio to process per batch.", show_default=True)
@click.option("--process-batch-size", default=100, help="Batch size for processing audio files.", show_default=True)
@click.option("--embedding-batch-size", default=32, help="Batch size for extracting embeddings.", show_default=True)
@click.option("--tokenizer-max-length", default=96, help="Maximum length for the tokenizer.", show_default=True)
@click.option("--debug/--no-debug", default=False, is_flag=True, help="Enable debug logging.", show_default=True)
def extract(
    name: str,
    repo_id: str,
    directory: str=PRECALCULATED_DIR,
    config: Optional[str]=None,
    split: str="train",
    audio_key: str="audio",
    audio_array_key: str="array",
    audio_sample_rate_key: str="sampling_rate",
    transcript_key: str="transcript",
    streaming: bool=True,
    trust_remote_code: bool=False,
    hours: float=1000.0,
    samples_per_file: int=10000,
    device_id: Optional[int]=None,
    sample_rate: int=16000,
    seconds_per_batch: float=1.44,
    process_batch_size: int=100,
    embedding_batch_size: int=32,
    tokenizer_max_length: int=96,
    debug: bool=False,
) -> None:
    """
    Creates a dataset of speech embeddings from a given repository.
    """
    with logging_context(debug):
        generator = PrecalculatedLabeledTrainingDatasetGenerator(
            dataset_path=repo_id,
            config_name=config,
            split=split,
            audio_key=audio_key,
            audio_array_key=audio_array_key,
            audio_sample_rate_key=audio_sample_rate_key,
            transcript_key=transcript_key,
            device_id=device_id,
            sample_rate=sample_rate,
            seconds_per_batch=seconds_per_batch,
            process_batch_size=process_batch_size,
            embedding_batch_size=embedding_batch_size,
            tokenizer_max_length=tokenizer_max_length
        )

        generator(
            name=name,
            output_dir=directory,
            max_hours=hours,
            dataset_streaming=streaming,
            trust_remote_code=trust_remote_code,
            samples_per_file=samples_per_file,
        )

@main.command()
@click.argument("source", type=str, nargs=-1)
@click.argument("target", type=str, nargs=1)
@click.option("--directory", default=PRECALCULATED_DIR, help="Directory to save the embeddings to.", show_default=True)
@click.option("--reset/--no-reset", default=True, is_flag=True, help="Reset the target file if it already exists.")
@click.option("--half/--no-half", default=False, is_flag=True, help="Use half-precision floating point for the embeddings.", show_default=True)
@click.option("--delete/--no-delete", default=False, is_flag=True, help="Delete source embeddings after combining.")
@click.option("--batch-size", default=10, help="Batch size for reading and appending embeddings. Larger numbers will process faster but consume more memory.", show_default=True)
@click.option("--debug/--no-debug", default=False, is_flag=True, help="Enable debug logging.", show_default=True)
def combine(
    source: List[str],
    target: str,
    directory: str=PRECALCULATED_DIR,
    reset: bool=False,
    half: bool=False,
    delete: bool=False,
    batch_size: int=10,
    debug: bool=False,
) -> None:
    """
    Combines precalculated embeddings from one or more directories into a single .npy file.
    """
    with logging_context(debug):
        target_path = os.path.join(directory, target)
        if os.path.exists(target_path) and reset:
            os.remove(target_path)

        embedding_directories = []
        embedding_files = []

        for name in source:
            embedding_directories.append(os.path.join(directory, name))
            embedding_files.extend([
                os.path.join(embedding_directories[-1], f)
                for f in os.listdir(embedding_directories[-1])
                if f.endswith(".npy")
            ])

        with AppendableNumpyArrayFile(target_path) as array:
            batch: List[np.ndarray[Any, Any]] = []
            for i, filename in enumerate(tqdm.tqdm(sorted(embedding_files))):
                logger.debug(f"Processing {filename} ({i+1}/{len(embedding_files)})")
                data = np.load(filename)
                if half:
                    data = data.astype(np.float16)
                batch.append(data)
                if batch and len(batch) % batch_size == 0:
                    array.append(np.concatenate(batch, axis=0))
                    batch = []
                    gc.collect()
                if delete:
                    os.remove(filename)
            if batch:
                array.append(np.concatenate(batch, axis=0))

        if delete:
            for embedding_directory in embedding_directories:
                os.rmdir(embedding_directory)

@main.command()
@click.argument("phrase", type=str, nargs=1)
@click.option("--additional-phrase", type=str, default=None, multiple=True, help="Additional phrases to use for training.", show_default=True)
@click.option("--wandb-entity", type=str, default=None, help="W&B entity to use for logging.", show_default=True)
@click.option("--perceptron", "architecture", flag_value="perceptron", default=DEFAULT_ARCHITECTURE=="perceptron", help="Use a perceptron architecture.", show_default=True)
@click.option("--transformer", "architecture", flag_value="transformer", default=DEFAULT_ARCHITECTURE=="transformer", help="Use a transformer architecture.", show_default=True)
@click.option("--use-half-layers/--no-use-half-layers", default=DEFAULT_USE_HALF_LAYERS, is_flag=True, help="Use enumerated striped-attention layers for the perceptron model.", show_default=True)
@click.option("--use-gating/--no-use-gating", default=DEFAULT_USE_GATING, is_flag=True, help="Use gated MLP layers for the perceptron model.", show_default=True)
@click.option("--layer-dim", type=int, default=DEFAULT_LAYER_DIM, help="Dimension of the linear layers to use for the model.", show_default=True)
@click.option("--num-layers", type=int, default=DEFAULT_LAYERS, help="The number of perceptron blocks.", show_default=True)
@click.option("--num-heads", type=int, default=DEFAULT_HEADS, help="The number of attention heads to use when using the transformer model.", show_default=True)
@click.option("--steps", type=int, default=DEFAULT_STEPS, help="Number of optimization steps to take.", show_default=True)
@click.option("--stages", type=int, default=DEFAULT_STAGES, help="Number of training stages.", show_default=True)
@click.option("--threshold", type=float, default=DEFAULT_ACTIVATION_THRESHOLD, help="Threshold to use for wake-word detection.", show_default=True)
@click.option("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate for the optimizer.", show_default=True)
@click.option("--high-loss-threshold", type=float, default=DEFAULT_HIGH_LOSS_THRESHOLD, help="Threshold for high loss values (e.g. with the default 0.001, a value is high-loss if it's supposed to be 0 and is higher than 0.001 or supposed to be 1 and is lower than 0.999)", show_default=True)
@click.option("--target-false-positive-rate", type=float, default=DEFAULT_TARGET_FALSE_POSITIVE_RATE, help="Target false positive rate for the model.", show_default=True)
@click.option("--dynamic-negative-weight/--no-dynamic-negative-weight", default=True, is_flag=True, help="Dynamically adjust the negative weight based on target false positive rate at each validation step (instead of in-between stages.)", show_default=True)
@click.option("--negative-weight", type=float, default=DEFAULT_NEGATIVE_WEIGHT, help="Negative weight for the loss function.", show_default=True)
@click.option("--training-full-default-dataset", "training_default_size", flag_value="full", help="Use the full precalculated default training set.", default=True, show_default=True)
@click.option("--training-large-default-dataset", "training_default_size", flag_value="large", help="Use the large precalculated default training set.", default=False, show_default=True)
@click.option("--training-medium-default-dataset", "training_default_size", flag_value="medium", help="Use the medium precalculated default training set.", default=False, show_default=True)
@click.option("--training-no-default-dataset", "training_default_size", flag_value="none", help="Do not use a precalculated default training set.", default=False, show_default=True)
@click.option("--training-dataset", type=click.Path(exists=True, dir_okay=False, file_okay=True), default=None, help="Use a custom precalculated training set.", show_default=True)
@click.option("--augment-phrase-prob", type=float, default=DEFAULT_AUGMENT_PHRASE_PROB, help="Probability of augmenting the phrase.", show_default=True)
@click.option("--augment-phrase-default-words/--augment-phrase-no-default-words", default=True, is_flag=True, help="Use the default words for augmentation.", show_default=True)
@click.option("--augment-phrase-word", type=str, default=None, multiple=True, help="Custom words to use for augmentation.", show_default=True)
@click.option("--augmentation-default-background-dataset/--augmentation-no-default-background-dataset", default=True, is_flag=True, help="Use the default background dataset for augmentation.", show_default=True)
@click.option("--augmentation-background-dataset", default=None, multiple=True, help="Use a custom background dataset for augmentation.", show_default=True)
@click.option("--augmentation-default-impulse-dataset/--augmentation-no-default-impulse-dataset", default=True, is_flag=True, help="Use the default impulse dataset for augmentation.", show_default=True)
@click.option("--augmentation-impulse-dataset", default=None, multiple=True, help="Use a custom impulse dataset for augmentation.", show_default=True)
@click.option("--augmentation-dataset-streaming/--augmentation-dataset-no-streaming", default=False, is_flag=True, help="Stream the augmentation datasets, instead of downloading first.", show_default=True)
@click.option("--augmentation-seven-band-prob", type=float, default=DEFAULT_AUGMENT_SEVEN_BAND_PROB, help="Probability of applying the seven band equalization augmentation.", show_default=True)
@click.option("--augmentation-seven-band-gain-db", type=float, default=DEFAULT_AUGMENT_SEVEN_BAND_GAIN_DB, help="Gain in decibels for the seven band equalization augmentation.", show_default=True)
@click.option("--augmentation-tanh-distortion-prob", type=float, default=DEFAULT_AUGMENT_TANH_DISTORTION_PROB, help="Probability of applying the tanh distortion augmentation.", show_default=True)
@click.option("--augmentation-tanh-distortion-min", type=float, default=DEFAULT_AUGMENT_TANH_MIN_DISTORTION, help="Minimum value for the tanh distortion augmentation.", show_default=True)
@click.option("--augmentation-tanh-distortion-max", type=float, default=DEFAULT_AUGMENT_TANH_MAX_DISTORTION, help="Maximum value for the tanh distortion augmentation.", show_default=True)
@click.option("--augmentation-pitch-shift-prob", type=float, default=DEFAULT_AUGMENT_PITCH_SHIFT_PROB, help="Probability of applying the pitch shift augmentation.", show_default=True)
@click.option("--augmentation-pitch-shift-semitones", type=int, default=DEFAULT_AUGMENT_PITCH_SHIFT_SEMITONES, help="Number of semitones to shift the pitch for the pitch shift augmentation.", show_default=True)
@click.option("--augmentation-band-stop-prob", type=float, default=DEFAULT_AUGMENT_BAND_STOP_PROB, help="Probability of applying the band stop filter augmentation.", show_default=True)
@click.option("--augmentation-colored-noise-prob", type=float, default=DEFAULT_AUGMENT_COLORED_NOISE_PROB, help="Probability of applying the colored noise augmentation.", show_default=True)
@click.option("--augmentation-colored-noise-min-snr-db", type=float, default=DEFAULT_AUGMENT_COLORED_NOISE_MIN_SNR_DB, help="Minimum signal-to-noise ratio for the colored noise augmentation.", show_default=True)
@click.option("--augmentation-colored-noise-max-snr-db", type=float, default=DEFAULT_AUGMENT_COLORED_NOISE_MAX_SNR_DB, help="Maximum signal-to-noise ratio for the colored noise augmentation.", show_default=True)
@click.option("--augmentation-colored-noise-min-f-decay", type=float, default=DEFAULT_AUGMENT_COLORED_NOISE_MIN_F_DECAY, help="Minimum frequency decay for the colored noise augmentation.", show_default=True)
@click.option("--augmentation-colored-noise-max-f-decay", type=float, default=DEFAULT_AUGMENT_COLORED_NOISE_MAX_F_DECAY, help="Maximum frequency decay for the colored noise augmentation.", show_default=True)
@click.option("--augmentation-background-noise-prob", type=float, default=DEFAULT_AUGMENT_BACKGROUND_NOISE_PROB, help="Probability of applying the background noise augmentation.", show_default=True)
@click.option("--augmentation-background-noise-min-snr-db", type=float, default=DEFAULT_AUGMENT_BACKGROUND_NOISE_MIN_SNR_DB, help="Minimum signal-to-noise ratio for the background noise augmentation.", show_default=True)
@click.option("--augmentation-background-noise-max-snr-db", type=float, default=DEFAULT_AUGMENT_BACKGROUND_NOISE_MAX_SNR_DB, help="Maximum signal-to-noise ratio for the background noise augmentation.", show_default=True)
@click.option("--augmentation-gain-prob", type=float, default=DEFAULT_AUGMENT_GAIN_PROB, help="Probability of applying the gain augmentation.", show_default=True)
@click.option("--augmentation-reverb-prob", type=float, default=DEFAULT_AUGMENT_REVERB_PROB, help="Probability of applying the reverb augmentation.", show_default=True)
@click.option("--logging-steps", type=int, default=DEFAULT_LOGGING_STEPS, help="How often to log step details.", show_default=True)
@click.option("--validation-steps", type=int, default=DEFAULT_VALIDATION_STEPS, help="How often to validate the model.", show_default=True)
@click.option("--checkpoint-steps", type=int, default=DEFAULT_CHECKPOINT_STEPS, help="How often to save the model.", show_default=True)
@click.option("--positive-samples", type=int, default=DEFAULT_POSITIVE_SAMPLES, help="Number of positive samples to use for training. Will synthetically generate more when needed.", show_default=True)
@click.option("--adversarial-samples", type=int, default=DEFAULT_ADVERSARIAL_SAMPLES, help="Number of adversarial samples to use for training. Will synthetically generate more when needed.", show_default=True)
@click.option("--adversarial-phrases", type=int, default=DEFAULT_ADVERSARIAL_PHRASES, help="Number of adversarial phrases to use for training. Will synthetically generate more when needed.", show_default=True)
@click.option("--adversarial-phrase-custom", type=str, default=None, multiple=True, help="Custom adversarial phrases to use for training.", show_default=True)
@click.option("--positive-batch-size", type=int, default=DEFAULT_POSITIVE_BATCH_SIZE, help="The number of positive samples to include in each batch during training.", show_default=True)
@click.option("--negative-batch-size", type=int, default=DEFAULT_NEGATIVE_BATCH_SIZE, help="The number of negative samples to include in each batch during training.", show_default=True)
@click.option("--adversarial-batch-size", type=int, default=DEFAULT_ADVERSARIAL_BATCH_SIZE, help="The number of adversarial samples to include in each batch during training.", show_default=True)
@click.option("--num-batch-threads", type=int, default=DEFAULT_BATCH_THREADS, help="The number of threads to spawn for creating training batches.", show_default=True)
@click.option("--validation-positive-batch-size", type=int, default=DEFAULT_VALIDATION_POSITIVE_BATCH_SIZE, help="The number of positive samples to include in each batch during validation.", show_default=True)
@click.option("--validation-negative-batch-size", type=int, default=DEFAULT_VALIDATION_NEGATIVE_BATCH_SIZE, help="The number of negative samples to include in each batch during validation.", show_default=True)
@click.option("--validation-samples", type=int, default=DEFAULT_VALIDATION_SAMPLES, help="The number of samples to use for validation. Will synthetically generate more when needed.", show_default=True)
@click.option("--validation-num-batch-threads", type=int, default=1, help="The number of threads to spawn for creating validation batches.", show_default=True)
@click.option("--validation-default-dataset/--validation-no-default-dataset", default=True, is_flag=True, help="Use the default validation dataset.", show_default=True)
@click.option("--validation-dataset", type=click.Path(exists=True, dir_okay=False, file_okay=True), default=None, help="Use a custom precalculated validation set.", show_default=True)
@click.option("--testing-positive-samples", type=int, default=DEFAULT_TESTING_POSITIVE_SAMPLES, help="The number of positive samples to use for testing. Will synthetically generate more when needed.", show_default=True)
@click.option("--testing-adversarial-samples", type=int, default=DEFAULT_TESTING_ADVERSARIAL_SAMPLES, help="The number of adversarial samples to use for testing. Will synthetically generate more when needed.", show_default=True)
@click.option("--testing-positive-batch-size", type=int, default=None, help="The number of positive samples to include in each batch during testing. Default matches the size used during training.", show_default=True)
@click.option("--testing-adversarial-batch-size", type=int, default=None, help="The number of adversarial samples to include in each batch during testing. Default matches the size used during training.", show_default=True)
@click.option("--testing-num-batch-threads", type=int, default=1, help="The number of threads to spawn for creating testing batches.", show_default=True)
@click.option("--resume/--no-resume", default=False, is_flag=True, help="Resume training from the last checkpoint.", show_default=True)
@click.option("--debug/--no-debug", default=False, is_flag=True, help="Enable debug logging.", show_default=True)
def train(
    phrase: str,
    additional_phrase: List[str]=[],
    wandb_entity: Optional[str]=None,
    architecture: str=DEFAULT_ARCHITECTURE,
    use_half_layers: bool=DEFAULT_USE_HALF_LAYERS,
    use_gating: bool=DEFAULT_USE_GATING,
    layer_dim: int=DEFAULT_LAYER_DIM,
    num_layers: int=DEFAULT_LAYERS,
    num_heads: int=DEFAULT_HEADS,
    steps: int=DEFAULT_STEPS,
    stages: int=DEFAULT_STAGES,
    threshold: float=DEFAULT_ACTIVATION_THRESHOLD,
    learning_rate: float=DEFAULT_LEARNING_RATE,
    high_loss_threshold: float=DEFAULT_HIGH_LOSS_THRESHOLD,
    target_false_positive_rate: float=DEFAULT_TARGET_FALSE_POSITIVE_RATE,
    dynamic_negative_weight: bool=True,
    negative_weight: float=DEFAULT_NEGATIVE_WEIGHT,
    training_default_size: str="full",
    training_dataset: Optional[str]=None,
    augment_phrase_prob: float=DEFAULT_AUGMENT_PHRASE_PROB,
    augment_phrase_default_words: bool=True,
    augment_phrase_word: List[str]=[],
    augmentation_default_background_dataset: bool=True,
    augmentation_background_dataset: List[str]=[],
    augmentation_default_impulse_dataset: bool=True,
    augmentation_impulse_dataset: List[str]=[],
    augmentation_dataset_streaming: bool=False,
    augmentation_seven_band_prob: float=DEFAULT_AUGMENT_SEVEN_BAND_PROB,
    augmentation_seven_band_gain_db: float=DEFAULT_AUGMENT_SEVEN_BAND_GAIN_DB,
    augmentation_tanh_distortion_prob: float=DEFAULT_AUGMENT_TANH_DISTORTION_PROB,
    augmentation_tanh_distortion_min: float=DEFAULT_AUGMENT_TANH_MIN_DISTORTION,
    augmentation_tanh_distortion_max: float=DEFAULT_AUGMENT_TANH_MAX_DISTORTION,
    augmentation_pitch_shift_prob: float=DEFAULT_AUGMENT_PITCH_SHIFT_PROB,
    augmentation_pitch_shift_semitones: int=DEFAULT_AUGMENT_PITCH_SHIFT_SEMITONES,
    augmentation_band_stop_prob: float=DEFAULT_AUGMENT_BAND_STOP_PROB,
    augmentation_colored_noise_prob: float=DEFAULT_AUGMENT_COLORED_NOISE_PROB,
    augmentation_colored_noise_min_snr_db: float=DEFAULT_AUGMENT_COLORED_NOISE_MIN_SNR_DB,
    augmentation_colored_noise_max_snr_db: float=DEFAULT_AUGMENT_COLORED_NOISE_MAX_SNR_DB,
    augmentation_colored_noise_min_f_decay: float=DEFAULT_AUGMENT_COLORED_NOISE_MIN_F_DECAY,
    augmentation_colored_noise_max_f_decay: float=DEFAULT_AUGMENT_COLORED_NOISE_MAX_F_DECAY,
    augmentation_background_noise_prob: float=DEFAULT_AUGMENT_BACKGROUND_NOISE_PROB,
    augmentation_background_noise_min_snr_db: float=DEFAULT_AUGMENT_BACKGROUND_NOISE_MIN_SNR_DB,
    augmentation_background_noise_max_snr_db: float=DEFAULT_AUGMENT_BACKGROUND_NOISE_MAX_SNR_DB,
    augmentation_gain_prob: float=DEFAULT_AUGMENT_GAIN_PROB,
    augmentation_reverb_prob: float=DEFAULT_AUGMENT_REVERB_PROB,
    logging_steps: int=DEFAULT_LOGGING_STEPS,
    validation_steps: int=DEFAULT_VALIDATION_STEPS,
    checkpoint_steps: int=DEFAULT_CHECKPOINT_STEPS,
    positive_samples: int=DEFAULT_POSITIVE_SAMPLES,
    adversarial_samples: int=DEFAULT_ADVERSARIAL_SAMPLES,
    adversarial_phrases: int=DEFAULT_ADVERSARIAL_PHRASES,
    adversarial_phrase_custom: List[str]=[],
    positive_batch_size: int=DEFAULT_POSITIVE_BATCH_SIZE,
    negative_batch_size: int=DEFAULT_NEGATIVE_BATCH_SIZE,
    adversarial_batch_size: int=DEFAULT_ADVERSARIAL_BATCH_SIZE,
    num_batch_threads: int=DEFAULT_BATCH_THREADS,
    validation_positive_batch_size: int=DEFAULT_VALIDATION_POSITIVE_BATCH_SIZE,
    validation_negative_batch_size: int=DEFAULT_VALIDATION_NEGATIVE_BATCH_SIZE,
    validation_samples: int=DEFAULT_VALIDATION_SAMPLES,
    validation_num_batch_threads: int=1,
    validation_default_dataset: bool=True,
    validation_dataset: Optional[str]=None,
    testing_positive_samples: int=DEFAULT_TESTING_POSITIVE_SAMPLES,
    testing_adversarial_samples: int=DEFAULT_TESTING_ADVERSARIAL_SAMPLES,
    testing_positive_batch_size: Optional[int]=None,
    testing_adversarial_batch_size: Optional[int]=None,
    testing_num_batch_threads: int=1,
    resume: bool=False,
    debug: bool=False,
) -> None:
    """
    Trains a wake word detection model.
    """
    phrase_augment_words = []
    augment_background_datasets = []
    augment_impulse_datasets = []

    if augment_phrase_default_words:
        phrase_augment_words.extend(DEFAULT_AUGMENT_PHRASE_WORDS)
    if augment_phrase_word:
        phrase_augment_words.extend(augment_phrase_word)

    if augmentation_default_background_dataset:
        if isinstance(DEFAULT_BACKGROUND_DATASET, str):
            augment_background_datasets.append(DEFAULT_BACKGROUND_DATASET)
        else:
            augment_background_datasets.extend(DEFAULT_BACKGROUND_DATASET)
    if augmentation_background_dataset:
        augment_background_datasets.extend(augmentation_background_dataset)
    
    if augmentation_default_impulse_dataset:
        if isinstance(DEFAULT_IMPULSE_DATASET, str):
            augment_impulse_datasets.append(DEFAULT_IMPULSE_DATASET)
        else:
            augment_impulse_datasets.extend(DEFAULT_IMPULSE_DATASET)
    if augmentation_impulse_dataset:
        augment_impulse_datasets.extend(augmentation_impulse_dataset)

    with logging_context(debug):
        training, validation, testing = WakeWordTrainingDatasetIterator.all(
            additional_wake_phrases=additional_phrase,
            adversarial_per_batch=adversarial_batch_size,
            augment_background_dataset=augment_background_datasets,
            augment_background_noise_max_snr_db=augmentation_background_noise_max_snr_db,
            augment_background_noise_min_snr_db=augmentation_background_noise_min_snr_db,
            augment_background_noise_prob=augmentation_background_noise_prob,
            augment_band_stop_prob=augmentation_band_stop_prob,
            augment_colored_noise_max_f_decay=augmentation_colored_noise_max_f_decay,
            augment_colored_noise_max_snr_db=augmentation_colored_noise_max_snr_db,
            augment_colored_noise_min_f_decay=augmentation_colored_noise_min_f_decay,
            augment_colored_noise_min_snr_db=augmentation_colored_noise_min_snr_db,
            augment_colored_noise_prob=augmentation_colored_noise_prob,
            augment_dataset_streaming=augmentation_dataset_streaming,
            augment_gain_prob=augmentation_gain_prob,
            augment_impulse_dataset=augment_impulse_datasets,
            augment_pitch_shift_prob=augmentation_pitch_shift_prob,
            augment_pitch_shift_semitones=augmentation_pitch_shift_semitones,
            augment_reverb_prob=augmentation_reverb_prob,
            augment_seven_band_gain_db=augmentation_seven_band_gain_db,
            augment_seven_band_prob=augmentation_seven_band_prob,
            augment_tanh_distortion_prob=augmentation_tanh_distortion_prob,
            augment_tanh_max_distortion=augmentation_tanh_distortion_max,
            augment_tanh_min_distortion=augmentation_tanh_distortion_min,
            custom_adversarial_phrases=adversarial_phrase_custom,
            custom_training=training_dataset,
            large_training=training_default_size in ["full", "large"],
            medium_training=training_default_size in ["full", "medium"],
            negative_per_batch=negative_batch_size,
            num_adversarial_phrases=adversarial_phrases,
            num_adversarial_samples=adversarial_samples,
            num_batch_threads=num_batch_threads,
            num_positive_samples=positive_samples,
            phrase_augment_prob=augment_phrase_prob,
            phrase_augment_words=phrase_augment_words,
            positive_per_batch=positive_batch_size,
            testing_adversarial_per_batch=testing_adversarial_batch_size,
            testing_num_adversarial_samples=testing_adversarial_samples,
            testing_num_batch_threads=testing_num_batch_threads,
            testing_num_positive_samples=testing_positive_samples,
            testing_positive_per_batch=testing_positive_batch_size,
            validation_custom=validation_dataset,
            validation_include_precalculated=validation_default_dataset,
            validation_negative_batch_size=validation_negative_batch_size,
            validation_num_batch_threads=validation_num_batch_threads,
            validation_num_positive_samples=validation_samples,
            validation_positive_batch_size=validation_positive_batch_size,
            wake_phrase=phrase,
        )

        import torch
        from heybuddy.trainer import WakeWordTrainer

        trainer = WakeWordTrainer(
            architecture=architecture, # type: ignore[arg-type]
            use_half_layers=use_half_layers,
            use_gating=use_gating,
            layer_dim=layer_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        if torch.cuda.is_available():
            trainer.to("cuda")

        name = safe_name(phrase)
        if resume:
            trainer.resume(name)

        trainer(
            activation_threshold=threshold,
            checkpoint_steps=checkpoint_steps,
            dynamic_negative_weight=dynamic_negative_weight,
            high_loss_threshold=high_loss_threshold,
            learning_rate=learning_rate,
            max_negative_weight=negative_weight,
            name=name,
            num_stages=stages,
            num_steps=steps,
            target_false_positive_rate=target_false_positive_rate,
            testing=testing,
            training=training,
            validation=validation,
            validation_steps=validation_steps,
            wandb_entity=wandb_entity,
        )

@main.command()
@click.argument("checkpoint", type=click.Path(exists=True, dir_okay=False, file_okay=True), nargs=1)
@click.argument("audio", type=click.Path(exists=True, dir_okay=False, file_okay=True), nargs=1)
@click.option("--threshold", type=float, default=DEFAULT_ACTIVATION_THRESHOLD, help="Threshold to use for wake-word detection.", show_default=True)
@click.option("--device-id", type=int, default=None, help="Device ID to use for processing. None uses CPU.", show_default=True)
@click.option("--debug/--no-debug", default=False, is_flag=True, help="Enable debug logging.", show_default=True)
def predict(
    checkpoint: str,
    audio: str,
    threshold: float=DEFAULT_ACTIVATION_THRESHOLD,
    device_id: Optional[int]=None,
    debug: bool=False
) -> None:
    """
    Predicts wake word times in an audio file using a single checkpoint.
    """
    with logging_context(debug):
        import torch
        from heybuddy.wakeword import WakeWordMLPModel
        if torch.cuda.is_available() and device_id is not None:
            device = torch.device(f"cuda:{device_id}")
        else:
            device = torch.device("cpu")
        model = WakeWordMLPModel.from_file(checkpoint, device=device)
        model.eval()
        predicted_times = model.predict_timecodes(audio, threshold=threshold)
        if not predicted_times:
            print("No wake-word utterances detected")
        elif len(predicted_times) == 1:
            print(f"Wake-word utterance detected at {predicted_times[0]:.1f} second(s)")
        else:
            print(f"{len(predicted_times)} wake-word utterances detected at the following times:")
            for predicted_time in predicted_times:
                print(f"  {predicted_time:.1f} second(s)")

@main.command()
@click.argument("checkpoints", type=click.Path(exists=True, dir_okay=False, file_okay=True), nargs=-1)
@click.option("--threshold", type=float, default=DEFAULT_ACTIVATION_THRESHOLD, help="Threshold to use for wake-word detection.", show_default=True)
@click.option("--buffer-size", type=int, default=4096, help="Size of the audio buffer to read.", show_default=True)
@click.option("--device-id", type=int, default=None, help="Device ID to use for processing. None uses CPU.", show_default=True)
@click.option("--debug/--no-debug", default=False, is_flag=True, help="Enable debug logging.", show_default=True)
def listen(
    checkpoints: List[str],
    threshold: float=DEFAULT_ACTIVATION_THRESHOLD,
    buffer_size: int=4096,
    device_id: Optional[int]=None,
    debug: bool=False
) -> None:
    """
    Listens for wake words in real-time using one or more checkpoints.
    """
    try:
        import pyaudio
    except ImportError:
        print("Please install the pyaudio package to use the listen command.")
        return

    import torch
    import torchaudio # type: ignore[import-untyped]

    with logging_context(debug):
        audio = pyaudio.PyAudio()
        device_index = None
        num_devices = audio.get_device_count()

        if num_devices > 1:
            print("Multiple audio devices detected. Please specify the device ID to use.")
            while device_index is None:
                print("The available devices are:")
                for i in range(num_devices):
                    print(f"  {i}: {audio.get_device_info_by_index(i)['name']}")
                try:
                    device_index = int(input("Enter the device ID to use: "))
                    if device_index < 0 or device_index >= num_devices:
                        raise ValueError(device_index)
                except ValueError as e:
                    print(f"Invalid device ID: {e}")
                    device_index = None

        sample_rates = [16000, 44100, 48000]
        for i, sample_rate in enumerate(sample_rates):
            try:
                if audio.is_format_supported(
                    sample_rate,
                    input_device=device_index,
                    input_channels=1,
                    input_format=pyaudio.paInt16
                ):
                    break
            except ValueError:
                if i == len(sample_rates) - 1:
                    raise ValueError("No supported sample rates found")

        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=buffer_size,
            input_device_index=device_index,
        )
        model_names = [
            os.path.splitext(os.path.basename(checkpoint))[0]
            for checkpoint in checkpoints
        ]
        max_model_name_length = max(len(model_name) for model_name in model_names)
        table_header = f"{'Model':<{max_model_name_length}} | {'Score':>6} | Active | Activations | Duration"

        threads = [
            WakeWordModelThread(
                checkpoint=checkpoint,
                return_scores=True,
            )
            for checkpoint in checkpoints
        ]

        for thread in threads:
            thread.start()

        if sample_rate != 16000:
            import torchaudio
            resample = torchaudio.transforms.Resample(48000, 16000)
        else:
            resample = lambda x: x

        num_activations = [0] * len(threads)
        last_frame: Optional[torch.Tensor] = None
        max_samples = 32000

        try:
            while True:
                data = torch.frombuffer(bytearray(stream.read(buffer_size, exception_on_overflow=False)), dtype=torch.int16)
                data = data.float().div_(32768.0)
                data = resample(data)

                if last_frame is not None:
                    data = torch.cat([last_frame, data], dim=0)[-max_samples:]

                for thread in threads:
                    thread.put(data.clone())

                prediction_results = [
                    thread.get()
                    for thread in threads
                ]

                table_lines = [
                    table_header,
                    "-" * len(table_header)
                ]

                for i, (model_name, (prediction_result, duration)) in enumerate(zip(model_names, prediction_results)):
                    is_activated = prediction_result[0] > threshold
                    num_activations[i] += int(is_activated)
                    table_lines.append(f"{model_name:<{max_model_name_length}} | {prediction_result[0]:>6.3f} | {'Yes' if is_activated else 'No':>6} | {num_activations[i]:>11} | {human_duration(duration):>8}")

                print("\033[F"*len(table_lines))
                print("\n".join(table_lines), end="")

                last_frame = data

        except Exception as e:
            for thread in threads:
                thread.stop()
            for thread in threads:
                thread.join()
            raise e

@main.command()
@click.argument("checkpoint", type=click.Path(exists=True, dir_okay=False, file_okay=True), nargs=1)
@click.option("-v", "--opset-version", type=int, default=19, help="ONNX opset version to use.", show_default=True)
@click.option("-o", "--output", type=click.Path(exists=False, dir_okay=False, file_okay=True), default=None, help="Output file for the ONNX model.", show_default=True)
def convert(
    checkpoint: str,
    opset_version: int=23,
    output: Optional[str]=None
) -> None:
    """
    Converts a checkpoint to ONNX.
    """
    from heybuddy.wakeword import WakeWordMLPModel
    filename = os.path.splitext(os.path.basename(checkpoint))[0]
    if output is not None:
        destination = output
    else:
        destination = os.path.join(
            os.path.dirname(checkpoint),
            f"{filename}.onnx"
        )
    if os.path.exists(destination):
        os.remove(destination)

    model = WakeWordMLPModel.from_file(checkpoint)
    model.save_onnx(destination, opset_version=opset_version)
    click.echo(f"Model saved to {destination}")

if __name__ == '__main__':
    main()
