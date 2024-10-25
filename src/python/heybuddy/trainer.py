from __future__ import annotations

import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import numpy as np
import wandb

from typing import Any, Tuple, Union, Optional, List, Dict
from typing_extensions import Literal

from tqdm import tqdm
from time import perf_counter
from datasets import Dataset # type: ignore[import-untyped]

from heybuddy.constants import *
from heybuddy.wakeword import WakeWordMLPModel, WakeWordTransformerModel
from heybuddy.embeddings import SpeechEmbeddings
from heybuddy.dataset import TrainingDatasetIterator, WakeWordTrainingDatasetIterator
from heybuddy.util import logger, human_duration

DatasetType = Union[Dataset, TrainingDatasetIterator]

class Trainer(nn.Module):
    """
    Base class for training models.
    """
    def __init__(
        self,
        checkpoint_dir: str="./checkpoints",
        learning_rate: float=DEFAULT_LEARNING_RATE,
        **model_kwargs: Any,
    ) -> None:
        """
        :param checkpoint_dir: The directory in which to save checkpoints.
        :param learning_rate: The learning rate for the optimizer.
        """
        super(Trainer, self).__init__()
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.model = self.create_model(**model_kwargs)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate) # type: ignore[attr-defined]
        self.learning_rate = learning_rate

    def create_model(self, **kwargs: Any) -> nn.Module:
        """
        Create the model.
        """
        raise NotImplementedError()

    def resume(self, name: str) -> None:
        """
        Resume training from a checkpoint.
        """
        import codecs
        torch.serialization.add_safe_globals([
            np.ndarray,
            np.dtype,
            np.dtypes.Float64DType,
            np.core.multiarray._reconstruct, # type: ignore[attr-defined]
            codecs.encode
        ])
        model_checkpoints = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith(name) and f.endswith(".pt") and not f.endswith("_optimizer.pt")
        ]
        model_checkpoint_times = [
            (f, os.path.getmtime(os.path.join(self.checkpoint_dir, f)))
            for f in model_checkpoints
        ]
        optimizer_checkpoints = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith(name) and f.endswith("_optimizer.pt")
        ]
        optimizer_checkpoint_times = [
            (f, os.path.getmtime(os.path.join(self.checkpoint_dir, f)))
            for f in optimizer_checkpoints
        ]

        # Get the latest model/optimizer pair
        if not model_checkpoints or not optimizer_checkpoints:
            raise FileNotFoundError(f"Checkpoint {name} not found.")

        model_checkpoint_times.sort(key=lambda x: x[1], reverse=True)
        optimizer_checkpoint_times.sort(key=lambda x: x[1], reverse=True)

        resume_model_checkpoint: Optional[str] = None
        resume_optimizer_checkpoint: Optional[str] = None

        for i, (model_checkpoint, model_checkpoint_time) in enumerate(model_checkpoint_times):
            for j, (optimizer_checkpoint, optimizer_checkpoint_time) in enumerate(optimizer_checkpoint_times):
                # Check if times are within 2 seconds of each other
                if abs(model_checkpoint_time - optimizer_checkpoint_time) < 2:
                    resume_model_checkpoint = model_checkpoint
                    resume_optimizer_checkpoint = optimizer_checkpoint
                    break
            if resume_model_checkpoint is not None and resume_optimizer_checkpoint is not None:
                break

        if resume_model_checkpoint is None or resume_optimizer_checkpoint is None:
            raise FileNotFoundError(f"Checkpoint {name} not found.")

        logger.info(f"Resuming training from {resume_model_checkpoint} and {resume_optimizer_checkpoint}.")
        self.model.load_state_dict(
            torch.load(
                os.path.join(self.checkpoint_dir, resume_model_checkpoint),
                weights_only=True
            )
        )
        self.optimizer.load_state_dict(
            torch.load(
                os.path.join(self.checkpoint_dir, resume_optimizer_checkpoint),
                weights_only=True
            )
        )

    @property
    def device(self) -> torch.device:
        """
        Get the device on which the model is located.
        """
        return next(self.parameters()).device

    def get_learning_rate(
        self,
        step: int,
        warmup_steps: int=0,
        hold_steps: int=0,
        total_steps: int=0,
        start_learning_rate: float=0.0,
        target_learning_rate: float=DEFAULT_LEARNING_RATE,
    ) -> np.ndarray[Any, Any]:
        """
        Gets the learning rate for the current step.
        Uses cosine decay with warmup and hold steps.
        """
        learning_rate = 0.5 * target_learning_rate * (1 + np.cos(
            np.pi * (step - warmup_steps - hold_steps) /
            float(total_steps - warmup_steps - hold_steps)
        ))
        if warmup_steps > 0:
            warmup_learning_rate = target_learning_rate * (step / warmup_steps)
        else:
            warmup_learning_rate = 0.0

        if hold_steps > 0:
            learning_rate = np.where(
                step > warmup_steps + hold_steps,
                learning_rate,
                target_learning_rate
            )

        return np.where(step < warmup_steps, warmup_learning_rate, learning_rate)

    def log_datasets(
        self,
        training: DatasetType,
        validation: Optional[DatasetType]=None,
        testing: Optional[DatasetType]=None,
    ) -> None:
        """
        Log the datasets.
        """
        if isinstance(training, WakeWordTrainingDatasetIterator):
            logger.info(f"Training dataset summary: {training.summary()}")
        if isinstance(validation, WakeWordTrainingDatasetIterator):
            logger.info(f"Validation dataset summary: {validation.summary()}")
        if isinstance(testing, WakeWordTrainingDatasetIterator):
            logger.info(f"Testing dataset summary: {testing.summary()}")

    def log_tensor_metrics(self, description: str, tensor: torch.Tensor) -> None:
        """
        Logs metrics for one tensor.
        """
        t_start = tensor[0]
        t_end = tensor[-1]
        t_min, t_max = tensor.aminmax()
        t_mean = tensor.mean()
        t_std = tensor.std()
        t_slope = (t_end - t_start) / tensor.shape[0]
        logger.info(f"{description}: Start: {t_start}, End: {t_end}, Min: {t_min}, Max: {t_max}, Mean: {t_mean}, Std: {t_std}, Slope: {t_slope}")

    def save_checkpoint(
        self,
        name: str,
        optimizer: bool=True,
    ) -> None:
        """
        Save a checkpoint of the model.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{name}.pt")
        torch.save(self.model.state_dict(), checkpoint_path)
        if optimizer:
            optimizer_path = os.path.join(self.checkpoint_dir, f"{name}_optimizer.pt")
            torch.save(self.optimizer.state_dict(), optimizer_path)

    def __call__(self, training: DatasetType, **kwargs: Any) -> None:
        """
        Train the model.
        """
        raise NotImplementedError()

class WakeWordTrainer(Trainer):
    """
    Trainer class for the model.
    """
    def __init__(
        self,
        checkpoint_dir: str="./checkpoints",
        learning_rate: float=DEFAULT_LEARNING_RATE,
        input_shape: Tuple[int, int]=(16, 96),
        num_layers: int=DEFAULT_LAYERS,
        layer_dim: int=DEFAULT_LAYER_DIM,
        num_heads: int=DEFAULT_HEADS,
        architecture: Literal["perceptron", "transformer"]=DEFAULT_ARCHITECTURE, # type: ignore[assignment]
        **model_kwargs: Any,
    ) -> None:
        super(WakeWordTrainer, self).__init__(
            checkpoint_dir=checkpoint_dir,
            learning_rate=learning_rate,
            input_shape=input_shape,
            num_layers=num_layers,
            layer_dim=layer_dim,
            num_heads=num_heads,
            architecture=architecture,
            **model_kwargs,
        )
        self.input_shape = input_shape
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.architecture = architecture
        self.layer_dim = layer_dim

    def create_model(
        self,
        input_shape: Tuple[int, int]=(16, 96),
        architecture: Literal["perceptron", "transformer"]=DEFAULT_ARCHITECTURE, # type: ignore[assignment]
        layer_dim: int=DEFAULT_LAYER_DIM,
        num_layers: int=DEFAULT_LAYERS,
        num_heads: int=DEFAULT_HEADS,
        use_gating: bool=DEFAULT_USE_GATING,
        use_half_layers: bool=DEFAULT_USE_HALF_LAYERS,
        **kwargs: Any
    ) -> nn.Module:
        """
        Create the model.
        """
        if architecture == "perceptron":
            return WakeWordMLPModel(
                input_shape=input_shape,
                num_layers=num_layers,
                layer_dim=layer_dim,
                use_gating=use_gating,
                use_half_layers=use_half_layers,
                **kwargs,
            )
        elif architecture == "transformer":
            return WakeWordTransformerModel(
                dim=layer_dim,
                input_shape=input_shape,
                num_layers=num_layers,
                num_heads=num_heads,
            )
        raise ValueError(f"Invalid architecture: {architecture}")

    @property
    def embeddings(self) -> SpeechEmbeddings:
        """
        Get the embeddings model.
        """
        if not hasattr(self, "_embeddings"):
            self._embeddings = SpeechEmbeddings(
                device_id=(self.device.index or 0) if self.device.type != "cpu" else None
            )
        return self._embeddings

    def get_features(self, datum: Dict[str, Any]) -> np.ndarray[Any, Any]:
        """
        Get the features for the datum.
        """
        if "features" in datum:
            return datum["features"] # type: ignore[no-any-return]
        if "audio" not in datum:
            raise ValueError("Datum must contain either 'features' or 'audio'.")
        return self.embeddings(datum["audio"]["array"]) # type: ignore[return-value]

    def num_false_positives(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        activation_threshold: float=DEFAULT_ACTIVATION_THRESHOLD,
    ) -> torch.Tensor:
        """
        Compute the false positive rate for the given predictions and labels.
        """
        return (y - x <= -activation_threshold).sum()

    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        weight: Optional[torch.Tensor]=None,
    ) -> torch.Tensor:
        """
        Compute the binary cross entropy loss between the predicted and target values.
        """
        if weight is None:
            return nn.functional.binary_cross_entropy(x, y)
        return nn.functional.binary_cross_entropy(x, y, weight.to(self.device))

    def train_epoch(
        self,
        training: DatasetType,
        validation: Optional[DatasetType]=None,
        testing: Optional[DatasetType]=None,
        num_steps: int=DEFAULT_STEPS,
        warmup_steps: int=DEFAULT_WARMUP_STEPS, # Recommened to be 20% of num_steps
        hold_steps: int=DEFAULT_HOLD_STEPS, # Recommened to be 1/3 of num_steps
        negative_weight_schedule: Union[float,List[float]]=DEFAULT_NEGATIVE_WEIGHT,
        negative_weight_adjust_ratio: Optional[float]=None, # When passed, will update the negative weight schedule at each step
        target_false_positive_rate: float=DEFAULT_TARGET_FALSE_POSITIVE_RATE, # Target false positives per hour
        validation_steps: int=DEFAULT_VALIDATION_STEPS, # How often to validate
        checkpoint_steps: int=DEFAULT_CHECKPOINT_STEPS, # How often to save a checkpoint
        logging_steps: int=DEFAULT_LOGGING_STEPS, # How often to log metrics
        learning_rate: float=DEFAULT_LEARNING_RATE,
        high_loss_threshold: float=DEFAULT_HIGH_LOSS_THRESHOLD,
        activation_threshold: float=DEFAULT_ACTIVATION_THRESHOLD,
        description: str="Training",
        name: str="heybuddy",
        last_loss: float=0.0,
        last_recall: float=0.0,
        last_false_positive_rate: float=0.0,
        last_validation_false_positive_per_hour: float=0.0,
        last_validation_recall: float=0.0,
        last_testing_accuracy: float=0.0,
        last_testing_recall: float=0.0,
        last_testing_false_positive_rate: float=0.0,
        use_wandb: bool=False,
    ) -> Tuple[
        torch.Tensor, # Learning rate history
        torch.Tensor, # Negative weight history
        torch.Tensor, # Loss history
        torch.Tensor, # High loss rate history
        torch.Tensor, # Recall history
        torch.Tensor, # False positive rate history
        Optional[torch.Tensor], # Validation false positive per hour
        Optional[torch.Tensor], # Validation recall history
        Optional[torch.Tensor], # Testing accuracy history
        Optional[torch.Tensor], # Testing recall history
        Optional[torch.Tensor], # Testing false positive rate history,
    ]:
        """
        Perform a single epoch of training.
        """
        accumulation_steps = 1
        accumulated_samples = 0
        accumulated_predictions = torch.Tensor().to(self.device)
        accumulated_labels = torch.Tensor().to(self.device)

        recall = torchmetrics.Recall(task="binary", threshold=activation_threshold).to(self.device)
        accuracy = torchmetrics.Accuracy(task="binary", threshold=activation_threshold).to(self.device)

        learning_rate_history: List[torch.Tensor] = []
        negative_weight_history: List[torch.Tensor] = []
        loss_history: List[torch.Tensor] = []
        high_loss_rate_history: List[torch.Tensor] = []
        recall_history: List[torch.Tensor] = []
        false_positive_rate_history: List[torch.Tensor] = []

        testing_accuracy_history: List[torch.Tensor] = []
        testing_recall_history: List[torch.Tensor] = []
        testing_false_positive_rate_history: List[torch.Tensor] = []

        validation_false_positive_per_hour: List[torch.Tensor] = []
        validation_recall_history: List[torch.Tensor] = []

        for step, datum in tqdm(enumerate(training), total=num_steps, desc=description):
            # Training set can be longer than num_steps, so break if we reach the requested steps
            if step >= num_steps:
                break

            # Get features and label from datum
            x, y = datum[0].to(self.device), datum[1].to(self.device)
            # Gather details to log to wandb
            step_details: Dict[str, Any] = {}
            step_learning_rate = self.get_learning_rate(
                step,
                warmup_steps=warmup_steps,
                hold_steps=hold_steps,
                total_steps=num_steps,
                target_learning_rate=learning_rate,
            )
            step_details["learning_rate"] = step_learning_rate
            learning_rate_history.append(torch.tensor(step_learning_rate))

            # Update learning rate for optimizer
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = step_learning_rate

            # Zero the gradients
            self.optimizer.zero_grad()
            y_pred = self.model(x)

            # Gather high loss values
            negative_high_loss = y_pred[(y==0) & (y_pred.squeeze() >= high_loss_threshold)]
            positive_high_loss = y_pred[(y==1) & (y_pred.squeeze() < 1 - high_loss_threshold)]
            high_loss_rate = (negative_high_loss.shape[0] + positive_high_loss.shape[0]) / y_pred.shape[0]
            high_loss_rate_history.append(torch.tensor(high_loss_rate))
            step_details["high_loss_rate"] = high_loss_rate

            # Get labels associated with high loss values
            y = torch.cat([
                y[(y==0) & (y_pred.squeeze() >= high_loss_threshold)],
                y[(y==1) & (y_pred.squeeze() < 1 - high_loss_threshold)]
            ]).to(self.device, dtype=torch.float32)

            # Reassign y_pred to only include high loss values
            y_pred = torch.cat([
                negative_high_loss,
                positive_high_loss,
            ])

            # Get weight schedule
            if isinstance(negative_weight_schedule, float) or isinstance(negative_weight_schedule, int):
                negative_weight_coefficient = negative_weight_schedule
            elif len(negative_weight_schedule) <= step:
                logger.warning(f"Weights schedule is too short; expected at least {step+1} steps, got {len(negative_weight_schedule)} steps. Using last value.")
                negative_weight_coefficient = negative_weight_schedule[-1]
            else:
                negative_weight_coefficient = negative_weight_schedule[step]

            step_details["negative_weight"] = negative_weight_coefficient
            negative_weight_history.append(torch.tensor(negative_weight_coefficient))
            weight = torch.ones(y.shape[0]) * negative_weight_coefficient
            weight[y==1] = 1.0
            weight = weight.to(self.device).unsqueeze(1)

            y = y.unsqueeze(1)

            if y_pred.shape[0] != 0:
                # Accumulate the loss
                loss = self.loss(y_pred, y, weight) / accumulation_steps
                accumulated_samples += y_pred.shape[0]

                if y_pred.shape[0] >= 128:
                    accumulated_predictions = y_pred
                    accumulated_labels = y
                if accumulated_samples < 128:
                    accumulation_steps += 1
                    accumulated_predictions = torch.cat([accumulated_predictions, y_pred])
                    accumulated_labels = torch.cat([accumulated_labels, y])
                    if loss_history and recall_history and false_positive_rate_history:
                        loss_history.append(loss_history[-1])
                        recall_history.append(recall_history[-1])
                        false_positive_rate_history.append(false_positive_rate_history[-1])
                else:
                    # Perform the optimization step
                    loss.backward() # type: ignore[no-untyped-call]
                    self.optimizer.step()

                    accumulation_steps = 1
                    accumulated_samples = 0

                    num_false_labels = accumulated_labels[accumulated_labels==0].shape[0]
                    false_positives = self.num_false_positives(
                        accumulated_predictions,
                        accumulated_labels,
                        activation_threshold=activation_threshold
                    )
                    false_positive_rate = false_positives / max(num_false_labels, 1)

                    recall_rate = recall(accumulated_predictions, accumulated_labels)

                    loss_history.append(loss.detach().cpu())
                    recall_history.append(recall_rate.detach().cpu())
                    false_positive_rate_history.append(false_positive_rate.detach().cpu())

                    step_details["loss"] = loss_history[-1]
                    step_details["recall"] = recall_history[-1]
                    step_details["false_positive_rate"] = false_positive_rate_history[-1]

                    accumulated_predictions = torch.Tensor().to(self.device)
                    accumulated_labels = torch.Tensor().to(self.device)
            elif loss_history and recall_history and false_positive_rate_history:
                loss_history.append(loss_history[-1])
                recall_history.append(recall_history[-1])
                false_positive_rate_history.append(false_positive_rate_history[-1])
            else:
                loss_history.append(torch.tensor(last_loss))
                recall_history.append(torch.tensor(last_recall))
                false_positive_rate_history.append(torch.tensor(last_false_positive_rate))

            if step > 0 and step % validation_steps == 0:
                # Perform validations
                if validation is not None:
                    num_negative_labels = 0
                    accumulated_validation_predictions = []
                    accumulated_validation_labels = []

                    for validation_step, validation_datum in enumerate(validation):
                        with torch.no_grad():
                            x, y = validation_datum[0].to(self.device), validation_datum[1].to(self.device)
                            y_pred = self.model(x)[:, 0]
                            accumulated_validation_predictions.append(y_pred)
                            accumulated_validation_labels.append(y)
                            num_negative_labels += y[y==0].shape[0]

                    num_hours_validation = num_negative_labels * 1.44 / 3600 # 1.44 seconds per sample, 3600 seconds per hour

                    validation_false_positive_rate = self.num_false_positives(
                        torch.cat(accumulated_validation_predictions),
                        torch.cat(accumulated_validation_labels),
                        activation_threshold=activation_threshold
                    ) / num_hours_validation

                    validation_false_positive_per_hour.append(validation_false_positive_rate.detach().cpu())

                    validation_recall_rate = recall(
                        torch.cat(accumulated_validation_predictions),
                        torch.cat(accumulated_validation_labels)
                    )

                    validation_recall_history.append(validation_recall_rate.detach().cpu())

                    step_details["validation_false_positive_rate_per_hour"] = validation_false_positive_per_hour[-1]
                    step_details["validation_recall"] = validation_recall_history[-1]

                    if negative_weight_adjust_ratio is not None:
                        assert isinstance(negative_weight_schedule, float), "Negative weight schedule must be a float when using dynamic negative weight adjustment."
                        if validation_false_positive_rate > target_false_positive_rate:
                            negative_weight_schedule = negative_weight_schedule * negative_weight_adjust_ratio
                        else:
                            negative_weight_schedule = max(1.0, negative_weight_schedule / negative_weight_adjust_ratio)

                if testing is not None:
                    accumulated_testing_predictions = []
                    accumulated_testing_labels = []

                    for testing_step, testing_datum in enumerate(testing):
                        with torch.no_grad():
                            x, y = testing_datum[0].to(self.device), testing_datum[1].to(self.device)
                            y_pred = self.model(x)[:, 0]
                            accumulated_testing_predictions.append(y_pred)
                            accumulated_testing_labels.append(y)

                    test_predictions = torch.cat(accumulated_testing_predictions)
                    test_labels = torch.cat(accumulated_testing_labels)

                    test_false_positives = self.num_false_positives(
                        test_predictions,
                        test_labels,
                        activation_threshold=activation_threshold
                    )
                    test_false_positive_rate = test_false_positives / test_labels[test_labels==0].shape[0]

                    testing_false_positive_rate_history.append(test_false_positive_rate.detach().cpu())
                    testing_recall_history.append(recall(test_predictions, test_labels).detach().cpu())
                    testing_accuracy_history.append(accuracy(test_predictions, test_labels).detach().cpu())

                    step_details["testing_false_positive_rate"] = testing_false_positive_rate_history[-1]
                    step_details["testing_recall"] = testing_recall_history[-1]
                    step_details["testing_accuracy"] = testing_accuracy_history[-1]

            elif validation_false_positive_per_hour or testing_accuracy_history or testing_recall_history or testing_false_positive_rate_history:
                if validation is not None:
                    validation_false_positive_per_hour.append(validation_false_positive_per_hour[-1])
                    validation_recall_history.append(validation_recall_history[-1])

                if testing is not None:
                    testing_false_positive_rate_history.append(testing_false_positive_rate_history[-1])
                    testing_recall_history.append(testing_recall_history[-1])
                    testing_accuracy_history.append(testing_accuracy_history[-1])
            else:
                if validation is not None:
                    validation_false_positive_per_hour.append(torch.tensor(last_validation_false_positive_per_hour))
                    validation_recall_history.append(torch.tensor(last_validation_recall))

                if testing is not None:
                    testing_false_positive_rate_history.append(torch.tensor(last_testing_false_positive_rate))
                    testing_recall_history.append(torch.tensor(last_testing_recall))
                    testing_accuracy_history.append(torch.tensor(last_testing_accuracy))

            if use_wandb and (step == 0 or step % logging_steps == 0 or step % validation_steps == 0 or step == num_steps - 1):
                wandb.log(step_details)

            if step > 0 and step % checkpoint_steps == 0:
                self.save_checkpoint(f"{name}_{step}")

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        return (
            torch.stack(learning_rate_history),
            torch.stack(negative_weight_history),
            torch.stack(loss_history),
            torch.stack(high_loss_rate_history),
            torch.stack(recall_history),
            torch.stack(false_positive_rate_history),
            None if not validation_false_positive_per_hour else torch.stack(validation_false_positive_per_hour),
            None if not validation_recall_history else torch.stack(validation_recall_history),
            None if not testing_accuracy_history else torch.stack(testing_accuracy_history),
            None if not testing_recall_history else torch.stack(testing_recall_history),
            None if not testing_false_positive_rate_history else torch.stack(testing_false_positive_rate_history),
        )

    def log_metrics(
        self,
        learning_rate: torch.Tensor,
        negative_weight: torch.Tensor,
        loss: torch.Tensor,
        high_loss_rate: torch.Tensor,
        recall: torch.Tensor,
        false_positive_rate: torch.Tensor,
        validation_false_positive_rate_per_hour: Optional[torch.Tensor]=None,
        validation_recall: Optional[torch.Tensor]=None,
        testing_accuracy: Optional[torch.Tensor]=None,
        testing_recall: Optional[torch.Tensor]=None,
        testing_false_positive_rate: Optional[torch.Tensor]=None,
        description: str="Training",
        duration: float=0.0,
    ) -> None:
        """
        Log the metrics for the training.
        """
        if duration > 0.0:
            logger.info(f"{description} duration: {human_duration(duration)}")
        self.log_tensor_metrics(f"{description} Learning Rate", learning_rate)
        self.log_tensor_metrics(f"{description} Negative Weight", negative_weight)
        self.log_tensor_metrics(f"{description} Loss", loss)
        self.log_tensor_metrics(f"{description} High Loss Rate", high_loss_rate)
        self.log_tensor_metrics(f"{description} Recall", recall)
        self.log_tensor_metrics(f"{description} False Positive Rate", false_positive_rate)
        if validation_false_positive_rate_per_hour is not None:
            self.log_tensor_metrics(f"{description} Validation FP/HR", validation_false_positive_rate_per_hour)
        if validation_recall is not None:
            self.log_tensor_metrics(f"{description} Validation Recall", validation_recall)
        if testing_accuracy is not None:
            self.log_tensor_metrics(f"{description} Testing Accuracy", testing_accuracy)
        if testing_recall is not None:
            self.log_tensor_metrics(f"{description} Testing Recall", testing_recall)
        if testing_false_positive_rate is not None:
            self.log_tensor_metrics(f"{description} Testing FP Rate", testing_false_positive_rate)

    def graph_metrics(
        self,
        learning_rate: torch.Tensor,
        negative_weight: torch.Tensor,
        loss: torch.Tensor,
        high_loss_rate: torch.Tensor,
        recall: torch.Tensor,
        false_positive_rate: torch.Tensor,
        validation_false_positive_rate_per_hour: Optional[torch.Tensor]=None,
        validation_recall: Optional[torch.Tensor]=None,
        testing_accuracy: Optional[torch.Tensor]=None,
        testing_recall: Optional[torch.Tensor]=None,
        testing_false_positive_rate: Optional[torch.Tensor]=None,
        name: str="heybuddy",
        duration: float=0.0,
    ) -> None:
        """
        Graph the metrics for the training.
        """
        import matplotlib.pyplot as plt
        try:
            from qbstyles import mpl_style # type: ignore[import-untyped,import-not-found,unused-ignore]
            mpl_style(dark=True)
        except ImportError:
            logger.debug("Could not import qbstyles; using default style. To use qbstyles, install it with 'pip install qbstyles'.")
            pass

        num_axes = 6

        addon_axis = num_axes
        num_additional_axes = (
            (validation_false_positive_rate_per_hour is not None) +
            (validation_recall is not None) +
            (testing_accuracy is not None) +
            (testing_recall is not None) +
            (testing_false_positive_rate is not None)
        )

        fig, ax = plt.subplots(
            num_axes + num_additional_axes,
            1,
            figsize=(12, 16 + (num_additional_axes * 4)),
            constrained_layout=True
        )

        ax[0].plot(learning_rate, label="Learning Rate")
        ax[0].set_title("Learning Rate")
        ax[0].set_ylabel("Learning Rate")

        ax[1].plot(negative_weight, label="False Activation Penalty")
        ax[1].set_title("False Activation Penalty")
        ax[1].set_ylabel("False Activation Penalty")
        ax[1].set_yscale("log")

        ax[2].plot(loss, label="Loss")
        ax[2].set_title("Loss")
        ax[2].set_ylabel("Loss")

        ax[3].plot(high_loss_rate, label="High Loss Rate")
        ax[3].set_title("High Loss Rate")
        ax[3].set_ylabel("High Loss Rate")
        ax[3].set_ylim(0, 1)

        ax[4].plot(recall, label="Recall")
        ax[4].set_title("Recall")
        ax[4].set_ylabel("Recall")
        ax[4].set_ylim(0, 1)

        ax[5].plot(false_positive_rate, label="False Positive Rate")
        ax[5].set_title("False Positive Rate")
        ax[5].set_ylabel("False Positive Rate")

        if validation_false_positive_rate_per_hour is not None:
            ax[addon_axis].plot(validation_false_positive_rate_per_hour, label="Validation False Positive Rate per Hour")
            ax[addon_axis].set_title("Validation False Positive Rate per Hour")
            ax[addon_axis].set_ylabel("False Positive Rate per Hour")
            addon_axis += 1

        if validation_recall is not None:
            ax[addon_axis].plot(validation_recall, label="Validation Recall")
            ax[addon_axis].set_title("Validation Recall")
            ax[addon_axis].set_ylabel("Recall")
            ax[addon_axis].set_ylim(0, 1)
            addon_axis += 1

        if testing_accuracy is not None:
            ax[addon_axis].plot(testing_accuracy, label="Testing Accuracy")
            ax[addon_axis].set_title("Testing Accuracy")
            ax[addon_axis].set_ylabel("Accuracy")
            ax[addon_axis].set_ylim(0, 1)
            addon_axis += 1

        if testing_recall is not None:
            ax[addon_axis].plot(testing_recall, label="Testing Recall")
            ax[addon_axis].set_title("Testing Recall")
            ax[addon_axis].set_ylabel("Recall")
            ax[addon_axis].set_ylim(0, 1)
            addon_axis += 1

        if testing_false_positive_rate is not None:
            ax[addon_axis].plot(testing_false_positive_rate, label="Testing False Positive Rate")
            ax[addon_axis].set_title("Testing False Positive Rate")
            ax[addon_axis].set_ylabel("False Positive Rate")
            addon_axis += 1

        # Hide the main plot's axis and grid but keep the background
        fig.axes[0].set_axis_off()

        for a in ax.flat:
            a.patch.set_visible(True)  # Keep subplot backgrounds visible

        # Hide main figure axis and grid
        fig.patch.set_facecolor('white')  # Set the background color of the figure
        fig.patch.set_alpha(1.0)  # Ensure the background is not transparent
        plt.savefig(f"{name}_metrics.png")

    def __call__(
        self,
        training: DatasetType,
        validation: Optional[DatasetType]=None,
        testing: Optional[DatasetType]=None,
        num_steps: int=DEFAULT_STEPS,
        num_stages: int=DEFAULT_STAGES,
        max_negative_weight: float=DEFAULT_NEGATIVE_WEIGHT,
        logging_steps: int=DEFAULT_LOGGING_STEPS, # How often to log metrics
        validation_steps: int=DEFAULT_VALIDATION_STEPS, # How often to validate
        checkpoint_steps: int=DEFAULT_CHECKPOINT_STEPS, # How often to save a checkpoint
        target_false_positive_rate: float=DEFAULT_TARGET_FALSE_POSITIVE_RATE, # Target false positives per hour
        negative_weight_adjust_ratio: float=DEFAULT_NEGATIVE_WEIGHT_ADJUST_RATIO, # How much to adjust the negative weight between stages when false positive rate is too high
        dynamic_negative_weight: bool=DEFAULT_DYNAMIC_NEGATIVE_WEIGHT, # Whether to adjust the negative weight based on the false positive rate during training
        batch_size_adjust_ratio: float=DEFAULT_BATCH_SIZE_ADJUST_RATIO, # How much to adjust the batch size between stages
        learning_rate_adjust_ratio: float=DEFAULT_LEARNING_RATE_ADJUST_RATIO, # How much to adjust the learning rate between stages
        step_adjust_ratio: float=DEFAULT_STEP_ADJUST_RATIO, # How much to adjust the number of steps between stages
        learning_rate: float=DEFAULT_LEARNING_RATE, # Initial learning rate
        high_loss_threshold: float=DEFAULT_HIGH_LOSS_THRESHOLD, # Threshold for high loss rate
        activation_threshold: float=DEFAULT_ACTIVATION_THRESHOLD, # Threshold for activation
        wandb_entity: Optional[str]=None, # W&B entity
        name: str="heybuddy",
        **kwargs: Any,
    ) -> None:
        """
        Train the model in an automated way.

        Performs three separate training sequences at different rates.
        """
        start_time = perf_counter()
        learning_rate_history: List[torch.Tensor] = []
        negative_weight_history: List[torch.Tensor] = []
        loss_history: List[torch.Tensor] = []
        high_loss_rate_history: List[torch.Tensor] = []
        recall_history: List[torch.Tensor] = []
        false_positive_rate_history: List[torch.Tensor] = []

        validation_false_positive_rate_per_hour_history: List[Optional[torch.Tensor]] = []
        validation_recall_history: List[Optional[torch.Tensor]] = []

        testing_accuracy_history: List[Optional[torch.Tensor]] = []
        testing_recall_history: List[Optional[torch.Tensor]] = []
        testing_false_positive_rate_history: List[Optional[torch.Tensor]] = []

        # Make sure dataset batchers are started
        if isinstance(training, TrainingDatasetIterator):
            training.start()
        if isinstance(validation, TrainingDatasetIterator):
            validation.start()
        if isinstance(testing, TrainingDatasetIterator):
            testing.start()

        # Start W&B run
        if wandb_entity is not None:
            wandb.init(
                project=f"hey-buddy-{name}",
                entity=wandb_entity,
                config={
                    "num_steps": num_steps,
                    "num_stages": num_stages,
                    "num_params": sum(p.numel() for p in self.model.parameters()),
                    "max_negative_weight": max_negative_weight,
                    "logging_steps": logging_steps,
                    "validation_steps": validation_steps,
                    "checkpoint_steps": checkpoint_steps,
                    "target_false_positive_rate": target_false_positive_rate,
                    "negative_weight_adjust_ratio": negative_weight_adjust_ratio,
                    "dynamic_negative_weight": dynamic_negative_weight,
                    "batch_size_adjust_ratio": batch_size_adjust_ratio,
                    "learning_rate_adjust_ratio": learning_rate_adjust_ratio,
                    "step_adjust_ratio": step_adjust_ratio,
                    "learning_rate": learning_rate,
                    "high_loss_threshold": high_loss_threshold,
                    "activation_threshold": activation_threshold,
                    "layer_dim": self.layer_dim,
                    "num_layers": self.num_layers,
                    "num_heads": self.num_heads,
                    "architecture": self.architecture,
                    "training_dataset": training.metadata() if isinstance(training, TrainingDatasetIterator) else len(training),
                    "validation_dataset": validation.metadata() if isinstance(validation, TrainingDatasetIterator) else len(validation) if validation is not None else None,
                    "testing_dataset": testing.metadata() if isinstance(testing, TrainingDatasetIterator) else len(testing) if testing is not None else None,
                }
            )

        for i in range(num_stages):
            if dynamic_negative_weight:
                weights = max_negative_weight
                epoch_negative_weight_adjust_ratio = negative_weight_adjust_ratio
            else:
                weights = np.linspace(1, max_negative_weight, num_steps).tolist()
                epoch_negative_weight_adjust_ratio = None

            stage_start_time = perf_counter()
            lr, nw, loss, hlr, recall, fp, v_fp, v_recall, t_accuracy, t_recall, t_fp = self.train_epoch(
                training,
                validation=validation,
                testing=testing,
                num_steps=num_steps,
                negative_weight_schedule=weights,
                negative_weight_adjust_ratio=epoch_negative_weight_adjust_ratio,
                target_false_positive_rate=target_false_positive_rate,
                learning_rate=learning_rate,
                warmup_steps=num_steps//5,
                hold_steps=num_steps//3,
                logging_steps=logging_steps,
                validation_steps=validation_steps,
                checkpoint_steps=checkpoint_steps,
                description=f"Training Stage {i+1}",
                high_loss_threshold=high_loss_threshold,
                activation_threshold=activation_threshold,
                name=f"{name}_{i}",
                last_loss=0.0 if not loss_history else float(loss_history[-1][-1]),
                last_recall=0.0 if not recall_history else float(recall_history[-1][-1]),
                last_false_positive_rate=0.0 if not false_positive_rate_history else float(false_positive_rate_history[-1][-1]),
                last_validation_false_positive_per_hour=0.0 if not validation_false_positive_rate_per_hour_history else float(validation_false_positive_rate_per_hour_history[-1][-1]), # type: ignore[index]
                last_validation_recall=0.0 if not validation_recall_history else float(validation_recall_history[-1][-1]), # type: ignore[index]
                last_testing_accuracy=0.0 if not testing_accuracy_history else float(testing_accuracy_history[-1][-1]), # type: ignore[index]
                last_testing_recall=0.0 if not testing_recall_history else float(testing_recall_history[-1][-1]), # type: ignore[index]
                last_testing_false_positive_rate=0.0 if not testing_false_positive_rate_history else float(testing_false_positive_rate_history[-1][-1]), # type: ignore[index]
                use_wandb=wandb_entity is not None,
            )
            stage_duration = perf_counter() - stage_start_time

            # Log metrics
            self.log_metrics(
                learning_rate=lr,
                negative_weight=nw,
                loss=loss,
                high_loss_rate=hlr,
                recall=recall,
                false_positive_rate=fp,
                validation_false_positive_rate_per_hour=v_fp,
                validation_recall=v_recall,
                testing_accuracy=t_accuracy,
                testing_recall=t_recall,
                testing_false_positive_rate=t_fp,
                description=f"Training Stage {i+1}",
                duration=stage_duration,
            )

            # Add to history
            learning_rate_history.append(lr)
            negative_weight_history.append(nw)
            loss_history.append(loss)
            high_loss_rate_history.append(hlr)
            recall_history.append(recall)
            false_positive_rate_history.append(fp)
            validation_false_positive_rate_per_hour_history.append(v_fp)
            validation_recall_history.append(v_recall)
            testing_accuracy_history.append(t_accuracy)
            testing_recall_history.append(t_recall)
            testing_false_positive_rate_history.append(t_fp)

            # Update learning rate, steps and negative weight
            learning_rate *= learning_rate_adjust_ratio
            num_steps = max(validation_steps, int(num_steps * step_adjust_ratio))

            if v_fp is not None and dynamic_negative_weight:
                max_negative_weight = float(nw[-1])

            # Adjust the batch size
            if isinstance(training, WakeWordTrainingDatasetIterator):
                training.multiply_batch_size(batch_size_adjust_ratio)

        total_duration = perf_counter() - start_time
        learning_rate_overall = torch.cat(learning_rate_history).cpu()
        negative_weight_overall = torch.cat(negative_weight_history).cpu()
        loss_overall = torch.cat(loss_history).cpu()
        high_loss_rate_overall = torch.cat(high_loss_rate_history).cpu()
        recall_overall = torch.cat(recall_history).cpu()
        false_positive_rate_overall = torch.cat(false_positive_rate_history).cpu()

        if all([v_fp is None for v_fp in validation_false_positive_rate_per_hour_history]):
            validation_false_positive_rate_per_hour_overall = None
        else:
            validation_false_positive_rate_per_hour_overall = torch.cat([v_fp for v_fp in validation_false_positive_rate_per_hour_history if v_fp is not None]).cpu()

        if all([v_recall is None for v_recall in validation_recall_history]):
            validation_recall_overall = None
        else:
            validation_recall_overall = torch.cat([v_recall for v_recall in validation_recall_history if v_recall is not None]).cpu()

        if all([t_acc is None for t_acc in testing_accuracy_history]):
            testing_accuracy_overall = None
        else:
            testing_accuracy_overall = torch.cat([t_acc for t_acc in testing_accuracy_history if t_acc is not None]).cpu()

        if all([t_recall is None for t_recall in testing_recall_history]):
            testing_recall_overall = None
        else:
            testing_recall_overall = torch.cat([t_recall for t_recall in testing_recall_history if t_recall is not None]).cpu()

        if all([t_fp is None for t_fp in testing_false_positive_rate_history]):
            testing_false_positive_rate_overall = None
        else:
            testing_false_positive_rate_overall = torch.cat([t_fp for t_fp in testing_false_positive_rate_history if t_fp is not None]).cpu()

        self.log_datasets(
            training=training,
            validation=validation,
            testing=testing,
        )

        self.log_metrics(
            learning_rate=learning_rate_overall,
            negative_weight=negative_weight_overall,
            loss=loss_overall,
            high_loss_rate=high_loss_rate_overall,
            recall=recall_overall,
            false_positive_rate=false_positive_rate_overall,
            validation_false_positive_rate_per_hour=validation_false_positive_rate_per_hour_overall,
            validation_recall=validation_recall_overall,
            testing_accuracy=testing_accuracy_overall,
            testing_recall=testing_recall_overall,
            testing_false_positive_rate=testing_false_positive_rate_overall,
            description="Training Overall",
            duration=total_duration,
        )

        self.graph_metrics(
            learning_rate=learning_rate_overall,
            negative_weight=negative_weight_overall,
            loss=loss_overall,
            high_loss_rate=high_loss_rate_overall,
            recall=recall_overall,
            false_positive_rate=false_positive_rate_overall,
            validation_false_positive_rate_per_hour=validation_false_positive_rate_per_hour_overall,
            validation_recall=validation_recall_overall,
            testing_accuracy=testing_accuracy_overall,
            testing_recall=testing_recall_overall,
            testing_false_positive_rate=testing_false_positive_rate_overall,
            name=name,
            duration=total_duration,
        )

        self.save_checkpoint(f"{name}_final")

        # Make sure dataset batchers are stopped
        if isinstance(training, TrainingDatasetIterator):
            training.stop()
        if isinstance(validation, TrainingDatasetIterator):
            validation.stop()
        if isinstance(testing, TrainingDatasetIterator):
            testing.stop()
