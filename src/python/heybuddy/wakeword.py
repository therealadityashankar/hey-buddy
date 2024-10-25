from __future__ import annotations

import torch
import torch.nn as nn

import numpy as np

from typing import Tuple, List, Union, Optional, Any, TYPE_CHECKING

from heybuddy.constants import *
from heybuddy.embeddings import SpeechEmbeddings, get_speech_embeddings
from heybuddy.util import (
    audio_to_bct_tensor,
    get_activation,
    ActivationFunctionLiteral,
    PretrainedONNXModel
)
from heybuddy.modules import (
    Module,
    TransformerBlock,
    ModulatingFinalLayer,
    MultiLayerPerceptron,
    GatedMultiLayerPerceptron,
)

if TYPE_CHECKING:
    from heybuddy.util.typing_util import AudioType, SingleAudioType

__all__ = [
    "WakeWordMLPModel",
    "WakeWordTransformerModel",
    "WakeWordONNXModel"
]

class WakeWordInferenceMixin:
    """
    A mixin class for inference with the model (for predicting on audio clips).
    """
    @property
    def speech_embeddings(self) -> SpeechEmbeddings:
        """
        Gets the pretrained speech embeddings model (for inference).
        """
        if not hasattr(self, "_speech_embeddings"):
            self._speech_embeddings = get_speech_embeddings(
                device_id=None if self.device.type == "cpu" else self.device.index or 0 # type: ignore[attr-defined]
            )
        return self._speech_embeddings

    def predict_timecodes(
        self,
        audio: SingleAudioType,
        threshold: float=0.5,
        embedding_spectrogram_batch_size: int=32,
        embedding_batch_size: int=32,
    ) -> List[float]:
        """
        Predicts per-second on an audio clip.
        """
        # Standardize to tensor
        audio_tensor, sample_rate = audio_to_bct_tensor(
            audio,
            sample_rate=16000,
        ) # 1, c, t

        # Ensure tensor is mono
        _, c, t = audio_tensor.shape
        if c == 1:
            audio_tensor = audio_tensor[0, 0, :]
        else:
            audio_tensor = audio_tensor[0, :, :].mean(dim=0)

        # Pad to the nearest 1s
        silence_frames = t % 16000
        if silence_frames > 0:
            audio_tensor = torch.cat([
                audio_tensor,
                torch.zeros(16000 - silence_frames, device=audio_tensor.device, dtype=audio_tensor.dtype)
            ])

        # Add 1s silence to beginning and end
        audio_tensor = torch.cat([
            torch.zeros(16000, device=audio_tensor.device, dtype=audio_tensor.dtype),
            audio_tensor,
            torch.zeros(16000, device=audio_tensor.device, dtype=audio_tensor.dtype),
        ])

        # Stack into 2s overlapping windows
        audio_tensor = torch.stack([
            audio_tensor[i:i+32000]
            for i in range(0, audio_tensor.shape[0] - 16000, 16000)
        ]).unsqueeze(1) # n, c, t

        # Get predictions
        predictions = self.predict(
            audio_tensor,
            threshold=threshold,
            embedding_spectrogram_batch_size=embedding_spectrogram_batch_size,
            embedding_batch_size=embedding_batch_size,
        )

        # Get approximate time of each prediction
        num_predictions = len(predictions)
        predicted_times = []
        for i, prediction in enumerate(predictions):
            if prediction:
                if i < num_predictions - 1 and predictions[i+1]:
                    predicted_times.append(i + 0.5)
                elif i == num_predictions - 1 and predictions[i-1]:
                    continue
                else:
                    predicted_times.append(i)

        return predicted_times

    def format_embeddings(self, embeddings: Any) -> Any:
        """
        Formats the embeddings for the model.
        """
        return embeddings

    def format_predictions(self, predictions: Any) -> Any:
        """
        Formats the predictions for the model.
        """
        return predictions

    @torch.no_grad()
    def predict(
        self,
        audio: AudioType,
        threshold: float=0.5,
        embedding_spectrogram_batch_size: int=32,
        embedding_batch_size: int=32,
        return_scores: bool=False,
        min_frames: int=23040,
    ) -> Union[Tuple[bool, ...], Tuple[float, ...]]:
        """
        Predicts on one or more audio clips.
        """
        audio_tensor, sample_rate = audio_to_bct_tensor(
            audio,
            sample_rate=16000,
        )
        n, c, t = audio_tensor.shape
        if t < min_frames: # Pad to 2s (by default)
            pad_frames = min_frames - t
            pad_left = int(pad_frames / 2)
            pad_right = pad_frames - pad_left

            audio_tensor = torch.cat([
                torch.zeros(n, c, pad_left, device=audio_tensor.device, dtype=audio_tensor.dtype),
                audio_tensor,
                torch.zeros(n, c, pad_right, device=audio_tensor.device, dtype=audio_tensor.dtype),
            ], dim=-1)

        embeddings = self.speech_embeddings(
            audio_tensor,
            embedding_batch_size=embedding_batch_size,
            spectrogram_batch_size=embedding_spectrogram_batch_size,
        )
        predictions = self.format_predictions(
            self(self.format_embeddings(embeddings)) # type: ignore[operator]
        )

        if return_scores:
            return tuple(predictions.flatten())
        else:
            return tuple(predictions > threshold)

class WakeWordMLPModel(WakeWordInferenceMixin, Module):
    """
    The NN for the model.
    
    input > flatten > linear > layernorm > relu > linear > layernorm > relu > linear > sigmoid > output

    Input shape is (batch_size, 16, 96), where:
        96 is the embedding size of the audio features (frozen speech embeddings extracted from a pre-trained model).
        16 is the number of embeddings in the sequence.
    """
    def __init__(
        self,
        input_shape: Tuple[int, int]=(16, 96),
        layer_dim: int=DEFAULT_LAYER_DIM,
        num_layers: int=DEFAULT_LAYERS,
        use_gating: bool=DEFAULT_USE_GATING,
        use_half_layers: bool=DEFAULT_USE_HALF_LAYERS,
        dropout: float=0.1,
        activation: Optional[ActivationFunctionLiteral]="silu",
    ) -> None:
        super(WakeWordMLPModel, self).__init__()
        self.input_shape = input_shape
        self.input_features = input_shape[0] * input_shape[1]
        self.use_gating = use_gating
        self.use_half_layers = use_half_layers

        self.dropout = nn.Dropout(dropout)
        self.flatten = nn.Flatten()

        mlp_class = GatedMultiLayerPerceptron if use_gating else MultiLayerPerceptron

        # Fully-connected layer
        self.norm_in = nn.LayerNorm(self.input_features)
        self.mlp_in = mlp_class(
            input_dim=self.input_features,
            hidden_dim=layer_dim,
            output_dim=layer_dim,
            activation=activation,
        )

        # Half-connected layers
        self.half_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.input_features // 2),
                mlp_class(
                    input_dim=self.input_features // 2,
                    hidden_dim=layer_dim,
                    output_dim=layer_dim,
                    activation=activation,
                )
            )
            for i in range(len(self.half_indices))
        ])

        # MLP
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(layer_dim),
                mlp_class(
                    input_dim=layer_dim,
                    hidden_dim=layer_dim,
                    output_dim=layer_dim,
                    activation=activation,
                )
            )
            for i in range(num_layers)
        ])

        # Out
        self.norm_out = nn.LayerNorm(layer_dim)
        self.mlp_out = mlp_class(
            input_dim=layer_dim,
            hidden_dim=layer_dim,
            output_dim=1,
            activation=activation,
        )
        self.sigmoid = nn.Sigmoid()

    @classmethod
    def from_file(
        cls,
        path: str,
        device: Optional[torch.device]=None,
    ) -> WakeWordMLPModel:
        """
        Loads the model from a file.
        """
        state_dict = torch.load(path, weights_only=True, map_location="cpu")
        layer_dim = state_dict["norm_out.weight"].shape[0]

        # Find the number of layers in this state dict
        num_layers = 0
        while True:
            if f"layers.{num_layers}.0.weight" in state_dict:
                num_layers += 1
            else:
                break

        model = cls(
            layer_dim=layer_dim,
            num_layers=num_layers,
        )
        model.load_state_dict(state_dict)
        if device is not None:
            model.to(device)
        return model

    @property
    def half_indices(self) -> List[List[int]]:
        """
        The attention indices for the half-connected layers.
        """
        if not self.use_half_layers:
            return []
        return [
            [0,1,2,3,4,5,6,7],       # 1111111100000000
            [8,9,10,11,12,13,14,15], # 0000000011111111
            [0,1,2,3,8,9,10,11],     # 1111000011110000
            [4,5,6,7,12,13,14,15],   # 0000111100001111
            [4,5,6,7,8,9,10,11],     # 0000111111110000
            [0,1,2,3,12,13,14,15],   # 1111000000001111
            [0,1,4,5,8,9,12,13],     # 1100110011001100
            [2,3,6,7,10,11,14,15],   # 0011001100110011
            [0,1,6,7,8,9,14,15],     # 1100001111000011
            [2,3,4,5,10,11,12,13],   # 0011110000111100
            [0,2,4,6,8,10,12,14],    # 1010101010101010
            [1,3,5,7,9,11,13,15],    # 0101010101010101
            [0,3,4,7,8,11,12,15],    # 1001100110011001
            [1,2,5,6,9,10,13,14],    # 0110011001100110
            [0,5,2,7,8,13,10,15],    # 1011010110110101
            [1,4,3,6,9,12,11,14],    # 0100101001010010
        ]

    def format_embeddings(self, embeddings: Any) -> Any:
        """
        Formats the embeddings for the model.
        """
        return torch.tensor(embeddings, device=self.device, dtype=torch.float32)

    def format_predictions(self, predictions: Any) -> Any:
        """
        Formats the predictions for the model.
        """
        return predictions.cpu().numpy()

    def save_onnx(
        self,
        path: str,
        opset_version: int=19
    ) -> None:
        """
        Saves the model to an ONNX file.
        """
        self.to("cpu")
        torch.onnx.export(
            self,
            torch.randn(self.input_shape).unsqueeze(0),
            path,
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        """
        x = self.dropout(x)
        states = self.mlp_in(self.norm_in(self.flatten(x)))

        for half_indices, half_layer in zip(self.half_indices, self.half_layers):
            states = states + half_layer(self.flatten(x[:, half_indices, :]))

        for layer in self.layers:
            states = layer(states)

        states = self.sigmoid(self.mlp_out(self.norm_out(states)))
        return states # type: ignore[no-any-return]

class WakeWordTransformerModel(WakeWordInferenceMixin, Module):
    """
    The advanced NN for the model.
    """
    def __init__(
        self,
        input_shape: Tuple[int, int]=(16, 96),
        dim: int=DEFAULT_LAYER_DIM,
        num_layers: int=DEFAULT_LAYERS,
        num_heads: int=DEFAULT_HEADS,
        multiple_of: int=18,
        norm_epsilon: float=1e-5,
        dropout: float=0.1,
        activation: Optional[ActivationFunctionLiteral]="silu",
    ) -> None:
        """
        :param input_shape: The shape of the input.
        :param dim: The dimension of the model.
        :param num_layers: The number of layers.
        :param num_heads: The number of heads.
        :param multiple_of: The multiple of the model.
        :param norm_epsilon: The epsilon for the normalization.
        :param dropout: The dropout for the model.
        """
        super(WakeWordTransformerModel, self).__init__()
        self.input_shape = input_shape
        self.input_features = input_shape[0] * input_shape[1]
        self.input_frames = input_shape[0]
        self.input_dim = input_shape[1]
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim = dim

        self.dropout = nn.Dropout(dropout)

        # Input layer
        self.linear_in = nn.Linear(self.input_dim, dim)
        self.activation = get_activation(activation)
        self.layernorm = nn.LayerNorm(dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                multiple_of=multiple_of,
                norm_epsilon=norm_epsilon,
                layer_id=i
            )
            for i in range(num_layers)
        ])

        # Output layer
        self.final_layer = ModulatingFinalLayer(
            hidden_size=16,
            output_size=1
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        """
        x = self.dropout(x)

        # Input layer
        x = self.activation(self.layernorm(self.linear_in(x)))

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Output layer
        x = x.transpose(1, 2)
        x = self.final_layer(x)[:, :, 0]
        x = self.sigmoid(x)
        x = torch.amax(x, dim=(1), keepdim=True)
        return x

    @classmethod
    def from_file(
        cls,
        path: str,
        device: Optional[torch.device]=None,
    ) -> WakeWordTransformerModel:
        """
        Loads the model from a file.
        """
        state_dict = torch.load(path, weights_only=True, map_location="cpu")
        model = cls()
        model.load_state_dict(state_dict, strict=False)
        if device is not None:
            model.to(device)
        return model

    def save_onnx(
        self,
        path: str,
        opset_version: int=19
    ) -> None:
        """
        Saves the model to an ONNX file.
        """
        self.to("cpu")
        torch.onnx.export(
            self,
            torch.randn(self.input_shape).unsqueeze(0),
            path,
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
        )

class WakeWordONNXModel(WakeWordInferenceMixin, PretrainedONNXModel):
    """
    The ONNX model for the model.
    """
    def __call__(self, input: np.ndarray[Any, Any], retry: bool=True) -> np.ndarray[Any, Any]:
        """
        Forward pass of the model.
        """
        return super(WakeWordONNXModel, self).__call__( # type: ignore[no-any-return]
            self,
            input=input,
            retry=retry
        )[0]
