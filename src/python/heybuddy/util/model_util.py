from __future__ import annotations

from typing import Union, Tuple, Optional, TYPE_CHECKING
from threading import Event, Thread
from queue import Queue, Empty
from time import perf_counter

if TYPE_CHECKING:
    from heybuddy.util.typing_util import AudioType

__all__ = ["WakeWordModelThread"]

class WakeWordModelThread(Thread):
    """
    A thread that runs a wake word model.
    """
    POLLING_INTERVAL = 0.01
    input_queue: Queue[AudioType]
    output_queue: Queue[Tuple[Union[Tuple[bool, ...], Tuple[float, ...]], float]]

    def __init__(
        self,
        checkpoint: str,
        device_id: Optional[int]=None,
        threshold: float=0.5,
        return_scores: bool=False,
        embedding_spectrogram_batch_size: int=32,
        embedding_batch_size: int=32,
    ) -> None:
        """
        :param checkpoint: The path to the checkpoint file.
        :param device_id: The device ID to use for inference. If None, use CPU.
        :param threshold: The threshold for the prediction.
        :param return_scores: Whether to return the scores of the prediction.
        :param embedding_spectrogram_batch_size: The batch size for embedding spectrogram.
        :param embedding_batch_size: The batch size for embedding.
        """
        super().__init__()
        self.checkpoint = checkpoint
        self.device_id = device_id
        self.return_scores = return_scores
        self.threshold = threshold
        self.embedding_spectrogram_batch_size = embedding_spectrogram_batch_size
        self.embedding_batch_size = embedding_batch_size
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.stop_event = Event()

    def stop(self) -> None:
        """
        Stop the thread.
        """
        self.stop_event.set()

    @property
    def stopped(self) -> bool:
        """
        Whether the thread is stopped.
        """
        return self.stop_event.is_set()

    def run(self) -> None:
        """
        Run the thread.
        """
        import torch
        from heybuddy.wakeword import WakeWordMLPModel, WakeWordONNXModel
        if self.checkpoint.endswith(".onnx"):
            model = WakeWordONNXModel.from_file(self.checkpoint, device_id=self.device_id)
        else:
            if torch.cuda.is_available() and self.device_id is not None:
                device = torch.device(f"cuda:{self.device_id}")
            else:
                device = torch.device("cpu")

            model = WakeWordMLPModel.from_file(self.checkpoint, device=device) # type: ignore[assignment]
            model.eval() # type: ignore[attr-defined]

        while not self.stopped:
            try:
                audio = self.input_queue.get(timeout=self.POLLING_INTERVAL)
                start_time = perf_counter()
                prediction = model.predict( # type: ignore[attr-defined]
                    audio,
                    threshold=self.threshold,
                    embedding_spectrogram_batch_size=self.embedding_spectrogram_batch_size,
                    embedding_batch_size=self.embedding_batch_size,
                    return_scores=self.return_scores,
                )
                duration = perf_counter() - start_time
                self.output_queue.put((prediction, duration))
            except Empty:
                pass

    def put(self, audio: AudioType) -> None:
        """
        Put an audio to the input queue.

        :param audio: The audio to put.
        """
        self.input_queue.put(audio)
    
    def get(self, block: bool=True, timeout: Optional[float]=None) -> Tuple[Union[Tuple[bool, ...], Tuple[float, ...]], float]:
        """
        Get the prediction from the output queue.

        :param block: Whether to block until the prediction is available.
        :param timeout: The timeout in seconds.
        :return: The prediction.
        """
        return self.output_queue.get(block=block, timeout=timeout)
