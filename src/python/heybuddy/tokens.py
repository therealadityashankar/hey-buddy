from __future__ import annotations

from typing import List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from tokenizers import Tokenizer # type: ignore[import-untyped]

__all__ = [
    "PretrainedTokenizer",
    "BERTTokenizer"
]

class PretrainedTokenizer:
    """
    A wrapper around the tokenizers library to load a pretrained tokenizer.

    Providers some utility functions for use with heybuddy.
    """
    pretrained_path: str
    tokenizer: Tokenizer

    def __init__(
        self,
        length: Optional[int] = None,
        pad_with: int = 0,
    ) -> None:
        """
        Load the pretrained tokenizer.
        """
        from tokenizers import Tokenizer
        self.length = length
        self.pad_with = pad_with
        self.tokenizer = Tokenizer.from_pretrained(self.pretrained_path)

    def decode(
        self,
        tensor_or_list: Union[torch.Tensor, List[int]]
    ) -> str:
        """
        Decode the input tensor and return the decoded text.
        """
        if self.length is not None:
            # Find last non-padding token
            if isinstance(tensor_or_list, list):
                last_token = next(i for i, x in enumerate(tensor_or_list[::-1]) if x != self.pad_with)
            else:
                last_token = (tensor_or_list != self.pad_with).nonzero(as_tuple=True)[0].max() + 1 # type: ignore[assignment]
            tensor_or_list = tensor_or_list[:last_token]
        return str(self.tokenizer.decode([int(i) for i in tensor_or_list]))

    def __call__(self, text: str) -> torch.Tensor:
        """
        Tokenize the input text and return the tokenized tensor.
        """
        import torch
        token_ids = torch.tensor(self.tokenizer.encode(text).ids[1:-1])
        if self.length is not None:
            token_ids = token_ids[:self.length]
            num_token_ids = token_ids.shape[0]
            if num_token_ids < self.length:
                token_ids = torch.cat([
                    token_ids,
                    torch.full((self.length - num_token_ids,), self.pad_with)
                ])
        return token_ids

class BERTTokenizer(PretrainedTokenizer):
    """
    A wrapper around the tokenizers library to load a BERT tokenizer.
    """
    pretrained_path = "bert-base-uncased"
