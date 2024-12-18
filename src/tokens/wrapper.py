from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizer


@dataclass
class SpecialTokens:
    """Standard interface for special tokens across different tokenizers"""

    pad_token: str = "[PAD]"
    cls_token: str = "[CLS]"
    sep_token: str = "[SEP]"
    mask_token: str = "[MASK]"
    bos_token: str = "<s>"
    eos_token: str = "</s>"


class TokenizerWrapper:
    """Wrapper for any HuggingFace tokenizer to provide consistent special token handling"""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Map actual tokenizer special tokens to standardized ones
        self.special_tokens = SpecialTokens(
            pad_token=tokenizer.pad_token or SpecialTokens.pad_token,
            cls_token=tokenizer.cls_token
            or tokenizer.bos_token
            or SpecialTokens.cls_token,
            sep_token=tokenizer.sep_token
            or tokenizer.eos_token
            or SpecialTokens.sep_token,
            mask_token=tokenizer.mask_token or SpecialTokens.mask_token,
            bos_token=tokenizer.bos_token
            or tokenizer.cls_token
            or SpecialTokens.bos_token,
            eos_token=tokenizer.eos_token
            or tokenizer.sep_token
            or SpecialTokens.eos_token,
        )

        # Cache special token IDs
        self.pad_token_id = tokenizer.convert_tokens_to_ids(
            self.special_tokens.pad_token
        )
        self.cls_token_id = tokenizer.convert_tokens_to_ids(
            self.special_tokens.cls_token
        )
        self.sep_token_id = tokenizer.convert_tokens_to_ids(
            self.special_tokens.sep_token
        )
        self.mask_token_id = tokenizer.convert_tokens_to_ids(
            self.special_tokens.mask_token
        )
        self.bos_token_id = tokenizer.convert_tokens_to_ids(
            self.special_tokens.bos_token
        )
        self.eos_token_id = tokenizer.convert_tokens_to_ids(
            self.special_tokens.eos_token
        )

        # Copy key attributes from underlying tokenizer
        self.vocab_size = len(tokenizer)

    def __len__(self) -> int:
        return len(self.tokenizer)

    def encode(
        self,
        text: str | list[str],
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
    ) -> list[int] | torch.Tensor:
        """Tokenize text with consistent special token handling"""
        return self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
        )

    def decode(
        self,
        token_ids: list[int] | torch.Tensor,
        skip_special_tokens: bool = False,
    ) -> str | list[str]:
        """Decode token IDs back to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def __call__(self, *args, **kwargs):
        """Pass through to underlying tokenizer with consistent special token handling"""
        return self.encode(*args, **kwargs)
