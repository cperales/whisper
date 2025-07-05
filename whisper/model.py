import base64
import gzip
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import onnxruntime

from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function
from .transcribe import transcribe as transcribe_function

scaled_dot_product_attention = None
SDPA_AVAILABLE = False


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = np.exp(-log_timescale_increment * np.arange(channels // 2))
    scaled_time = np.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return np.cat([np.sin(scaled_time), np.cos(scaled_time)], dim=1)


@contextmanager
def disable_sdpa():
    prev_state = MultiHeadAttention.use_sdpa
    try:
        MultiHeadAttention.use_sdpa = False
        yield
    finally:
        MultiHeadAttention.use_sdpa = prev_state


class Whisper:
    """A thin wrapper around an onnxruntime.InferenceSession."""

    def __init__(self, session: onnxruntime.InferenceSession):
        self.session = session
        self.is_multilingual = True
        self.num_languages = 0

    def set_alignment_heads(self, _dump: bytes) -> None:  # no-op for ONNX models
        return

    def transcribe(self, *args, **kwargs):
        return transcribe_function(self, *args, **kwargs)

    def decode(self, *args, **kwargs):
        return decode_function(self, *args, **kwargs)

    def detect_language(self, *args, **kwargs):
        return detect_language_function(self, *args, **kwargs)
