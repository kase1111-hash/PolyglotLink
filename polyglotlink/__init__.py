"""
PolyglotLink - Semantic API Translator for IoT Device Ecosystems

A unified semantic layer for IoT device communication that automatically
translates heterogeneous device protocols and data formats into a
normalized, semantically-enriched message format.
"""

__version__ = "0.1.0"
__author__ = "PolyglotLink Team"
__license__ = "MIT"

from polyglotlink.models.schemas import (
    NormalizedMessage,
    Protocol,
    RawMessage,
)

__all__ = [
    "__version__",
    "NormalizedMessage",
    "Protocol",
    "RawMessage",
]
