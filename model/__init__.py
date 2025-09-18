"""Public API for the `model` package.

Exports generic building blocks from `model.modules` and the ART architecture
from `model.ART`. Keep this file minimal so import sites can do:

    from model import ArtifactRemovalTransformer, ExpandConv1x1

without reaching into submodules.
"""

# Expose core modules and models for easy import
from .modules import (
    PositionalEmbedding,
    ScaledDotProductAttention,
    MultiHeadAttention,
    FeedForward,
    ExpandConv1x1,
)

from .ART import (
    ArtifactRemovalTransformer
)

__all__ = [
    "PositionalEmbedding",
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "FeedForward",
    "ExpandConv1x1",
    "ArtifactRemovalTransformer",
]
