"""Speculative decoding prototype over finite queryable vocabularies."""

from .sampling import SamplingSpec, SpeculativeStats, direct_sample, speculative_sample

__all__ = [
    "SamplingSpec",
    "SpeculativeStats",
    "direct_sample",
    "speculative_sample",
]
