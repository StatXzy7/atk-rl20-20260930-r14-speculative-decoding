"""Exact-distribution speculative decoding prototype."""

from .probability import (
    Distribution,
    accept_prob,
    apply_transforms,
    bernoulli,
    categorical_sample,
    normalize,
    residual_distribution,
    temperature_scale,
    top_p,
)
from .models import (
    ConditionalModel,
    DistributionProvider,
    GreedyWrapper,
    RandomWeightsModel,
    TransitionTableModel,
)
from .speculative import Counters, SpeculativeDecoder
from .enumeration import (
    ExactStats,
    exact_speculative,
    exact_target,
)

__all__ = [
    "Distribution",
    "ConditionalModel",
    "DistributionProvider",
    "GreedyWrapper",
    "RandomWeightsModel",
    "TransitionTableModel",
    "Counters",
    "SpeculativeDecoder",
    "ExactStats",
    "exact_speculative",
    "exact_target",
    "accept_prob",
    "bernoulli",
    "apply_transforms",
    "categorical_sample",
    "normalize",
    "residual_distribution",
    "temperature_scale",
    "top_p",
]
