"""Small finite-vocabulary conditional models used as target and draft."""

from __future__ import annotations

import hashlib
import random
from fractions import Fraction
from typing import Mapping, Protocol, Sequence

from .distributions import normalize


class TokenModel(Protocol):
    vocab: tuple
    eos: object

    def conditional(self, prefix: Sequence) -> Mapping:
        """Return unnormalized or normalized logits/weights for this prefix."""


class TableModel:
    """Conditional table whose key is the full prefix after the fixed prefix."""

    def __init__(self, vocab: Sequence, eos: object, table: Mapping[tuple, Mapping]):
        if eos not in vocab:
            raise ValueError("EOS must be in vocabulary")
        self.vocab = tuple(vocab)
        self.eos = eos
        self.table = {tuple(key): normalize(value) for key, value in table.items()}

    def conditional(self, prefix: Sequence) -> Mapping:
        key = tuple(prefix)
        if key not in self.table:
            raise KeyError(f"missing conditional distribution for prefix {key!r}")
        return self.table[key]


class SimpleFiniteModel:
    """Small float-weight model with reproducible cached conditionals."""

    def __init__(self, vocab, eos, seed, zero_rate=0.18, scale=4.0):
        if eos not in vocab:
            raise ValueError("EOS must be in vocabulary")
        self.vocab = tuple(vocab)
        self.eos = eos
        self.seed = seed
        self.zero_rate = zero_rate
        self.scale = scale
        self.calls = 0
        self._cache = {}

    def conditional(self, prefix):
        self.calls += 1
        key = tuple(prefix)
        if key not in self._cache:
            seed_text = repr((self.seed, key, tuple(self.vocab), self.eos)).encode()
            stable_seed = int.from_bytes(
                hashlib.sha256(seed_text).digest()[:8], "little"
            )
            stream = random.Random(stable_seed)
            while True:
                weights = {}
                for token in self.vocab:
                    if token == self.eos:
                        value = stream.random() * 2.5
                    elif stream.random() < self.zero_rate:
                        value = 0.0
                    else:
                        value = 1.0 + stream.random() * self.scale
                    weights[token] = value
                if sum(weights.values()) > 0:
                    self._cache[key] = weights
                    break
        return dict(self._cache[key])


class RandomFiniteModel:
    """Reproducible random transition table expanded only for queried prefixes."""

    def __init__(
        self,
        vocab: Sequence,
        eos: object,
        seed: int = 0,
        max_non_eos_weight: int = 4,
        zero_rate: float = 0.18,
    ):
        if eos not in vocab:
            raise ValueError("EOS must be in vocabulary")
        self.vocab = tuple(vocab)
        self.eos = eos
        self.seed = seed
        self.max_non_eos_weight = max_non_eos_weight
        self.zero_rate = zero_rate
        self.calls = 0
        self._cache: dict[tuple, dict[object, Fraction]] = {}

    def conditional(self, prefix: Sequence) -> Mapping:
        self.calls += 1
        key = tuple(prefix)
        if key not in self._cache:
            seed_material = repr(
                (self.seed, key, tuple(self.vocab), self.eos)
            ).encode("utf-8")
            stable_seed = int.from_bytes(
                hashlib.sha256(seed_material).digest()[:8], "little"
            )
            stream = random.Random(stable_seed)
            weights = {}
            while True:
                for token in self.vocab:
                    if token == self.eos:
                        weight = stream.randrange(0, 3)
                    elif stream.random() < self.zero_rate:
                        weight = 0
                    else:
                        weight = stream.randrange(1, self.max_non_eos_weight + 1)
                    weights[token] = Fraction(weight)
                if sum(weights.values()) > 0:
                    break
            self._cache[key] = weights
        return dict(self._cache[key])
