"""Queryable conditional probability models over a finite vocabulary.

Every model exposes non-negative *weights*
``w(x | history tuple)``; probabilities are ``w / sum w``.  Integer weights
give exact ``Fraction`` probabilities, which the enumeration uses.
"""

from __future__ import annotations

import hashlib
import random
from fractions import Fraction
from typing import Optional, Protocol, Sequence, Union

from .probability import Distribution, apply_transforms, normalize

Number = Union[int, float, Fraction]


class ConditionalModel(Protocol):
    vocab_size: int
    eos_id: Optional[int]

    def weights(self, history: tuple[int, ...]) -> Sequence[Number]: ...


def _digest(history: tuple[int, ...], salt: str) -> int:
    h = hashlib.sha256()
    h.update(salt.encode())
    for token in history:
        h.update(b"t")
        h.update(token.to_bytes(4, "little", signed=False))
    return int.from_bytes(h.digest()[:8], "little")


class RandomWeightsModel:
    """Pseudo-random conditional weights, deterministic in the history."""

    def __init__(
        self,
        vocab_size: int,
        *,
        seed: int = 0,
        eos_id: Optional[int] = None,
        eos_boost: int = 1,
        salt: str = "",
        max_weight: int = 9,
        zeros: float = 0.0,
    ):
        self.vocab_size = vocab_size
        self.eos_id = eos_id
        self.seed = seed
        self.eos_boost = eos_boost
        self.salt = f"{seed}:{salt}"
        self.max_weight = max_weight
        self.zeros = zeros

    def weights(self, history: tuple[int, ...]) -> list[int]:
        stream = random.Random(_digest(history, self.salt))
        out = []
        for token in range(self.vocab_size):
            if self.zeros > 0 and stream.random() < self.zeros:
                out.append(0)
            else:
                out.append(1 + stream.randrange(self.max_weight))
        if self.eos_id is not None:
            out[self.eos_id] += self.eos_boost
        if sum(out) == 0:
            out[0] = 1
        return out


class TransitionTableModel:
    """Exact transition table keyed on suffixes (backoff to shorter suffix).

    ``rows`` maps a suffix tuple to a weight vector.  Suffixes are tried from
    longest to empty; the empty tuple is the required default row.  This
    allows hand-built small machines with known zero-probability edges.
    """

    def __init__(
        self,
        vocab_size: int,
        rows: dict[tuple[int, ...], Sequence[Number]],
        *,
        eos_id: Optional[int] = None,
        order: int = 2,
    ):
        if () not in rows:
            raise ValueError("rows must contain a default () row")
        for key, row in rows.items():
            if len(row) != vocab_size or sum(row) == 0:
                raise ValueError(f"bad row for suffix {key}")
        self.vocab_size = vocab_size
        self.rows = {k: list(v) for k, v in rows.items()}
        self.eos_id = eos_id
        self.order = order

    def weights(self, history: tuple[int, ...]) -> list[Number]:
        for length in range(min(self.order, len(history)), -1, -1):
            key = history[len(history) - length :] if length else ()
            if key in self.rows:
                return self.rows[key]
        raise AssertionError("default row missing")


class GreedyWrapper:
    """One-hot argmax draft.

    This is explicitly a *degenerate* proposal (all but one token have
    probability zero).  It is never presented as having full support;
    acceptance/residual math handles the zeros exactly.
    """

    def __init__(self, base: ConditionalModel):
        self.base = base
        self.vocab_size = base.vocab_size
        self.eos_id = base.eos_id
        self.greedy = True

    def weights(self, history: tuple[int, ...]) -> list[Number]:
        row = self.base.weights(history)
        best = max(range(len(row)), key=lambda i: (row[i], -i))
        return [1 if i == best else 0 for i in range(len(row))]


class DistributionProvider:
    """A queryable *transformed* policy: raw weights -> temperature -> top-p.

    ``prob(history)`` is the distribution that must actually be matched.
    When truncation is enabled this is the **post-transform** law; callers
    must verify equivalence against this, not against the raw model.
    Results are memoized per history because enumeration revisits prefixes.
    """

    def __init__(
        self,
        model: ConditionalModel,
        *,
        temperature: Number = 1,
        top_p_cutoff: Number = 1,
        fraction: bool = True,
        cache: bool = True,
    ):
        self.model = model
        self.vocab_size = model.vocab_size
        self.eos_id = model.eos_id
        self.temperature = temperature
        self.top_p_cutoff = top_p_cutoff
        self.fraction = fraction
        self.query_count = 0
        self._cache: dict[tuple[int, ...], Distribution] = {} if cache else {}
        self._caching = cache

    def raw_probs(self, history: tuple[int, ...]) -> list[Number]:
        weights = self.model.weights(history)
        if self.fraction:
            weights = [
                Fraction(int(w)) if not isinstance(w, Fraction) else w
                for w in weights
            ]
        return normalize(weights)

    def prob(self, history: tuple[int, ...]) -> Distribution:
        if self._caching and history in self._cache:
            return self._cache[history]
        self.query_count += 1
        weights = self.model.weights(history)
        if self.fraction:
            weights = [
                Fraction(int(w)) if not isinstance(w, Fraction) else w
                for w in weights
            ]
        out = apply_transforms(
            weights,
            temperature=self.temperature,
            top_p_cutoff=self.top_p_cutoff,
        )
        if self._caching:
            self._cache[history] = out
        return out
