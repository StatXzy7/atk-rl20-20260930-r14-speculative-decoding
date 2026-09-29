"""Finite-vocabulary probability distributions and sampling transforms."""

from __future__ import annotations

from fractions import Fraction
from math import fsum, isfinite
from typing import Any, Mapping


def _is_fractional_number(values: list[Any]) -> bool:
    return all(
        isinstance(value, (int, Fraction)) and not isinstance(value, bool)
        for value in values
    )


def normalize(raw: Mapping[Any, float]) -> dict[Any, float]:
    """Normalize nonnegative finite weights without dropping zero entries."""
    if not raw:
        raise ValueError("distribution must contain at least one token")
    values = list(raw.values())
    if any(value < 0 for value in values):
        raise ValueError("probability weights must be nonnegative")
    if any(
        not isinstance(value, (int, Fraction)) and not isfinite(float(value))
        for value in values
    ):
        raise ValueError("probability weights must be finite")

    exact = _is_fractional_number(values)
    total = sum(values) if exact else fsum(float(v) for v in values)
    if total <= 0:
        raise ValueError("distribution must have positive total mass")
    return {token: value / total for token, value in raw.items()}


def _renormalize(probs: Mapping[Any, float]) -> dict[Any, float]:
    values = list(probs.values())
    exact = _is_fractional_number(values)
    total = sum(values) if exact else fsum(float(v) for v in values)
    return {token: value / total for token, value in probs.items() if value > 0}


def _top_k(probs: Mapping[Any, float], k: int | None) -> dict[Any, float]:
    if k is None:
        return dict(probs)
    if k <= 0:
        raise ValueError("top_k must be positive")
    ordered = sorted(probs, key=lambda token: (-probs[token], token))
    return {token: probs[token] for token in ordered[:k]}


def _top_p(probs: Mapping[Any, float], p: float | Fraction | None) -> dict[Any, float]:
    if p is None:
        return dict(probs)
    if p <= 0 or p > 1:
        raise ValueError("top_p must be in (0, 1]")
    ordered = sorted(probs, key=lambda token: (-probs[token], token))
    kept: dict[Any, float] = {}
    cumulative = 0
    for token in ordered:
        if cumulative >= p and kept:
            break
        kept[token] = probs[token]
        cumulative += probs[token]
    return kept


def transform_distribution(
    raw: Mapping[Any, float],
    temperature: float | Fraction = 1,
    top_k: int | None = None,
    top_p: float | Fraction | None = 1,
) -> dict[Any, float]:
    """Return the actual post-temperature/top-k/top-p target or proposal.

    Temperature is p_t proportional to raw_t^(1/T). Top-k is applied first,
    then top-p, and retained mass is renormalized. Zero temperature is
    deterministic argmax. Ties use vocabulary order.
    """
    if temperature < 0:
        raise ValueError("temperature must be nonnegative")

    base = normalize(raw)
    if temperature == 0:
        winner = min(base, key=lambda token: (-base[token], token))
        unit = Fraction(1) if isinstance(base[winner], Fraction) else 1.0
        return {token: (unit if token == winner else 0 * unit) for token in base}

    if temperature == 1:
        tempered = base
    elif isinstance(temperature, Fraction):
        inverse_temperature = Fraction(1) / temperature
        if inverse_temperature.denominator == 1:
            exponent = inverse_temperature.numerator
            tempered = _renormalize(
                {token: probability ** exponent for token, probability in base.items()}
            )
        else:
            tempered = _renormalize(
                {
                    token: probability ** float(inverse_temperature)
                    for token, probability in base.items()
                }
            )
    else:
        tempered = _renormalize(
            {
                token: probability ** (1.0 / float(temperature))
                for token, probability in base.items()
            }
        )

    truncated = _top_p(_top_k(tempered, top_k), top_p)
    return _renormalize(truncated)


def residual_distribution(
    target: Mapping[Any, float],
    proposal: Mapping[Any, float],
) -> dict[Any, float]:
    """Return normalized positive part (p-q)_+; empty means zero reject mass."""
    tokens = set(target) | set(proposal)
    positive = {
        token: max(0, target.get(token, 0) - proposal.get(token, 0))
        for token in sorted(tokens)
    }
    positive = {token: value for token, value in positive.items() if value > 0}
    if not positive:
        return {}

    values = list(positive.values())
    exact = _is_fractional_number(values)
    mass = sum(values) if exact else fsum(float(v) for v in values)
    return {token: value / mass for token, value in positive.items()}


def acceptance_probability(
    target: Mapping[Any, float],
    proposal: Mapping[Any, float],
    token: Any,
) -> float:
    p = target.get(token, 0)
    q = proposal.get(token, 0)
    if p <= 0 or q <= 0:
        return 0
    return min(1, p / q)


def sample_categorical(distribution: Mapping[Any, float], rng) -> Any:
    if not distribution:
        raise ValueError("cannot sample from an empty distribution")
    nonzero = [
        (token, probability)
        for token, probability in distribution.items()
        if probability > 0
    ]
    if len(nonzero) == 1:
        return nonzero[0][0]

    point = rng.random()
    cumulative = 0
    for token, probability in sorted(nonzero, key=lambda item: item[0]):
        cumulative += probability
        if point < cumulative:
            return token
    return nonzero[-1][0]
