"""Probability primitives.

Distributions are indexable sequences ``p`` with ``sum(p) == 1`` and
``p[i] >= 0``.  Everything works with ``fractions.Fraction`` (exact rational
arithmetic, used by the enumeration proof) or ``float`` (used by the Monte
Carlo driver).
"""

from __future__ import annotations

import math
import random
from fractions import Fraction
from typing import Sequence, Union

Number = Union[int, float, Fraction]
Distribution = Sequence[Number]


def normalize(weights: Sequence[Number]) -> list[Number]:
    total = sum(weights, type(weights[0])(0)) if weights else None
    if total is None or total == 0:
        raise ValueError("cannot normalize a zero-mass vector")
    return [w / total for w in weights]


def temperature_scale(
    weights: Sequence[Number], temperature: Number = 1
) -> list[Number]:
    r"""Return weights**(1/T) *before* renormalization (the logits view).

    With exact ``Fraction`` weights this is only possible when 1/T is a
    non-negative integer (e.g. T=1, 1/2, 1/3); irrational powers are not
    rationals and cannot be used in the exact enumeration.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if isinstance(weights[0], Fraction):
        t = temperature if isinstance(temperature, Fraction) else Fraction(temperature)
        # Exact rational exponent requires 1/T to be an integer (T = 1/n).
        inv = t.denominator if t.numerator == 1 else None
        if inv is None:
            raise ValueError(
                "exact Fraction temperature needs T = 1/n for integer n"
            )
        return [w ** inv for w in weights]
    return [w ** (1.0 / float(temperature)) for w in weights]


def top_p(
    weights: Sequence[Number],
    cutoff: Number = 1,
    *,
    tie_order: str = "index",
) -> list[Number]:
    """Nucleus truncation of *unnormalized* weights.

    Keep the smallest set of descending-probability tokens whose cumulative
    mass reaches ``cutoff`` (the token crossing the boundary is included).
    The survivor vector keeps the original entries and zeros out the rest;
    call :func:`normalize` afterwards.  Ties are broken by token index, which
    removes ambiguity for the exact tests.
    """
    total = sum(weights, type(weights[0])(0))
    if total == 0:
        raise ValueError("cannot truncate a zero-mass vector")
    if not 0 < cutoff <= 1:
        raise ValueError("top-p cutoff must lie in (0, 1]")
    probs = normalize(weights)
    order = sorted(range(len(probs)), key=lambda i: (-probs[i], i))
    kept: set[int] = set()
    cum = type(cutoff)(0)
    for idx in order:
        kept.add(idx)
        cum += probs[idx]
        if cum >= cutoff:
            break
    result = [w if i in kept else type(w)(0) for i, w in enumerate(weights)]
    assert tie_order == "index"
    return result


def apply_transforms(
    weights: Sequence[Number],
    *,
    temperature: Number = 1,
    top_p_cutoff: Number = 1,
) -> list[Number]:
    """Temperature then nucleus truncation, returning a normalized policy."""
    scaled = temperature_scale(weights, temperature)
    if top_p_cutoff < 1:
        scaled = top_p(scaled, top_p_cutoff)
    return normalize(scaled)


def accept_prob(p: Number, q: Number) -> Number:
    r"""Speculative acceptance ratio ``min(1, q/p)`` with zero support rules.

    ``p`` is the *draft* proposal mass, ``q`` the target mass at the same
    token.  A token the draft never proposes (``p == 0``) has no event to
    accept; a token with ``q == 0`` is always rejected (ratio 0).
    """
    if p < 0 or q < 0:
        raise ValueError("probabilities must be non-negative")
    if p == 0:
        return type(p)(0)
    ratio = q / p
    one = type(p)(1)
    return one if ratio >= 1 else ratio


def residual_distribution(
    p: Distribution, q: Distribution
) -> tuple[list[Number], Number]:
    r"""Return ``(residual, Z)`` with ``residual[i] = max(0, q_i-p_i)/Z``.

    The residual is the normalized *rejection* mass::

        Z = sum_i (q_i - min(q_i, p_i)) = sum_i max(0, q_i - p_i).

    ``Z == 0`` means target is dominated by the draft everywhere, so a
    rejection event has probability zero and there is no residual draw.
    """
    if len(p) != len(q):
        raise ValueError("draft and target must share the vocabulary")
    num_type = type(p[0])
    raw = [max(num_type(0), qi - pi) for pi, qi in zip(p, q)]
    z = sum(raw, num_type(0))
    if z == 0:
        return [num_type(0) for _ in p], num_type(0)
    return [ri / z for ri in raw], z


def categorical_sample(
    probs: Sequence[Number], rng: random.Random
) -> int:
    """Categorical draw.

    Float distributions use a uniform float.  ``Fraction`` distributions use
    exact rejection sampling on the common integer denominator, so the draw is
    genuinely unbiased (needed for the exact equivalence verification).
    """
    total = sum(probs, type(probs[0])(0))
    if isinstance(probs[0], Fraction):
        valid = total == 1
    else:
        valid = abs(float(total) - 1.0) <= 1e-12
    if not valid:
        raise ValueError(f"distribution does not sum to 1: {total}")
    if isinstance(probs[0], Fraction):
        denom = 1
        for mass in probs:
            denom = denom * mass.denominator // math.gcd(denom, mass.denominator)
        nbits = (denom - 1).bit_length()
        while True:
            m = rng.getrandbits(nbits)
            if m < denom:
                break
        cum_int = 0
        for token, mass in enumerate(probs):
            cum_int += mass.numerator * (denom // mass.denominator)
            if m < cum_int:
                return token
        return len(probs) - 1
    u = rng.random()
    cum: Number = 0
    for token, mass in enumerate(probs):
        cum += mass
        if u < cum:
            return token
    return len(probs) - 1


def bernoulli(mass: Number, rng: random.Random) -> bool:
    """Bernoulli draw; exact for ``Fraction`` mass, float otherwise."""
    if mass <= 0:
        return False
    if mass >= 1:
        return True
    if isinstance(mass, Fraction):
        denom = mass.denominator
        nbits = (denom - 1).bit_length()
        while True:
            m = rng.getrandbits(nbits)
            if m < denom:
                return m < mass.numerator
    return rng.random() < float(mass)
