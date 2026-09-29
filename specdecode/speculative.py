"""Speculative decoding driver with strict target-distribution preservation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from .models import DistributionProvider
from .probability import (
    accept_prob,
    bernoulli,
    categorical_sample,
    normalize,
    residual_distribution,
)


@dataclass
class Counters:
    """Effort counters (logical, hardware-agnostic)."""

    draft_queries: int = 0
    target_calls: int = 0
    target_positions_requested: int = 0
    target_positions_useful: int = 0
    accepted_draft_tokens: int = 0
    correction_tokens: int = 0
    bonus_tokens: int = 0
    rounds: int = 0

    @property
    def emitted(self) -> int:
        return self.accepted_draft_tokens + self.correction_tokens + self.bonus_tokens

    def as_dict(self) -> dict[str, int]:
        d = dict(self.__dict__)
        d["emitted"] = self.emitted
        return d


# Correct residual: proportional to max(0, q - p).
RESIDUAL_EXACT = "exact"
# Deliberately wrong: re-normalize q on rejection (forbidden by the task).
RESIDUAL_RENORM_TARGET = "renorm_target"
# Deliberately wrong: use q - p without clipping negatives (may go negative).
RESIDUAL_UNCLIPPED = "unclipped"
# Deliberately wrong: uniform over the rejection vocabulary.
RESIDUAL_UNIFORM = "uniform"


class SpeculativeDecoder:
    def __init__(
        self,
        target: DistributionProvider,
        draft: DistributionProvider,
        *,
        rng: Optional[random.Random] = None,
        residual_mode: str = RESIDUAL_EXACT,
        residual_fn=None,
    ):
        if target.vocab_size != draft.vocab_size:
            raise ValueError("vocabularies must match")
        self.target = target
        self.draft = draft
        self.rng = rng or random.Random()
        self.residual_mode = residual_mode
        self.residual_fn = residual_fn
        self.counters = Counters()

    def generate(
        self,
        prompt: tuple[int, ...],
        *,
        max_new_tokens: int,
        K: int,
    ) -> list[int]:
        if K < 0:
            raise ValueError("K must be >= 0")
        history = tuple(prompt)
        emitted: list[int] = []
        c = self.counters
        eos = self.target.eos_id

        while len(emitted) < max_new_tokens:
            remaining = max_new_tokens - len(emitted)
            draft_cap = min(K, remaining - 1) if eos is not None else min(K, remaining)
            # draft_cap < 0 means only a final EOS slot remains; skip drafts.
            draft_cap = max(0, draft_cap)

            draft_tokens: list[int] = []
            draft_dists = []
            draft_prefix = tuple(history)
            for _ in range(draft_cap):
                p = self.draft.prob(draft_prefix + tuple(draft_tokens))
                c.draft_queries += 1
                token = categorical_sample(p, self.rng)
                draft_tokens.append(token)
                draft_dists.append(p)
                if eos is not None and token == eos:
                    break

            extra_bonus = (
                eos is None
                or not draft_tokens
                or draft_tokens[-1] != eos
            )
            can_bonus = extra_bonus and len(emitted) + len(draft_tokens) < max_new_tokens
            n_target = len(draft_tokens) + (1 if can_bonus else 0)

            c.rounds += 1
            c.target_calls += 1
            c.target_positions_requested += n_target

            target_dists = [
                self.target.prob(history + tuple(draft_tokens[:i]))
                for i in range(n_target)
            ]

            rejected = False
            for i, token in enumerate(draft_tokens):
                p = draft_dists[i]
                q = target_dists[i]
                if bernoulli(accept_prob(p[token], q[token]), self.rng):
                    history = history + (token,)
                    emitted.append(token)
                    c.accepted_draft_tokens += 1
                    c.target_positions_useful += 1
                    if eos is not None and token == eos:
                        rejected = False
                        rejected_at = None
                        break
                else:
                    rejected = True
                    rejected_at = i
                    break

            if rejected:
                i = rejected_at
                q = target_dists[i]
                residual = (
                    self.residual_fn(
                        history + tuple(draft_tokens[:i]), draft_dists[i], q
                    )
                    if self.residual_fn is not None
                    else self._residual(draft_dists[i], q)
                )
                x = categorical_sample(residual, self.rng)
                history = history + (x,)
                emitted.append(x)
                c.correction_tokens += 1
                c.target_positions_useful += 1
                # positions i+1 .. n_target-1 evaluated but discarded.
                if eos is not None and x == eos:
                    break
            else:
                if can_bonus:
                    x = categorical_sample(target_dists[-1], self.rng)
                    history = history + (x,)
                    emitted.append(x)
                    c.bonus_tokens += 1
                    c.target_positions_useful += 1
                    if eos is not None and x == eos:
                        break
                elif draft_tokens and draft_tokens[-1] == eos:
                    break

        return emitted

    def _residual(self, p, q):
        mode = self.residual_mode
        zero = type(p[0])(0)
        if mode == RESIDUAL_EXACT:
            dist, z = residual_distribution(p, q)
            if z == 0:
                raise RuntimeError(
                    "rejection reached with zero rejection mass Z=0; "
                    "this event must have probability zero"
                )
            return dist
        if mode == RESIDUAL_RENORM_TARGET:
            return list(q)
        if mode == RESIDUAL_UNIFORM:
            v = type(p[0])(1) / type(p[0])(len(p))
            return [v] * len(p)
        if mode == RESIDUAL_UNCLIPPED:
            raw = [qi - pi for pi, qi in zip(p, q)]
            if all(r == 0 for r in raw):
                return list(q)
            shift = min(min(raw), zero)
            return normalize([r - shift for r in raw])
        raise ValueError(f"unknown residual mode {mode}")
