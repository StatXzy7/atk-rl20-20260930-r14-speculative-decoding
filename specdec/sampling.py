"""Speculative sampling with exact residual correction."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Sequence

from .distributions import (
    acceptance_probability,
    residual_distribution,
    sample_categorical,
    transform_distribution,
)


@dataclass(frozen=True)
class SamplingSpec:
    temperature: float | Fraction = 1
    top_k: int | None = None
    top_p: float | Fraction | None = 1


@dataclass
class SpeculativeStats:
    rounds: int = 0
    target_forwards: int = 0
    target_positions: int = 0
    draft_forwards: int = 0
    draft_positions: int = 0
    accepted_draft_tokens: int = 0
    bonus_tokens: int = 0
    correction_tokens: int = 0
    rejection_events: int = 0
    eos_stops: int = 0
    wasted_target_positions: int = 0
    acceptance_lengths: list[int] = field(default_factory=list)

    @property
    def generated_tokens(self) -> int:
        return self.accepted_draft_tokens + self.bonus_tokens + self.correction_tokens


def actual_distribution(model, prefix: Sequence, spec: SamplingSpec) -> dict:
    return transform_distribution(
        model.conditional(prefix),
        temperature=spec.temperature,
        top_k=spec.top_k,
        top_p=spec.top_p,
    )


def direct_sample(
    target,
    prefix: Sequence,
    spec: SamplingSpec,
    max_new_tokens: int,
    rng,
    stop_on_eos: bool = True,
) -> list:
    """Sample directly from the actual transformed target distribution."""
    generated: list = []
    for _ in range(max_new_tokens):
        context = list(prefix) + generated
        distribution = actual_distribution(target, context, spec)
        token = sample_categorical(distribution, rng)
        generated.append(token)
        if stop_on_eos and token == target.eos:
            break
    return generated


def speculative_sample(
    target,
    draft,
    prefix: Sequence,
    spec: SamplingSpec,
    max_new_tokens: int,
    rng,
    max_draft_tokens: int,
    stop_on_eos: bool = True,
) -> tuple[list, SpeculativeStats]:
    """Run bounded speculative decoding from finite conditional models.

    One ``target_forwards`` is one batched call containing candidate prefixes.
    ``target_positions`` counts conditional distributions actually evaluated.
    """
    if max_draft_tokens < 0:
        raise ValueError("max_draft_tokens must be nonnegative")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    if target.eos != draft.eos:
        raise ValueError("target and draft must use the same EOS token")

    base = list(prefix)
    generated: list = []
    stats = SpeculativeStats()

    while len(generated) < max_new_tokens:
        stats.rounds += 1
        remaining = max_new_tokens - len(generated)
        draft_limit = min(max_draft_tokens, remaining)

        candidates: list = []
        proposals: list[dict] = []
        for _ in range(draft_limit):
            context = base + generated + candidates
            q = actual_distribution(draft, context, spec)
            stats.draft_forwards += 1
            stats.draft_positions += 1
            token = sample_categorical(q, rng)
            candidates.append(token)
            proposals.append(q)
            if stop_on_eos and token == draft.eos:
                break

        query_prefixes = [
            base + generated + candidates[:index]
            for index in range(len(candidates))
        ]
        draft_ended_at_eos = bool(
            candidates and stop_on_eos and candidates[-1] == target.eos
        )
        # A bonus follows all m candidates, so it needs m+1 <= remaining.
        evaluate_bonus = len(candidates) < remaining and not draft_ended_at_eos
        if evaluate_bonus:
            query_prefixes.append(base + generated + candidates)

        target_batch = [
            actual_distribution(target, query_prefix, spec)
            for query_prefix in query_prefixes
        ]
        stats.target_forwards += 1
        stats.target_positions += len(target_batch)

        accepted_this_round = 0
        rejection_index = None
        for index, token in enumerate(candidates):
            p = target_batch[index]
            q = proposals[index]
            if rng.random() < acceptance_probability(p, q, token):
                generated.append(token)
                accepted_this_round += 1
                stats.accepted_draft_tokens += 1
                if stop_on_eos and token == target.eos:
                    stats.eos_stops += 1
                    break
            else:
                rejection_index = index
                break

        if rejection_index is not None:
            stats.rejection_events += 1
            p = target_batch[rejection_index]
            q = proposals[rejection_index]
            residual = residual_distribution(p, q)
            if not residual:
                raise RuntimeError(
                    "rejection event has zero residual mass; this branch is empty"
                )
            correction = sample_categorical(residual, rng)
            generated.append(correction)
            stats.correction_tokens += 1
            stats.wasted_target_positions += (
                len(candidates) - rejection_index - 1 + int(evaluate_bonus)
            )
            stats.acceptance_lengths.append(accepted_this_round)
            if stop_on_eos and correction == target.eos:
                stats.eos_stops += 1
        else:
            eos_accepted = bool(
                generated
                and stop_on_eos
                and generated[-1] == target.eos
                and candidates
                and generated[-1] == candidates[-1]
            )
            if evaluate_bonus and not eos_accepted:
                bonus_distribution = target_batch[len(candidates)]
                token = sample_categorical(bonus_distribution, rng)
                generated.append(token)
                stats.bonus_tokens += 1
                if stop_on_eos and token == target.eos:
                    stats.eos_stops += 1
            stats.acceptance_lengths.append(accepted_this_round)

        if generated and stop_on_eos and generated[-1] == target.eos:
            break

    return generated, stats
