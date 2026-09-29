"""Exact enumeration of the speculative-decoding output law.

This module never samples.  It sums the *joint* probability of every draft
path, every accept/reject Bernoulli outcome and every residual/bonus draw,
recursing over rounds.  The resulting law over complete output sequences is
compared token-for-token with the direct target autoregressive law.  All
arithmetic is rational (``fractions.Fraction``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Optional

from .models import DistributionProvider
from .probability import Distribution, accept_prob, residual_distribution

ZERO = Fraction(0)
ONE = Fraction(1)


@dataclass
class ExactStats:
    law: dict[tuple[int, ...], Fraction]
    counters: dict[str, Fraction] = field(default_factory=dict)


def _zero_counters() -> dict[str, Fraction]:
    return {
        "rounds": ZERO,
        "draft_queries": ZERO,
        "target_calls": ZERO,
        "target_positions_requested": ZERO,
        "target_positions_useful": ZERO,
        "accepted_draft_tokens": ZERO,
        "correction_tokens": ZERO,
        "bonus_tokens": ZERO,
        "emitted": ZERO,
    }


def exact_residual(p: Distribution, q: Distribution):
    dist, z = residual_distribution(p, q)
    return dist, z


def exact_speculative(
    target: DistributionProvider,
    draft: DistributionProvider,
    prompt: tuple[int, ...],
    *,
    max_new_tokens: int,
    K: int,
    residual_fn: Optional[
        Callable[[tuple[int, ...], Distribution, Distribution], Distribution]
    ] = None,
) -> ExactStats:
    if K < 0:
        raise ValueError("K must be >= 0")
    eos = target.eos_id
    memo: dict[tuple[int, ...], ExactStats] = {}

    def solve(history: tuple[int, ...]) -> ExactStats:
        if history in memo:
            return memo[history]
        emitted_here = len(history) - len(prompt)
        remaining = max_new_tokens - emitted_here
        counters = _zero_counters()
        law: dict[tuple[int, ...], Fraction] = {}

        if eos is not None and len(history) > len(prompt) and history[-1] == eos:
            memo[history] = ExactStats({(): ONE}, counters)
            return memo[history]
        if remaining <= 0:
            memo[history] = ExactStats({(): ONE}, counters)
            return memo[history]

        cap = min(K, remaining - 1) if eos is not None else min(K, remaining)
        cap = max(0, cap)

        # Enumerate draft paths: (tokens, joint draft mass, draft distributions).
        paths: list[tuple[tuple[int, ...], Fraction, list[Distribution]]] = []

        def grow(prefix, mass, dists):
            if len(prefix) >= cap:
                paths.append((tuple(prefix), mass, list(dists)))
                return
            p = draft.prob(history + tuple(prefix))
            for token, pm in enumerate(p):
                if pm == 0:
                    continue
                nxt = prefix + [token]
                if eos is not None and token == eos:
                    paths.append((tuple(nxt), mass * pm, list(dists) + [p]))
                else:
                    grow(nxt, mass * pm, list(dists) + [p])

        grow([], ONE, [])
        path_total = sum((w for _, w, _ in paths), ZERO)
        assert path_total == 1, path_total

        def add_law(tail_prob: Fraction, head: tuple[int, ...], rest_stats):
            if rest_stats is None:
                law[head] = law.get(head, ZERO) + tail_prob
            else:
                for tail, tp in rest_stats.law.items():
                    seq = head + tail
                    law[seq] = law.get(seq, ZERO) + tail_prob * tp

        def add_counters(weight: Fraction, child: Optional[ExactStats], extra: dict):
            for key, value in extra.items():
                counters[key] += weight * value
            if child is not None:
                for key, value in child.counters.items():
                    counters[key] += weight * value

        for draft_tokens, w_d, draft_dists in paths:
            m = len(draft_tokens)
            bonus_allowed = (
                (eos is None or m == 0 or draft_tokens[-1] != eos)
                and emitted_here + m < max_new_tokens
            )
            n_target = m + (1 if bonus_allowed else 0)
            qs = [
                target.prob(history + tuple(draft_tokens[:j]))
                for j in range(n_target)
            ]

            # Round-level effort, paid once per realized draft path.
            round_extra = {
                "rounds": 1,
                "target_calls": 1,
                "target_positions_requested": n_target,
                "draft_queries": m,
            }

            accept_prefix_mass = ONE
            for i, token in enumerate(draft_tokens):
                p = draft_dists[i]
                q = qs[i]
                a = accept_prob(p[token], q[token])
                reject_mass = w_d * accept_prefix_mass * (ONE - a)
                if reject_mass > 0:
                    if residual_fn is not None:
                        residual = residual_fn(history + tuple(draft_tokens[:i]), p, q)
                    else:
                        residual, z = exact_residual(p, q)
                        assert z > 0
                    head_acc = draft_tokens[:i]
                    for x, rx in enumerate(residual):
                        if rx == 0:
                            continue
                        branch = reject_mass * rx
                        head = head_acc + (x,)
                        if eos is not None and x == eos:
                            add_law(branch, head, None)
                            add_counters(
                                branch,
                                None,
                                {
                                    **round_extra,
                                    "target_positions_useful": i + 1,
                                    "accepted_draft_tokens": i,
                                    "correction_tokens": 1,
                                    "emitted": i + 1,
                                },
                            )
                        else:
                            child = solve(history + head)
                            add_law(branch, head, child)
                            add_counters(
                                branch,
                                child,
                                {
                                    **round_extra,
                                    "target_positions_useful": i + 1,
                                    "target_calls": 0,
                                    "target_positions_requested": 0,
                                    "rounds": 0,
                                    "draft_queries": 0,
                                    "accepted_draft_tokens": i,
                                    "correction_tokens": 1,
                                    "emitted": i + 1,
                                },
                            )
                accept_prefix_mass *= a

            all_accept = w_d * accept_prefix_mass
            if all_accept > 0:
                if bonus_allowed:
                    q_bonus = qs[m]
                    for x, bx in enumerate(q_bonus):
                        if bx == 0:
                            continue
                        branch = all_accept * bx
                        head = draft_tokens + (x,)
                        if eos is not None and x == eos:
                            add_law(branch, head, None)
                            add_counters(
                                branch,
                                None,
                                {
                                    **round_extra,
                                    "target_positions_useful": n_target,
                                    "accepted_draft_tokens": m,
                                    "bonus_tokens": 1,
                                    "emitted": m + 1,
                                },
                            )
                        else:
                            child = solve(history + head)
                            add_law(branch, head, child)
                            add_counters(
                                branch,
                                child,
                                {
                                    **round_extra,
                                    "target_positions_useful": n_target,
                                    "accepted_draft_tokens": m,
                                    "bonus_tokens": 1,
                                    "emitted": m + 1,
                                },
                            )
                else:
                    add_law(all_accept, draft_tokens, None)
                    add_counters(
                        all_accept,
                        None,
                        {
                            **round_extra,
                            "target_positions_useful": m,
                            "accepted_draft_tokens": m,
                            "emitted": m,
                        },
                    )

        total = sum(law.values(), ZERO)
        assert total == 1, total
        result = ExactStats(law, counters)
        memo[history] = result
        return result

    return solve(tuple(prompt))


def exact_target(
    target: DistributionProvider,
    prompt: tuple[int, ...],
    *,
    max_new_tokens: int,
) -> ExactStats:
    """Independent reference: direct autoregressive target sampling law."""
    eos = target.eos_id
    memo: dict[tuple[int, ...], ExactStats] = {}

    def solve(history: tuple[int, ...]) -> ExactStats:
        if history in memo:
            return memo[history]
        emitted_here = len(history) - len(prompt)
        if (
            eos is not None
            and len(history) > len(prompt)
            and history[-1] == eos
        ) or emitted_here >= max_new_tokens:
            return ExactStats({(): ONE}, _zero_counters())

        q = target.prob(history)
        law: dict[tuple[int, ...], Fraction] = {}
        counters = _zero_counters()
        for x, qx in enumerate(q):
            if qx == 0:
                continue
            head = (x,)
            if eos is not None and x == eos:
                law[head] = qx
            else:
                child = solve(history + head)
                for tail, tp in child.law.items():
                    law[head + tail] = qx * tp
                for key, value in child.counters.items():
                    counters[key] += qx * value
            counters["rounds"] += qx
            counters["target_calls"] += qx
            counters["target_positions_requested"] += qx
            counters["target_positions_useful"] += qx
            counters["emitted"] += qx
        assert sum(law.values(), ZERO) == 1
        memo[history] = ExactStats(law, counters)
        return memo[history]

    return solve(tuple(prompt))
