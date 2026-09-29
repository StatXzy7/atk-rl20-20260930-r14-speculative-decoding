"""Exact finite-sequence distribution enumeration for verification."""

from __future__ import annotations

from fractions import Fraction
from typing import Callable

from .distributions import acceptance_probability, residual_distribution
from .sampling import SamplingSpec, actual_distribution

Distribution = dict[object, Fraction]
PathProbability = dict[tuple, Fraction]


def direct_sequence_distribution(
    target, prefix, spec, max_new_tokens, stop_on_eos=True
) -> PathProbability:
    result: PathProbability = {}

    def walk(context, probability):
        emitted = context[len(prefix) :]
        if len(emitted) == max_new_tokens or (
            emitted and stop_on_eos and emitted[-1] == target.eos
        ):
            key = tuple(emitted)
            result[key] = result.get(key, 0) + probability
            return
        distribution = actual_distribution(target, context, spec)
        for token in target.vocab:
            token_probability = distribution.get(token, 0)
            if token_probability > 0:
                walk(context + [token], probability * token_probability)

    walk(list(prefix), Fraction(1))
    return result


def speculative_sequence_distribution(
    target, draft, prefix, spec, max_new_tokens, max_draft_tokens,
    stop_on_eos=True,
) -> PathProbability:
    return _enumerate(
        target, draft, prefix, spec, max_new_tokens, max_draft_tokens,
        residual_distribution, stop_on_eos,
    )


def probability_with_wrong_residual(
    target, draft, prefix, spec, max_new_tokens, max_draft_tokens,
    wrong_residual: Callable[[Distribution, Distribution], Distribution],
    stop_on_eos=True,
) -> PathProbability:
    return _enumerate(
        target, draft, prefix, spec, max_new_tokens, max_draft_tokens,
        wrong_residual, stop_on_eos,
    )


def _enumerate(target, draft, prefix, spec, max_new_tokens, max_draft_tokens,
               make_residual, stop_on_eos) -> PathProbability:
    if max_draft_tokens < 0 or max_new_tokens < 0:
        raise ValueError("length limits must be nonnegative")
    result: PathProbability = {}

    def finish(context, probability):
        emitted = context[len(prefix) :]
        if len(emitted) > max_new_tokens:
            raise AssertionError("enumeration exceeded generation budget")
        if len(emitted) == max_new_tokens or (
            emitted and stop_on_eos and emitted[-1] == target.eos
        ):
            key = tuple(emitted)
            result[key] = result.get(key, 0) + probability
        else:
            round_limit = min(max_draft_tokens, max_new_tokens - len(emitted))
            if round_limit == 0:
                distribution = actual_distribution(target, context, spec)
                for token in target.vocab:
                    token_probability = distribution.get(token, 0)
                    if token_probability > 0:
                        finish(context + [token], probability * token_probability)
            else:
                enumerate_round(context, [], Fraction(1), probability, round_limit)

    def enumerate_round(context, accepted, prefix_mass, round_probability, limit):
        emitted_count = len(context) - len(prefix) + len(accepted)
        if emitted_count == max_new_tokens:
            finish(context + accepted, round_probability * prefix_mass)
            return
        if accepted and stop_on_eos and accepted[-1] == target.eos:
            finish(context + accepted, round_probability * prefix_mass)
            return

        q = actual_distribution(draft, context + accepted, spec)
        p = actual_distribution(target, context + accepted, spec)
        correction = make_residual(p, q)

        for token in draft.vocab:
            qx = q.get(token, 0)
            if qx <= 0:
                continue
            alpha = acceptance_probability(p, q, token)
            reject_mass = round_probability * prefix_mass * qx * (1 - alpha)
            if reject_mass > 0:
                if not correction:
                    raise RuntimeError("positive rejection event has no residual mass")
                for y in target.vocab:
                    py = correction.get(y, 0)
                    if py > 0:
                        finish(
                            context + accepted + [y],
                            reject_mass * py,
                        )

            accept_mass = prefix_mass * qx * alpha
            if accept_mass <= 0:
                continue
            next_accepted = accepted + [token]
            if stop_on_eos and token == target.eos:
                finish(context + next_accepted, round_probability * accept_mass)
            elif len(next_accepted) == limit:
                emitted_after = len(context) - len(prefix) + len(next_accepted)
                if emitted_after < max_new_tokens:
                    bonus_p = actual_distribution(
                        target, context + next_accepted, spec
                    )
                    for bonus in target.vocab:
                        bonus_probability = bonus_p.get(bonus, 0)
                        if bonus_probability > 0:
                            finish(
                                context + next_accepted + [bonus],
                                round_probability * accept_mass * bonus_probability,
                            )
                else:
                    finish(
                        context + next_accepted,
                        round_probability * accept_mass,
                    )
            else:
                enumerate_round(
                    context,
                    next_accepted,
                    accept_mass,
                    round_probability,
                    limit,
                )

    finish(list(prefix), Fraction(1))
    return result
