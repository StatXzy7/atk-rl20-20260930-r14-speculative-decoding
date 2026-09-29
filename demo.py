"""Exact verification, Monte Carlo supplement, and accounting example."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from pathlib import Path
import random
import statistics

from specdec.distributions import residual_distribution, transform_distribution
from specdec.exact import (
    direct_sequence_distribution,
    probability_with_wrong_residual,
    speculative_sequence_distribution,
)
from specdec.models import RandomFiniteModel, SimpleFiniteModel, TableModel
from specdec.sampling import SamplingSpec, direct_sample, speculative_sample

VOCAB = ("A", "B", "C", "<eos>")
EOS = "<eos>"


def show_exact_verification():
    target = RandomFiniteModel(VOCAB, EOS, seed=9001, zero_rate=0.25)
    draft = RandomFiniteModel(VOCAB, EOS, seed=5150, zero_rate=0.10)
    spec = SamplingSpec(top_p=Fraction(3, 4))
    direct = direct_sequence_distribution(target, (), spec, 3)
    speculative = speculative_sequence_distribution(
        target, draft, (), spec, 3, max_draft_tokens=3
    )
    print("Exact enumeration")
    print(f"  output sequences: {len(direct)}")
    print(f"  direct mass: {sum(direct.values())}")
    print(f"  speculative mass: {sum(speculative.values())}")
    print(f"  identical exact distributions: {direct == speculative}")
    print("  examples:")
    for sequence in sorted(direct, key=lambda item: (-direct[item], item))[:5]:
        print(f"    {sequence}: {float(direct[sequence]):.8f}")
    return direct == speculative


def show_wrong_residual():
    target = TableModel(
        VOCAB,
        EOS,
        {
            (): {"A": 4, "B": 1, "C": 5, EOS: 0},
            ("A",): {"A": 1, "B": 2, "C": 3, EOS: 4},
            ("B",): {"A": 5, "B": 0, "C": 2, EOS: 3},
            ("C",): {"A": 0, "B": 4, "C": 1, EOS: 5},
        },
    )
    draft = TableModel(
        VOCAB,
        EOS,
        {
            (): {"A": 2, "B": 3, "C": 5, EOS: 0},
            ("A",): {"A": 4, "B": 1, "C": 2, EOS: 3},
            ("B",): {"A": 1, "B": 3, "C": 4, EOS: 2},
            ("C",): {"A": 3, "B": 2, "C": 4, EOS: 1},
        },
    )
    p = transform_distribution(target.conditional(()), top_p=1)
    q = transform_distribution(draft.conditional(()), top_p=1)
    direct = direct_sequence_distribution(target, (), SamplingSpec(), 2)
    target_resample = probability_with_wrong_residual(
        target,
        draft,
        (),
        SamplingSpec(),
        2,
        2,
        lambda p, q: {token: value for token, value in p.items() if value > 0},
    )
    uniform = probability_with_wrong_residual(
        target,
        draft,
        (),
        SamplingSpec(),
        2,
        2,
        lambda p, q: {token: Fraction(1, len(p)) for token in p},
    )
    print("\nResidual counterexamples")
    print(f"  first-position residual: {residual_distribution(p, q)}")
    print(f"  resampling target matches: {target_resample == direct}")
    print(f"  uniform correction matches: {uniform == direct}")
    print("  examples of corrupted probabilities:")
    shown = 0
    for key in sorted(set(direct) | set(target_resample)):
        if direct.get(key) != target_resample.get(key):
            print(
                f"    {key}: target={direct.get(key, 0)} "
                f"wrong={target_resample.get(key, 0)}"
            )
            shown += 1
            if shown == 3:
                break


def monte_carlo(target, draft, spec, samples, k):
    exact = direct_sequence_distribution(target, (), spec, 4)
    direct_counts = Counter()
    speculative_counts = Counter()
    aggregate = None
    for index in range(samples):
        direct_counts[
            tuple(direct_sample(target, (), spec, 4, random.Random(10_000 + index)))
        ] += 1
        output, stats = speculative_sample(
            target,
            draft,
            (),
            spec,
            4,
            random.Random(20_000 + index),
            max_draft_tokens=k,
        )
        speculative_counts[tuple(output)] += 1
        if aggregate is None:
            aggregate = stats
        else:
            aggregate.rounds += stats.rounds
            aggregate.target_forwards += stats.target_forwards
            aggregate.target_positions += stats.target_positions
            aggregate.draft_forwards += stats.draft_forwards
            aggregate.draft_positions += stats.draft_positions
            aggregate.accepted_draft_tokens += stats.accepted_draft_tokens
            aggregate.bonus_tokens += stats.bonus_tokens
            aggregate.correction_tokens += stats.correction_tokens
            aggregate.rejection_events += stats.rejection_events
            aggregate.eos_stops += stats.eos_stops
            aggregate.wasted_target_positions += stats.wasted_target_positions
            aggregate.acceptance_lengths.extend(stats.acceptance_lengths)

    max_abs_frequency_gap = max(
        abs(direct_counts[key] / samples - exact.get(key, 0))
        for key in set(direct_counts) | set(exact)
    )
    return exact, direct_counts, speculative_counts, aggregate, max_abs_frequency_gap


def show_monte_carlo_and_counts():
    target = SimpleFiniteModel(VOCAB, EOS, seed=31415, zero_rate=0.20)
    draft = SimpleFiniteModel(VOCAB, EOS, seed=27182, zero_rate=0.20)
    spec = SamplingSpec()
    samples = 40_000
    exact, direct_counts, speculative_counts, stats, gap = monte_carlo(
        target, draft, spec, samples, k=3
    )
    print("\nMonte Carlo supplement and accounting")
    print(f"  samples: {samples}")
    print(f"  max |empirical direct frequency - exact target|: {gap:.6f}")
    empirical_gap = max(
        abs(speculative_counts[key] / samples - exact.get(key, 0))
        for key in set(speculative_counts) | set(exact)
    )
    print(f"  max |empirical speculative frequency - exact target|: {empirical_gap:.6f}")
    print(f"  logical target forwards: {stats.target_forwards}")
    print(f"  target positions evaluated: {stats.target_positions}")
    print(f"  draft forwards / positions: {stats.draft_forwards} / {stats.draft_positions}")
    print(f"  accepted draft tokens: {stats.accepted_draft_tokens}")
    print(f"  bonus tokens: {stats.bonus_tokens}")
    print(f"  correction tokens: {stats.correction_tokens}")
    print(f"  rejection events: {stats.rejection_events}")
    print(f"  wasted target positions after first rejection: {stats.wasted_target_positions}")
    print(
        "  mean accepted draft tokens/event: "
        f"{statistics.mean(stats.acceptance_lengths):.4f}"
    )
    print(
        "  target forwards per output token: "
        f"{stats.target_forwards / stats.generated_tokens:.4f}"
    )
    print(
        "  target positions per output token: "
        f"{stats.target_positions / stats.generated_tokens:.4f}"
    )

    slow_draft = SimpleFiniteModel(VOCAB, EOS, seed=16180, zero_rate=0.65)
    _, _, _, slow_stats, _ = monte_carlo(target, slow_draft, spec, 2_000, k=3)
    print("\nLow-acceptance comparison")
    print(f"  rejection rate/round: {slow_stats.rejection_events / slow_stats.rounds:.4f}")
    print(f"  target forwards/output: {slow_stats.target_forwards / slow_stats.generated_tokens:.4f}")
    print(f"  target positions/output: {slow_stats.target_positions / slow_stats.generated_tokens:.4f}")
    print(f"  draft positions/output: {slow_stats.draft_positions / slow_stats.generated_tokens:.4f}")


def main():
    exact_ok = show_exact_verification()
    show_wrong_residual()
    show_monte_carlo_and_counts()
    if not exact_ok:
        raise SystemExit("exact distribution mismatch")


if __name__ == "__main__":
    main()
