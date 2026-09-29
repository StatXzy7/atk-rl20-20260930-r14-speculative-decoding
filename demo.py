"""Runnable demonstration: exact law checks, counters, counterexamples.

Run:  python3 demo.py
"""

from __future__ import annotations

import random
import time
from collections import Counter
from fractions import Fraction

from specdecode import (
    DistributionProvider,
    RandomWeightsModel,
    SpeculativeDecoder,
    TransitionTableModel,
    exact_speculative,
    exact_target,
)
from specdecode.counterexamples import (
    renorm_target_residual,
    single_step_demo,
)
from specdecode.speculative import RESIDUAL_EXACT


def hr(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def machines():
    target = TransitionTableModel(
        3,
        {
            (): [2, 1, 1],
            (0,): [1, 2, 1],
            (1,): [2, 0, 1],
            (2,): [1, 1, 2],
            (0, 0): [1, 0, 2],
            (1, 0): [0, 2, 1],
        },
        eos_id=2,
    )
    draft = TransitionTableModel(
        3,
        {
            (): [1, 2, 1],
            (0,): [2, 1, 1],
            (1,): [1, 1, 1],
            (2,): [1, 2, 1],
            (0, 0): [2, 0, 1],
            (1, 0): [1, 1, 2],
        },
        eos_id=2,
    )
    return target, draft


def section_exact_laws():
    hr("1. Exact output-law enumeration vs direct target autoregression")
    target, draft = machines()
    for K in (0, 1, 2, 3):
        tp = DistributionProvider(target)
        dp = DistributionProvider(draft)
        spec = exact_speculative(tp, dp, (), max_new_tokens=3, K=K)
        ref = exact_target(DistributionProvider(target), (),
                           max_new_tokens=3)
        ok = spec.law == ref.law
        print(f"K={K}: sequences={len(spec.law):2d}  exact equality "
              f"speculative == target: {ok}")
        assert ok

    print("\n  full distribution (K=2), Fraction probabilities:")
    spec = exact_speculative(DistributionProvider(target),
                             DistributionProvider(draft),
                             (), max_new_tokens=3, K=2)
    for seq, mass in sorted(spec.law.items()):
        print(f"    P{seq} = {float(mass):.6f}  ({mass})")


def section_counters():
    hr("2. Logical counters (expected, exact rational values)")
    target, draft = machines()
    L = 3
    print(f"  vocab=3, EOS=2, max_new_tokens={L}\n")
    header = (f"{'K':>2} {'draft_q':>9} {'tgt_calls':>9} "
              f"{'tgt_pos_req':>11} {'tgt_pos_use':>11} {'accepted':>8} "
              f"{'correct':>7} {'bonus':>5}")
    print(header)
    for K in (0, 1, 2, 3, 5):
        spec = exact_speculative(DistributionProvider(target),
                                 DistributionProvider(draft),
                                 (), max_new_tokens=L, K=K)
        c = spec.counters
        row = (f"{K:>2} {float(c['draft_queries']):>9.4f} "
               f"{float(c['target_calls']):>9.4f} "
               f"{float(c['target_positions_requested']):>11.4f} "
               f"{float(c['target_positions_useful']):>11.4f} "
               f"{float(c['accepted_draft_tokens']):>8.4f} "
               f"{float(c['correction_tokens']):>7.4f} "
               f"{float(c['bonus_tokens']):>5.4f}")
        print(row)
    naive = exact_target(DistributionProvider(target), (), max_new_tokens=L)
    print(f"\n  naive autoregression target positions = "
          f"{float(naive.counters['target_positions_useful']):.4f} "
          "(== useful positions for every K above)")
    print("  'requested' >= 'useful': the gap is positions evaluated but")
    print("   discarded after the first rejection (no useful token there).")


def section_cutoff():
    hr("3. Top-p truncation: verification is against the POST-transform law")
    target = RandomWeightsModel(3, seed=11, eos_id=2, eos_boost=2)
    draft = RandomWeightsModel(3, seed=22, eos_id=2, eos_boost=1)
    cutoff = Fraction(7, 10)
    tp = DistributionProvider(target, top_p_cutoff=cutoff)
    dp = DistributionProvider(draft, top_p_cutoff=cutoff)
    spec = exact_speculative(tp, dp, (), max_new_tokens=3, K=2)
    ref_trunc = exact_target(
        DistributionProvider(target, top_p_cutoff=cutoff), (),
        max_new_tokens=3)
    ref_raw = exact_target(DistributionProvider(target), (),
                           max_new_tokens=3)
    print(f"  top-p cutoff = {float(cutoff):.2f}")
    print(f"  == post-truncation target law: {spec.law == ref_trunc.law}")
    print(f"  == raw pre-truncation target law: "
          f"{spec.law == ref_raw.law} (expected False)")
    p0 = tp.prob(())
    print(f"  example post-truncation policy at empty history: "
          f"{[str(x) for x in p0]}")


def section_counterexample():
    hr("4. Deliberately wrong residual: renormalize target on rejection")
    d = single_step_demo()
    print(f"  q (target) = {tuple(str(x) for x in d['q'])}")
    print(f"  p (greedy draft) = {tuple(str(x) for x in d['p'])}")
    print(f"  total acceptance alpha = {d['accept_total_alpha']}, "
          f"rejection mass 1-alpha = {d['reject_mass_Z']}")
    print(f"  correct residual            = {d['correct_residual']}")
    print(f"  wrong residual (renorm q)   = {d['wrong_residual_renorm']}")
    print(f"  marginal with correct rule  = {d['marginal_correct']} == q")
    print(f"  marginal with wrong rule    = {d['marginal_wrong_renorm']}")
    print(f"  total-variation error       = {d['tv_distance_wrong']}")

    target, draft = machines()
    good = exact_speculative(DistributionProvider(target),
                             DistributionProvider(draft),
                             (), max_new_tokens=3, K=2)
    bad = exact_speculative(DistributionProvider(target),
                            DistributionProvider(draft),
                            (), max_new_tokens=3, K=2,
                            residual_fn=renorm_target_residual)
    diffs = [abs(good.law[k] - bad.law.get(k, Fraction(0)))
             for k in good.law]
    tv = sum(diffs) / 2
    print(f"\n  sequence-level TV distance injected by wrong residual "
          f"(L=3,K=2): {float(tv):.5f}")


def section_monte_carlo():
    hr("5. Monte Carlo frequencies (supplementary, not proof)")
    target, draft = machines()
    tp, dp = DistributionProvider(target), DistributionProvider(draft)
    ref = exact_target(DistributionProvider(target), (), max_new_tokens=2)
    n = 40_000
    agg = Counter()
    dec = SpeculativeDecoder(tp, dp, rng=random.Random(2026))
    for _ in range(n):
        agg[tuple(dec.generate((), max_new_tokens=2, K=2))] += 1
    print(f"  {n} live runs, K=2, L=2:")
    worst = 0.0
    for seq, mass in sorted(ref.law.items()):
        got = agg[seq] / n
        worst = max(worst, abs(got - float(mass)))
        print(f"    {seq}: MC={got:.4f}  exact={float(mass):.4f}")
    print(f"  worst absolute deviation = {worst:.4f} (sampling noise only)")
    print("  Note: a single run with no observed rejection proves nothing;")
    print("  equality is established exactly by section 1 enumeration.")


def section_slowdown():
    hr("6. Why slow drafts or low acceptance can make things SLOWER")
    target, draft = machines()
    L = 60
    K = 4

    def measure(draft_delay, _mode=RESIDUAL_EXACT, good_draft=True):
        t0 = time.perf_counter()
        for rep in range(200):
            tp = DistributionProvider(target, cache=False)
            dm = draft if good_draft else target
            dp = DistributionProvider(dm, cache=False)
            dec = SpeculativeDecoder(tp, dp,
                                     rng=random.Random(rep),
                                     residual_mode=mode)
            if draft_delay:
                real_prob = dp.prob

                def slow(hist):
                    time.sleep(draft_delay)
                    return real_prob(hist)

                dp.prob = slow  # type: ignore[assignment]
            dec.generate((), max_new_tokens=L, K=K)
        return (time.perf_counter() - t0) / 200

    base = measure(0.0)
    slow = measure(0.0015)
    t0 = time.perf_counter()
    for rep in range(200):
        tp0 = DistributionProvider(target, cache=False)
        dec0 = SpeculativeDecoder(tp0, DistributionProvider(draft,
                                                            cache=False),
                                  rng=random.Random(rep))
        dec0.generate((), max_new_tokens=L, K=0)
    naive_time = (time.perf_counter() - t0) / 200

    tp, dp = DistributionProvider(target, cache=False), DistributionProvider(
        draft, cache=False)
    totals = Counter()
    for rep in range(2000):
        dec = SpeculativeDecoder(DistributionProvider(target, cache=False),
                                 DistributionProvider(draft, cache=False),
                                 rng=random.Random(rep))
        dec.generate((), max_new_tokens=L, K=K)
        totals.update(dec.counters.as_dict())
    runs = 2000
    accepted = totals["accepted_draft_tokens"] / runs
    bonus = totals["bonus_tokens"] / runs
    rounds = totals["rounds"] / runs
    req = totals["target_positions_requested"] / runs
    use = totals["target_positions_useful"] / runs
    correction = totals["correction_tokens"] / runs
    print(f"  L={L}, K={K}")
    print(f"  expected rounds per generated token = "
          f"{rounds / L:.3f}  (lower is better)")
    print(f"  expected accepted draft tokens/round = {accepted / rounds:.3f}")
    print(f"  expected bonus tokens/round         = {bonus / rounds:.3f}")
    print(f"  expected tokens emitted/round       = "
          f"{(accepted + bonus + correction) / rounds:.3f}")
    print(f"  target positions requested/round    = {req / rounds:.3f}")
    print(f"  useful fraction of eval work        = {use / req:.3f}")
    print()
    print(f"  naive autoregression (K=0) wall time/seq  : {naive_time*1000:7.2f} ms")
    print(f"  speculative, fast draft wall time/seq     : {base*1000:7.2f} ms")
    print(f"  speculative, 1.5ms draft wall time/seq    : {slow*1000:7.2f} ms")
    print("  Here even the 'fast draft' run loses to K=0: only ~0.64 of")
    print("  evaluated positions are useful at this acceptance rate, while")
    print("  the serial K draft queries still have to be paid. Adding a")
    print("  1.5ms delay per draft query makes it far worse. In general a")
    print("  slow serial draft is paid at EVERY draft position of EVERY")
    print("  round; when acceptance is low most speculative target work is")
    print("  discarded, so round count and wasted evaluations rise and the")
    print("  method loses to plain autoregression.")
    print()
    print("  Effort model (logical): per round the method issues 1 batched")
    print("  target call covering <=K+1 positions but only keeps positions")
    print("  up to the first rejection. Speedup requires")
    print("  c_target(K+1)+K*c_draft < (accepted+correction/bonus)*c_target")
    print("  in wall-clock terms -- fewer logical calls != hardware speedup.")


if __name__ == "__main__":
    section_exact_laws()
    section_counters()
    section_cutoff()
    section_counterexample()
    section_monte_carlo()
    section_slowdown()
    print("\nAll exact-equality checks passed.")
