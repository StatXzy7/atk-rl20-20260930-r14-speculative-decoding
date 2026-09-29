from __future__ import annotations

import random
import unittest
from fractions import Fraction

from specdecode import (
    DistributionProvider,
    GreedyWrapper,
    RandomWeightsModel,
    SpeculativeDecoder,
    TransitionTableModel,
    exact_speculative,
    exact_target,
)
from specdecode.counterexamples import (
    renorm_target_residual,
    single_step_demo,
    uniform_residual,
    unclipped_residual,
)
from specdecode.probability import (
    accept_prob,
    categorical_sample,
    normalize,
    residual_distribution,
    top_p,
)
from specdecode.speculative import RESIDUAL_RENORM_TARGET


def small_machines(eos_id=2):
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
        eos_id=eos_id,
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
        eos_id=eos_id,
    )
    return target, draft


class PrimitiveTests(unittest.TestCase):
    def test_acceptance_zero_rules(self):
        self.assertEqual(accept_prob(Fraction(0), Fraction(0)), 0)
        self.assertEqual(accept_prob(Fraction(0), Fraction(1, 2)), 0)
        self.assertEqual(accept_prob(Fraction(1), Fraction(1, 3)), Fraction(1, 3))
        self.assertEqual(accept_prob(Fraction(1, 3), Fraction(2, 3)), 1)

    def test_residual_mass_identity(self):
        p = normalize([Fraction(1), Fraction(2), Fraction(1)])
        q = normalize([Fraction(3), Fraction(1), Fraction(1)])
        r, z = residual_distribution(p, q)
        self.assertAlmostEqual(float(z), float(sum(max(0, a - b) for a, b in zip(q, p))))
        self.assertAlmostEqual(float(sum(r)), 1.0)
        # Identity: Z = 1 - sum_i min(p_i, q_i)
        z_alt = 1 - sum(min(pi, qi) for pi, qi in zip(p, q))
        self.assertEqual(z, z_alt)

    def test_zero_rejection_mass(self):
        q = normalize([Fraction(1), Fraction(2)])
        p = normalize([Fraction(2), Fraction(4)])  # identical ratio
        _, z = residual_distribution(p, q)
        self.assertEqual(z, 0)

    def test_top_p_truncation_basic(self):
        w = [Fraction(5), Fraction(3), Fraction(1), Fraction(1)]
        truncated = normalize(top_p(w, Fraction(8, 10)))
        # 1/2 alone < 0.8, +3/10 = 0.8 -> first two kept exactly
        self.assertEqual(truncated, normalize([Fraction(5), Fraction(3), 0, 0]))

    def test_exact_categorical_sampler_is_unbiased(self):
        probs = normalize([Fraction(1), Fraction(3), Fraction(2)])
        counts = [0, 0, 0]
        rng = random.Random(123)
        n = 60_000
        for _ in range(n):
            counts[categorical_sample(probs, rng)] += 1
        for got, expected in zip(counts, probs):
            self.assertLess(abs(got / n - float(expected)), 0.01)


class ExactLawTests(unittest.TestCase):
    def _check(self, V, K, L, eos_id, *, temperature=1, cutoff=1,
               greedy_draft=False, seed=7, zeros=0.0):
        target_model = RandomWeightsModel(
            V, seed=seed, eos_id=eos_id, eos_boost=2, zeros=zeros
        )
        if greedy_draft:
            draft_model = GreedyWrapper(
                RandomWeightsModel(V, seed=seed + 100, eos_id=eos_id,
                                   eos_boost=2, zeros=zeros)
            )
        else:
            draft_model = RandomWeightsModel(
                V, seed=seed + 1, eos_id=eos_id, eos_boost=1, zeros=zeros
            )
        tp = DistributionProvider(target_model, temperature=temperature,
                                  top_p_cutoff=cutoff)
        dp = DistributionProvider(draft_model, temperature=temperature,
                                  top_p_cutoff=cutoff)
        spec = exact_speculative(tp, dp, (), max_new_tokens=L, K=K)
        ref = exact_target(tp, (), max_new_tokens=L)
        self.assertEqual(spec.law, ref.law)
        # Expected useful target positions must match naive autoregressive.
        self.assertEqual(
            spec.counters["target_positions_useful"],
            ref.counters["target_positions_useful"],
        )
        return spec, ref

    def test_grid_exact_equivalence(self):
        for V in (2, 3):
            for K in (0, 1, 2, 3):
                for L in (1, 2, 3):
                    with self.subTest(V=V, K=K, L=L, eos=True):
                        self._check(V, K, L, eos_id=V - 1)
                    with self.subTest(V=V, K=K, L=L, eos=False):
                        self._check(V, K, L, eos_id=None)

    def test_greedy_degenerate_proposal(self):
        # Greedy one-hot drafts: zero probabilities must be handled exactly.
        spec, ref = self._check(3, K=2, L=3, eos_id=2, greedy_draft=True)
        self.assertEqual(spec.law, ref.law)

    def test_hard_zero_rows_and_no_rejection_mass(self):
        target, draft = small_machines()
        tp = DistributionProvider(target)
        dp = DistributionProvider(draft)
        spec = exact_speculative(tp, dp, (1,), max_new_tokens=3, K=2)
        ref = exact_target(tp, (1,), max_new_tokens=3)
        self.assertEqual(spec.law, ref.law)
        # Identical-policy case has Z = 0 everywhere but zero rejection mass.
        same = DistributionProvider(target)
        spec2 = exact_speculative(same, DistributionProvider(target),
                                  (), max_new_tokens=3, K=2)
        ref2 = exact_target(DistributionProvider(target), (),
                            max_new_tokens=3)
        self.assertEqual(spec2.law, ref2.law)
        self.assertEqual(spec2.counters["correction_tokens"], 0)

    def test_k0_is_pure_autoregression(self):
        target, draft = small_machines()
        tp = DistributionProvider(target)
        spec = exact_speculative(tp, DistributionProvider(draft),
                                 (), max_new_tokens=3, K=0)
        ref = exact_target(tp, (), max_new_tokens=3)
        self.assertEqual(spec.law, ref.law)
        self.assertEqual(spec.counters["draft_queries"], 0)
        # Exactly one bonus token per round.
        self.assertEqual(spec.counters["bonus_tokens"],
                         ref.counters["emitted"])

    def test_eos_all_accept_has_no_bonus_and_no_correction(self):
        # Target forces EOS at the first position: draft always proposes it,
        # it is accepted, generation ends with exactly one useful position.
        target = TransitionTableModel(
            2, {(): [0, 1], (0,): [1, 0], (1,): [0, 1]}, eos_id=1)
        tp = DistributionProvider(target)
        spec = exact_speculative(tp, DistributionProvider(target),
                                 (), max_new_tokens=4, K=3)
        ref = exact_target(DistributionProvider(target), (),
                           max_new_tokens=4)
        self.assertEqual(spec.law, ref.law)
        self.assertEqual(spec.law, {(1,): Fraction(1)})
        c = spec.counters
        self.assertEqual(c["target_calls"], 1)
        self.assertEqual(c["target_positions_requested"], 1)
        self.assertEqual(c["target_positions_useful"], 1)
        self.assertEqual(c["bonus_tokens"], 0)
        self.assertEqual(c["correction_tokens"], 0)

        dec = SpeculativeDecoder(DistributionProvider(target, cache=False),
                                 DistributionProvider(target, cache=False),
                                 rng=random.Random(0))
        out = dec.generate((), max_new_tokens=4, K=3)
        self.assertEqual(out, [1])
        lc = dec.counters
        self.assertEqual((lc.target_calls, lc.target_positions_requested,
                          lc.target_positions_useful, lc.bonus_tokens),
                         (1, 1, 1, 0))

    def test_eos_and_length_bounds(self):
        target, draft = small_machines(eos_id=2)
        tp = DistributionProvider(target)
        dp = DistributionProvider(draft)
        for L in (1, 2, 4):
            spec = exact_speculative(tp, dp, (0,), max_new_tokens=L, K=3)
            ref = exact_target(tp, (0,), max_new_tokens=L)
            self.assertEqual(spec.law, ref.law)
            for seq in spec.law:
                # EOS only appears as the final token and length is bounded.
                self.assertLessEqual(len(seq), L)
                if 2 in seq:
                    self.assertEqual(seq.index(2), len(seq) - 1)

    def test_top_p_verifies_post_transform_law(self):
        target = RandomWeightsModel(3, seed=11, eos_id=2, eos_boost=2)
        draft = RandomWeightsModel(3, seed=22, eos_id=2, eos_boost=1)
        tp = DistributionProvider(target, top_p_cutoff=Fraction(7, 10))
        dp = DistributionProvider(draft, top_p_cutoff=Fraction(7, 10))
        spec = exact_speculative(tp, dp, (), max_new_tokens=3, K=2)
        ref = exact_target(tp, (), max_new_tokens=3)
        self.assertEqual(spec.law, ref.law)
        # Negative control: must NOT match the untruncated raw target law.
        raw = DistributionProvider(target, top_p_cutoff=1)
        raw_ref = exact_target(raw, (), max_new_tokens=3)
        self.assertNotEqual(ref.law, raw_ref.law)

    def test_temperature_half_exact(self):
        target = RandomWeightsModel(3, seed=5, eos_id=2)
        draft = RandomWeightsModel(3, seed=6, eos_id=2)
        tp = DistributionProvider(target, temperature=Fraction(1, 2))
        dp = DistributionProvider(draft, temperature=Fraction(1, 2))
        spec = exact_speculative(tp, dp, (), max_new_tokens=3, K=2)
        ref = exact_target(tp, (), max_new_tokens=3)
        self.assertEqual(spec.law, ref.law)

    def test_random_sparse_zeros_exact(self):
        # Several histories will contain genuine zero-weight tokens.
        self._check(4, K=2, L=3, eos_id=3, zeros=0.35, seed=31)

    def test_deterministic_target_all_accept(self):
        target = TransitionTableModel(
            2, {(): [1, 0], (0,): [0, 1], (1,): [1, 0]}, eos_id=1)
        tp = DistributionProvider(target)
        spec = exact_speculative(tp, DistributionProvider(target),
                                 (), max_new_tokens=3, K=2)
        ref = exact_target(DistributionProvider(target), (),
                           max_new_tokens=3)
        self.assertEqual(spec.law, ref.law)
        self.assertEqual(spec.counters["correction_tokens"], 0)

    def test_different_prompt_prefixes_query_correct_histories(self):
        target, draft = small_machines()
        tp = DistributionProvider(target)
        dp = DistributionProvider(draft)
        a = exact_speculative(tp, dp, (0, 1), max_new_tokens=2, K=2)
        b = exact_speculative(tp, dp, (1, 0), max_new_tokens=2, K=2)
        ra = exact_target(tp, (0, 1), max_new_tokens=2)
        rb = exact_target(tp, (1, 0), max_new_tokens=2)
        self.assertEqual(a.law, ra.law)
        self.assertEqual(b.law, rb.law)
        self.assertNotEqual(ra.law, rb.law)


class WrongResidualTests(unittest.TestCase):
    def test_single_step_counterexample_numbers(self):
        d = single_step_demo()
        self.assertEqual(d["marginal_correct"], d["q"])
        self.assertEqual(d["marginal_wrong_renorm"],
                         (Fraction(3, 4), Fraction(1, 4)))
        self.assertEqual(d["tv_distance_wrong"], Fraction(1, 4))

    def test_wrong_residuals_break_exact_law(self):
        target, draft = small_machines()
        tp = DistributionProvider(target)
        dp = DistributionProvider(draft)
        good = exact_speculative(tp, dp, (), max_new_tokens=3, K=2)
        for fn in (renorm_target_residual, uniform_residual,
                   unclipped_residual):
            with self.subTest(fn=fn.__name__):
                bad = exact_speculative(tp, dp, (),
                                        max_new_tokens=3, K=2,
                                        residual_fn=fn)
                self.assertNotEqual(bad.law, good.law)

    def test_wrong_residual_live_driver(self):
        # No EOS: with L=K=1 every run proposes one token and goes through
        # accept/reject, so a wrong correction is directly observable.
        target = TransitionTableModel(
            3, {(): [2, 1, 1], (0,): [1, 2, 1], (1,): [2, 0, 1],
                (2,): [1, 1, 2]}, eos_id=None)
        draft = TransitionTableModel(
            3, {(): [1, 2, 1], (0,): [2, 1, 1], (1,): [1, 1, 1],
                (2,): [1, 2, 1]}, eos_id=None)
        tp = DistributionProvider(target)
        dp = DistributionProvider(draft)
        rng = random.Random(0)
        dec = SpeculativeDecoder(tp, dp, rng=rng,
                                 residual_mode=RESIDUAL_RENORM_TARGET)
        n = 12_000
        from collections import Counter

        agg = Counter()
        for _ in range(n):
            seq = tuple(dec.generate((), max_new_tokens=1, K=1))
            agg[seq] += 1
        q0 = tp.prob(())
        # First-token marginal must deviate from target under the bug.
        freqs = [agg[(i,)] / n for i in range(3)]
        tv = sum(abs(f - float(q)) for f, q in zip(freqs, q0)) / 2
        # Exact bias is 1/16 for this p,q pair (see single_step_demo style).
        self.assertGreater(tv, 0.04)


class LiveDriverTests(unittest.TestCase):
    def test_live_matches_target_frequencies(self):
        target, draft = small_machines()
        tp = DistributionProvider(target)
        dp = DistributionProvider(draft)
        from collections import Counter

        n = 15_000
        agg = Counter()
        dec = SpeculativeDecoder(tp, dp, rng=random.Random(42))
        for _ in range(n):
            agg[tuple(dec.generate((), max_new_tokens=2, K=2))] += 1
        ref = exact_target(DistributionProvider(target), (),
                           max_new_tokens=2).law
        for seq, expected in ref.items():
            self.assertLess(abs(agg[seq] / n - float(expected)), 0.015)

    def test_live_counters_consistency(self):
        target, draft = small_machines()
        tp = DistributionProvider(target, cache=False)
        dp = DistributionProvider(draft, cache=False)
        dec = SpeculativeDecoder(tp, dp, rng=random.Random(7))
        out = dec.generate((0,), max_new_tokens=6, K=3)
        c = dec.counters
        self.assertEqual(c.emitted, len(out))
        self.assertGreaterEqual(c.target_positions_requested,
                                c.target_positions_useful)
        self.assertEqual(c.target_calls, c.rounds)
        self.assertGreaterEqual(c.draft_queries, c.accepted_draft_tokens)

    def test_k0_live_emits_one_token_per_target_position(self):
        target, draft = small_machines()
        tp = DistributionProvider(target, cache=False)
        dp = DistributionProvider(draft, cache=False)
        dec = SpeculativeDecoder(tp, dp, rng=random.Random(9))
        out = dec.generate((), max_new_tokens=5, K=0)
        c = dec.counters
        self.assertEqual(len(out), 5)
        self.assertEqual(c.target_positions_useful, 5)
        self.assertEqual(c.draft_queries, 0)

    def test_live_greedy_degenerate_draft(self):
        target = TransitionTableModel(
            3, {(): [2, 1, 1], (0,): [1, 2, 1], (1,): [2, 0, 1],
                (2,): [1, 1, 2]}, eos_id=None)
        greedy_draft = GreedyWrapper(
            TransitionTableModel(
                3,
                {(): [3, 1, 1], (0,): [1, 3, 1], (1,): [3, 0, 1],
                 (2,): [1, 1, 3]}))
        tp = DistributionProvider(target, cache=False)
        dp = DistributionProvider(greedy_draft, cache=False)
        dec = SpeculativeDecoder(tp, dp, rng=random.Random(11))
        out = dec.generate((), max_new_tokens=4, K=3)
        self.assertLessEqual(len(out), 4)
        from collections import Counter

        agg = Counter()
        n = 8_000
        tp2 = DistributionProvider(target)
        dp2 = DistributionProvider(greedy_draft)
        for seed in range(n):
            d2 = SpeculativeDecoder(tp2, dp2,
                                    rng=random.Random(10_000 + seed))
            agg[tuple(d2.generate((), max_new_tokens=1, K=1))] += 1
        q0 = tp2.prob(())
        for token, mass in enumerate(q0):
            self.assertLess(abs(agg[(token,)] / n - float(mass)), 0.02)


if __name__ == "__main__":
    unittest.main()
