from fractions import Fraction
from itertools import product
from pprint import pformat
import random
import unittest

from specdec.distributions import (
    acceptance_probability,
    residual_distribution,
    transform_distribution,
)
from specdec.exact import (
    direct_sequence_distribution,
    probability_with_wrong_residual,
    speculative_sequence_distribution,
)
from specdec.models import RandomFiniteModel, TableModel
from specdec.sampling import SamplingSpec, actual_distribution, speculative_sample


VOCAB = ("A", "B", "C", "<eos>")
EOS = "<eos>"


def random_model(seed: int, zero_rate: float = 0.2) -> RandomFiniteModel:
    return RandomFiniteModel(VOCAB, EOS, seed=seed, zero_rate=zero_rate)


def assert_distributions_equal(test, lhs, rhs):
    test.assertEqual(sum(lhs.values()), Fraction(1))
    test.assertEqual(sum(rhs.values()), Fraction(1))
    test.assertEqual(set(lhs), set(rhs), pformat((lhs, rhs)))
    for key in set(lhs) | set(rhs):
        test.assertEqual(lhs.get(key, 0), rhs.get(key, 0), pformat((key, lhs, rhs)))


class ExactDistributionTests(unittest.TestCase):
    def test_exact_equivalence_many_models_and_limits(self):
        specs = [
            SamplingSpec(),
            SamplingSpec(temperature=Fraction(1, 2)),
            SamplingSpec(top_p=Fraction(3, 4)),
            SamplingSpec(top_k=2),
            SamplingSpec(temperature=Fraction(1, 2), top_k=2, top_p=Fraction(4, 5)),
        ]
        for target_seed in range(4):
            target = random_model(100 + target_seed)
            for draft_seed in range(3):
                draft = random_model(200 + draft_seed)
                for k in range(4):
                    for spec in specs:
                        with self.subTest(
                            target=target_seed,
                            draft=draft_seed,
                            k=k,
                            spec=spec,
                        ):
                            direct = direct_sequence_distribution(
                                target, (), spec, 3
                            )
                            speculative = speculative_sequence_distribution(
                                target, draft, (), spec, 3, k
                            )
                            assert_distributions_equal(self, direct, speculative)

    def test_different_history_prefixes(self):
        target = random_model(301)
        draft = random_model(401)
        prefix = ("A", "C")
        spec = SamplingSpec(top_p=Fraction(2, 3))
        direct = direct_sequence_distribution(target, prefix, spec, 2)
        speculative = speculative_sequence_distribution(
            target, draft, prefix, spec, 2, 2
        )
        assert_distributions_equal(self, direct, speculative)

    def test_zero_probability_residual(self):
        p = {"A": Fraction(4, 10), "B": Fraction(1, 10), "C": Fraction(5, 10)}
        q = {"A": Fraction(2, 10), "B": Fraction(5, 10), "C": Fraction(3, 10)}
        residual = residual_distribution(p, q)
        self.assertEqual(
            residual,
            {"A": Fraction(1, 2), "C": Fraction(1, 2)},
        )
        self.assertEqual(acceptance_probability(p, q, "missing"), 0)
        self.assertEqual(acceptance_probability(p, q, "A"), 1)
        self.assertEqual(acceptance_probability(p, q, "B"), Fraction(1, 5))

    def test_no_rejection_mass_when_distributions_match(self):
        p = {"A": Fraction(3, 10), "B": Fraction(7, 10)}
        self.assertEqual(residual_distribution(p, p), {})
        self.assertEqual(acceptance_probability(p, p, "A"), 1)

    def test_greedy_draft_is_not_given_fake_support(self):
        p = {"A": Fraction(2, 10), "B": Fraction(8, 10)}
        greedy_q = {"A": 0, "B": 1}
        self.assertEqual(acceptance_probability(p, greedy_q, "A"), 0)
        self.assertEqual(acceptance_probability(p, greedy_q, "B"), Fraction(4, 5))
        self.assertEqual(residual_distribution(p, greedy_q), {"A": 1})

    def test_wrong_residual_target_resample_does_not_match(self):
        table = {
            (): {"A": 4, "B": 1, "C": 5, EOS: 0},
            ("A",): {"A": 1, "B": 2, "C": 3, EOS: 4},
            ("B",): {"A": 5, "B": 0, "C": 2, EOS: 3},
            ("C",): {"A": 0, "B": 4, "C": 1, EOS: 5},
        }
        target = TableModel(VOCAB, EOS, table)
        draft = TableModel(VOCAB, EOS, {key: dict(value) for key, value in table.items()})
        # Distinct draft weights.
        draft.table = {
            (): {"A": 2, "B": 3, "C": 5, EOS: 0},
            ("A",): {"A": 4, "B": 1, "C": 2, EOS: 3},
            ("B",): {"A": 1, "B": 3, "C": 4, EOS: 2},
            ("C",): {"A": 3, "B": 2, "C": 4, EOS: 1},
        }
        direct = direct_sequence_distribution(target, (), SamplingSpec(), 2)
        wrong = probability_with_wrong_residual(
            target,
            draft,
            (),
            SamplingSpec(),
            2,
            2,
            lambda p, q: {token: value for token, value in p.items() if value > 0},
        )
        self.assertNotEqual(direct, wrong)


    def test_greedy_draft_exact_sequence_distribution(self):
        target = random_model(701, zero_rate=0.25)
        greedy_weights = {}
        for length in range(3):
            import itertools
            for path in itertools.product(VOCAB, repeat=length):
                transformed = actual_distribution(target, path, SamplingSpec())
                winner = min(transformed, key=lambda token: (-transformed[token], token))
                greedy_weights[path] = {
                    token: (Fraction(1) if token == winner else Fraction(0))
                    for token in VOCAB
                }
        draft = TableModel(VOCAB, EOS, greedy_weights)
        direct = direct_sequence_distribution(target, (), SamplingSpec(), 3)
        speculative = speculative_sequence_distribution(
            target, draft, (), SamplingSpec(), 3, 3
        )
        assert_distributions_equal(self, direct, speculative)


class SamplerTests(unittest.TestCase):
    def test_k_zero_samples_one_target_position_per_round(self):
        target = random_model(501)
        draft = random_model(601)
        rng = random.Random(7)
        output, stats = speculative_sample(
            target, draft, (), SamplingSpec(), 5, rng, max_draft_tokens=0
        )
        self.assertLessEqual(len(output), 5)
        self.assertEqual(stats.draft_positions, 0)
        self.assertEqual(stats.target_positions, len(output))
        self.assertEqual(stats.target_forwards, len(output))
        self.assertEqual(stats.generated_tokens, len(output))

    def test_eos_and_remaining_length_bounds(self):
        table = {(): {"A": 2, "B": 3, "C": 0, EOS: 5}}
        eos_distribution = {token: 0 for token in VOCAB}
        eos_distribution[EOS] = 1
        uniform = {"A": 1, "B": 1, "C": 1, EOS: 1}
        for length in range(1, 4):
            for path in product(VOCAB, repeat=length):
                table[path] = eos_distribution if EOS in path else uniform
        model = TableModel(VOCAB, EOS, table)
        for seed in range(20):
            output, stats = speculative_sample(
                model, model, (), SamplingSpec(), 4, random.Random(seed), 3
            )
            self.assertLessEqual(len(output), 4)
            if EOS in output:
                self.assertEqual(output[-1], EOS)
                self.assertEqual(len(output) - 1, output.index(EOS))
            self.assertEqual(stats.generated_tokens, len(output))
            self.assertGreaterEqual(stats.target_positions, len(output))

    def test_runtime_uses_transformed_support(self):
        raw = {"A": 8, "B": 1, "C": 1, EOS: 0}
        spec = SamplingSpec(top_p=0.8)
        transformed = transform_distribution(raw, top_p=0.8)
        self.assertEqual(set(transformed), {"A"})
        eos_table = {token: 0 for token in VOCAB}
        eos_table[EOS] = 1
        target = TableModel(VOCAB, EOS, {(): raw})
        draft = TableModel(VOCAB, EOS, {(): raw})
        output, _ = speculative_sample(
            target, draft, (), spec, 1, random.Random(0), 1
        )
        self.assertEqual(output, ["A"])

    def test_all_accept_bonus_uses_prefix_after_candidates(self):
        eos_distribution = {token: 0 for token in VOCAB}
        eos_distribution[EOS] = 1
        table = {
            (): {"A": 1, "B": 0, "C": 0, EOS: 0},
            ("A",): {"A": 1, "B": 0, "C": 0, EOS: 0},
            ("A", "A"): {"A": 0, "B": 0, "C": 0, EOS: 1},
        }
        model = TableModel(VOCAB, EOS, table)
        output, stats = speculative_sample(
            model, model, (), SamplingSpec(), 3, random.Random(0), 2
        )
        self.assertEqual(output, ["A", "A", EOS])
        self.assertEqual(stats.accepted_draft_tokens, 2)
        self.assertEqual(stats.bonus_tokens, 1)
        self.assertEqual(stats.target_positions, 3)
        self.assertEqual(stats.target_forwards, 1)

    def test_rejection_counts_wasted_candidate_and_bonus_positions(self):
        eos_distribution = {token: 0 for token in VOCAB}
        eos_distribution[EOS] = 1
        target = TableModel(
            VOCAB,
            EOS,
            {
                (): {"A": Fraction(1, 2), "B": Fraction(1, 2), "C": 0, EOS: 0},
                ("A",): eos_distribution,
                ("B",): eos_distribution,
            },
        )
        draft = TableModel(
            VOCAB,
            EOS,
            {
                (): {"A": 1, "B": 0, "C": 0, EOS: 0},
                ("A",): eos_distribution,
            },
        )
        output, stats = speculative_sample(
            target, draft, (), SamplingSpec(), 1, random.Random(0), 2
        )
        self.assertEqual(stats.rejection_events, 1)
        self.assertEqual(stats.correction_tokens, 1)
        self.assertEqual(stats.wasted_target_positions, 0)


    def test_temperature_zero_is_deterministic_target(self):
        table = {
            (): {"A": 4, "B": 5, "C": 1, EOS: 0},
            ("A",): {"A": 1, "B": 1, "C": 1, EOS: 1},
            ("B",): {"A": 9, "B": 0, "C": 1, EOS: 0},
            ("C",): {"A": 1, "B": 1, "C": 1, EOS: 1},
            ("A", "A"): {"A": 1, "B": 0, "C": 0, EOS: 0},
            ("A", "B"): {"A": 1, "B": 0, "C": 0, EOS: 0},
            ("A", "C"): {"A": 1, "B": 0, "C": 0, EOS: 0},
            ("A", EOS): {"A": 0, "B": 0, "C": 0, EOS: 1},
            ("B", "A"): {"A": 1, "B": 0, "C": 0, EOS: 0},
            ("C", "A"): {"A": 1, "B": 0, "C": 0, EOS: 0},
            ("B", EOS): {"A": 0, "B": 0, "C": 0, EOS: 1},
            ("C", EOS): {"A": 0, "B": 0, "C": 0, EOS: 1},
        }
        target = TableModel(VOCAB, EOS, table)
        draft_table = {
            (): {"A": 1, "B": 1, "C": 1, EOS: 0},
            ("A",): {"A": 1, "B": 1, "C": 1, EOS: 1},
            ("B",): {"A": 1, "B": 1, "C": 1, EOS: 1},
            ("C",): {"A": 1, "B": 1, "C": 1, EOS: 1},
            ("B", "A"): {"A": 1, "B": 1, "C": 1, EOS: 1},
        }
        draft = TableModel(VOCAB, EOS, draft_table)
        direct = direct_sequence_distribution(
            target, (), SamplingSpec(temperature=0), 2
        )
        speculative = speculative_sequence_distribution(
            target, draft, (), SamplingSpec(temperature=0), 2, 2
        )
        self.assertEqual(direct, {("B", "A"): 1})
        self.assertEqual(speculative, direct)


if __name__ == "__main__":
    unittest.main()
