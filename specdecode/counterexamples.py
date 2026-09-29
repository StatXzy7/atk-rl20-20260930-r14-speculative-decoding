"""Deliberately incorrect residuals used as negative controls."""

from __future__ import annotations

from fractions import Fraction

from .probability import Distribution


def renorm_target_residual(
    history: tuple[int, ...], p: Distribution, q: Distribution
) -> Distribution:
    """Wrong: sample the correction directly from target q.

    This double-counts the target mass already consumed by accepted draft
    tokens.  The marginal becomes
    ``p(x) min(1, q_x/p_x) + (1 - alpha) q_x`` instead of q_x.
    """
    return list(q)


def uniform_residual(
    history: tuple[int, ...], p: Distribution, q: Distribution
) -> Distribution:
    """Wrong: uniform correction regardless of q - p overshoot."""
    v = Fraction(1, len(q))
    return [v] * len(q)


def unclipped_residual(
    history: tuple[int, ...], p: Distribution, q: Distribution
) -> Distribution:
    """Wrong: normalize q - p without clipping negative entries.

    If un-normalized entries sum to zero this is undefined; the shift used
    here keeps it a distribution but no longer equals the rejection mass.
    """
    from .probability import normalize

    raw = [qi - pi for pi, qi in zip(p, q)]
    shift = min(min(raw), Fraction(0))
    return normalize([r - shift for r in raw])


def mismatched_proposal_residual(
    history: tuple[int, ...], p: Distribution, q: Distribution
) -> Distribution:
    """Wrong: acceptance uses true p, but the residual reuses stale weights.

    Simulates an implementation bug where the correction branch computes
    ``max(0, q - p')`` with a proposal p' that differs from the one actually
    sampled (e.g. cached under a different history).
    """
    from .probability import residual_distribution

    # Fabricated "stale" proposal: swap first two coordinates.
    p_wrong = [p[1], p[0], *p[2:]]
    dist, _ = residual_distribution(p_wrong, q)
    return dist


def single_step_demo() -> dict:
    """Explicit one-token counterexample over {0, 1}.

    target q = (1/2, 1/2), draft p = (1, 0).  The greedy draft always
    proposes token 0:

    * token 0 accepted with prob 1/2 (mass consumed p_0*1 = 1/2);
    * token 0 rejected with prob 1/2, then correction must follow the
      residual (0, 1) so token 1 gets mass 1/2.

    Re-normalizing target on the rejection branch gives correction (1/2,1/2),
    yielding marginal (3/4, 1/4) -- a detectable, systematic bias.
    """
    p = (Fraction(1), Fraction(0))
    q = (Fraction(1, 2), Fraction(1, 2))
    alpha = sum(min(pi, qi) for pi, qi in zip(p, q))
    correct = (Fraction(0), Fraction(1))
    wrong = list(q)
    marginal_correct = tuple(
        min(pi, qi) + (1 - alpha) * ci for pi, qi, ci in zip(p, q, correct)
    )
    marginal_wrong = tuple(
        min(pi, qi) + (1 - alpha) * wi for pi, qi, wi in zip(p, q, wrong)
    )
    return {
        "p": p,
        "q": q,
        "accept_total_alpha": alpha,
        "reject_mass_Z": 1 - alpha,
        "correct_residual": correct,
        "wrong_residual_renorm": tuple(wrong),
        "marginal_correct": marginal_correct,
        "marginal_wrong_renorm": marginal_wrong,
        "tv_distance_wrong": sum(
            abs(a - b) for a, b in zip(marginal_wrong, q)
        )
        / 2,
    }
