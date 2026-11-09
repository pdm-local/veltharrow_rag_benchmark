"""Small, dependency-free statistics helpers shared across the analysis layer.

This module is intentionally **stdlib-only** (no numpy/pandas) so that any script —
including the standalone plotting and table generators in ``scripts/`` and ``tools/`` —
can import it without pulling in the plotting stack. Functions here operate on plain
counts, not DataFrames, and never touch any result-keyed identifiers (model labels,
tier names, retrieval-condition strings).
"""
from __future__ import annotations

import math


def wilson_ci(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Returns the ``(low, high)`` bounds **as percentages** (0–100). The Wilson
    interval is used instead of the normal approximation because it stays inside
    ``[0, 1]`` and behaves well for the small per-cell sample sizes here (n=400
    MCQs per model×condition). ``z=1.96`` gives a 95% interval.

    A ``total`` of 0 returns ``(0.0, 0.0)`` rather than dividing by zero.
    """
    if total == 0:
        return 0.0, 0.0
    p = correct / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return max(0.0, (centre - margin) * 100), min(100.0, (centre + margin) * 100)
