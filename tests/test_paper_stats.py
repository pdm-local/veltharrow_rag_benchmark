"""Check the paired statistical primitives against analytically known cases."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    'paper_stats', Path(__file__).resolve().parents[1] / 'tools/stats_tests.py'
)
stats = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stats)


def test_mcnemar_uses_discordant_pairs():
    a = dict(enumerate([False] * 6 + [True] * 4))
    b = dict(enumerate([True] * 10))
    assert stats.mcnemar_exact(a, b) == (10, 6, 0, 0.03125)
    assert stats.mcnemar_exact(a, a) == (10, 0, 0, 1.0)


def test_balanced_discordance_has_unit_p_value():
    assert stats.mcnemar_exact({'a': False, 'b': True},
                               {'a': True, 'b': False}) == (2, 1, 1, 1.0)


def test_holm_restores_order_and_enforces_monotonicity():
    assert stats.holm([.04, .001, .03, .9]) == pytest.approx([.09, .004, .09, .9])
