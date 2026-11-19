import pytest

from experiments.openevolve.colab.paper2_watchdog import (
    CONFIG,
    ROOT,
    budget_exhausted,
    parse_usage_balance,
)


def test_parse_usage_balance_from_colab_cli():
    output = "Current balance: 224.37 compute units\nUsage rate: 1.94/hr\nActive assignments: 1"

    assert parse_usage_balance(output) == pytest.approx(224.37)


def test_budget_stops_only_at_or_above_preregistered_cap():
    assert budget_exhausted(224.37, 209.38, 15.0) is False
    assert budget_exhausted(224.37, 209.37, 15.0) is True


def test_mutable_watchdog_control_state_is_inside_writable_root():
    assert CONFIG.is_relative_to(ROOT)
