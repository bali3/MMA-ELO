"""Tests for ablation scenario configuration and filtering."""

from __future__ import annotations

import pytest

from evaluation import default_ablation_configs, filter_ablation_configs


def test_default_ablation_configs_cover_required_scenarios() -> None:
    configs = default_ablation_configs()
    ids = {config.scenario_id for config in configs}
    expected = {
        "market_only_no_interactions",
        "market_only_with_interactions",
        "elo_only_no_interactions",
        "elo_only_with_interactions",
        "features_only_no_interactions",
        "features_only_with_interactions",
        "features_elo_no_interactions",
        "features_elo_with_interactions",
        "features_elo_market_no_interactions",
        "features_elo_market_with_interactions",
    }
    assert ids == expected
    assert len(configs) == 10


def test_filter_ablation_configs_supports_include_and_exclude_groups() -> None:
    configs = default_ablation_configs()
    selected = filter_ablation_configs(
        configs=configs,
        include_groups=["elo"],
        exclude_groups=["market"],
    )
    selected_ids = {config.scenario_id for config in selected}
    assert selected_ids == {
        "elo_only_no_interactions",
        "elo_only_with_interactions",
        "features_elo_no_interactions",
        "features_elo_with_interactions",
    }


def test_filter_ablation_configs_rejects_overlap_between_include_and_exclude() -> None:
    with pytest.raises(ValueError, match="both included and excluded"):
        filter_ablation_configs(
            configs=default_ablation_configs(),
            include_groups=["features"],
            exclude_groups=["features"],
        )
