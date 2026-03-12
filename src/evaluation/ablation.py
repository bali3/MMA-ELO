"""Chronology-safe ablation experiments for major feature groups."""

from __future__ import annotations

import csv
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Sequence

from features import generate_pre_fight_features
from models.baselines import TABULAR_FEATURE_COLUMNS
from ratings import EloConfig, generate_elo_ratings_history

from evaluation.walk_forward import build_walk_forward_folds

FeatureGroup = Literal["features", "elo", "market"]
WindowType = Literal["expanding", "rolling"]

VALID_FEATURE_GROUPS: tuple[FeatureGroup, ...] = ("features", "elo", "market")


@dataclass(frozen=True)
class AblationConfig:
    """Single ablation scenario configuration."""

    scenario_id: str
    include_features: bool
    include_elo: bool
    include_market: bool
    include_matchup_interactions: bool

    def active_groups(self) -> tuple[FeatureGroup, ...]:
        groups: list[FeatureGroup] = []
        if self.include_features:
            groups.append("features")
        if self.include_elo:
            groups.append("elo")
        if self.include_market:
            groups.append("market")
        return tuple(groups)


@dataclass(frozen=True)
class AblationScenarioMetrics:
    """Aggregate out-of-sample metrics for one ablation scenario."""

    config: AblationConfig
    fold_count: int
    predictions_count: int
    log_loss: float
    brier_score: float
    hit_rate: float
    calibration_ece: float


@dataclass(frozen=True)
class AblationExperimentResult:
    """Ablation experiment output artifacts and metrics."""

    model_run_id: str
    window_type: WindowType
    fold_count: int
    predictions_path: Path
    comparison_report_path: Path
    contribution_report_path: Path
    scenario_metrics: tuple[AblationScenarioMetrics, ...]


@dataclass(frozen=True)
class _BoutExample:
    bout_id: str
    event_id: str
    bout_datetime_utc: str
    bout_order: int
    fighter_id: str
    opponent_id: str
    label: float | None
    feature_cutoff_utc: str
    pre_fight_features: dict[str, float | None]
    pre_overall_elo: float | None
    opponent_pre_overall_elo: float | None
    market_implied_probability: float | None

    @property
    def elo_diff(self) -> float | None:
        if self.pre_overall_elo is None or self.opponent_pre_overall_elo is None:
            return None
        return self.pre_overall_elo - self.opponent_pre_overall_elo


@dataclass(frozen=True)
class _BoutTimelineEntry:
    bout_id: str
    bout_datetime_utc: str
    bout_order: int


@dataclass(frozen=True)
class _ModelRow:
    model_run_id: str
    model_name: str
    model_version: str
    scenario_id: str
    include_features: bool
    include_elo: bool
    include_market: bool
    include_matchup_interactions: bool
    window_type: WindowType
    fold_index: int
    train_start_utc: str
    train_end_utc: str
    test_start_utc: str
    test_end_utc: str
    bout_id: str
    event_id: str
    bout_datetime_utc: str
    fighter_id: str
    opponent_id: str
    label: float
    predicted_probability: float
    feature_cutoff_utc: str


@dataclass
class _LogisticModel:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float

    def predict_probability(self, feature_values: Sequence[float | None]) -> float:
        z = self.bias
        for index, value in enumerate(feature_values):
            filled = self.means[index] if value is None else value
            standardized = (filled - self.means[index]) / self.stds[index]
            z += self.weights[index] * standardized
        return _clamp_probability(_sigmoid(z))


def default_ablation_configs() -> tuple[AblationConfig, ...]:
    """Return required ablation matrix (five group bundles x interaction toggle)."""

    base_variants = (
        ("market_only", False, False, True),
        ("elo_only", False, True, False),
        ("features_only", True, False, False),
        ("features_elo", True, True, False),
        ("features_elo_market", True, True, True),
    )
    configs: list[AblationConfig] = []
    for variant, include_features, include_elo, include_market in base_variants:
        configs.append(
            AblationConfig(
                scenario_id=f"{variant}_no_interactions",
                include_features=include_features,
                include_elo=include_elo,
                include_market=include_market,
                include_matchup_interactions=False,
            )
        )
        configs.append(
            AblationConfig(
                scenario_id=f"{variant}_with_interactions",
                include_features=include_features,
                include_elo=include_elo,
                include_market=include_market,
                include_matchup_interactions=True,
            )
        )
    return tuple(configs)


def filter_ablation_configs(
    *,
    configs: Sequence[AblationConfig],
    include_groups: Sequence[str] | None = None,
    exclude_groups: Sequence[str] | None = None,
) -> tuple[AblationConfig, ...]:
    """Apply include/exclude group filters to a config set."""

    include_set = {item.strip().lower() for item in (include_groups or []) if item.strip() != ""}
    exclude_set = {item.strip().lower() for item in (exclude_groups or []) if item.strip() != ""}
    valid_groups = set(VALID_FEATURE_GROUPS)
    invalid = (include_set | exclude_set) - valid_groups
    if invalid:
        raise ValueError(f"Unknown feature groups: {', '.join(sorted(invalid))}.")

    overlap = include_set & exclude_set
    if overlap:
        raise ValueError(
            f"Groups cannot be both included and excluded: {', '.join(sorted(overlap))}."
        )

    selected: list[AblationConfig] = []
    for config in configs:
        active = set(config.active_groups())
        if include_set and not include_set.issubset(active):
            continue
        if active & exclude_set:
            continue
        selected.append(config)

    if len(selected) == 0:
        raise ValueError("No ablation scenarios left after applying include/exclude filters.")
    return tuple(selected)


def run_ablation_experiments(
    *,
    db_path: Path,
    promotion: str,
    window_type: WindowType,
    min_train_bouts: int,
    test_bouts: int,
    rolling_train_bouts: int | None,
    model_version: str,
    features_output_path: Path,
    ratings_output_path: Path,
    predictions_output_path: Path,
    comparison_report_output_path: Path,
    contribution_report_output_path: Path,
    calibration_bins: int = 10,
    model_run_id: str | None = None,
    include_groups: Sequence[str] | None = None,
    exclude_groups: Sequence[str] | None = None,
) -> AblationExperimentResult:
    """Run chronology-safe walk-forward ablation experiments."""

    if min_train_bouts < 1:
        raise ValueError("min_train_bouts must be at least 1.")
    if test_bouts < 1:
        raise ValueError("test_bouts must be at least 1.")
    if calibration_bins < 2:
        raise ValueError("calibration_bins must be at least 2.")
    if window_type not in {"expanding", "rolling"}:
        raise ValueError("window_type must be 'expanding' or 'rolling'.")
    if window_type == "rolling" and rolling_train_bouts is None:
        raise ValueError("rolling_train_bouts is required when window_type='rolling'.")
    if rolling_train_bouts is not None and rolling_train_bouts < 1:
        raise ValueError("rolling_train_bouts must be at least 1 when provided.")

    configs = filter_ablation_configs(
        configs=default_ablation_configs(),
        include_groups=include_groups,
        exclude_groups=exclude_groups,
    )
    resolved_run_id = model_run_id or _build_model_run_id()

    generate_pre_fight_features(
        db_path=db_path,
        output_path=features_output_path,
        promotion=promotion,
    )
    generate_elo_ratings_history(
        db_path=db_path,
        output_path=ratings_output_path,
        config=EloConfig(promotion=promotion),
    )

    features_by_key = _load_feature_rows(features_output_path)
    elo_by_key = _load_elo_rows(ratings_output_path)
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        bout_rows = _load_bout_rows(connection=connection, promotion=promotion)
        market_by_key = _load_market_implied_probabilities(connection=connection, promotion=promotion)

    examples = _build_examples(
        bout_rows=bout_rows,
        features_by_key=features_by_key,
        elo_by_key=elo_by_key,
        market_by_key=market_by_key,
    )
    timeline = _build_timeline(examples=examples)
    folds = build_walk_forward_folds(
        timeline=timeline,
        window_type=window_type,
        min_train_bouts=min_train_bouts,
        test_bouts=test_bouts,
        rolling_train_bouts=rolling_train_bouts,
    )

    all_rows: list[_ModelRow] = []
    scenario_results: list[AblationScenarioMetrics] = []

    for config in configs:
        feature_names = _build_feature_schema(config)
        scenario_probabilities: list[float] = []
        scenario_labels: list[float] = []

        for fold in folds:
            train_lookup = set(fold.train_bout_ids)
            test_lookup = set(fold.test_bout_ids)

            train_rows = [
                row for row in examples if row.bout_id in train_lookup and row.label is not None
            ]
            test_rows = [
                row for row in examples if row.bout_id in test_lookup and row.label is not None
            ]
            if len(train_rows) == 0:
                raise ValueError(f"Fold {fold.fold_index} has no train rows for scenario {config.scenario_id}.")
            if len(test_rows) == 0:
                raise ValueError(f"Fold {fold.fold_index} has no test rows for scenario {config.scenario_id}.")

            train_vectors = [_vector_for_example(row, feature_names=feature_names) for row in train_rows]
            test_vectors = [_vector_for_example(row, feature_names=feature_names) for row in test_rows]
            model = _fit_logistic_model(
                features=train_vectors,
                labels=[float(row.label) for row in train_rows],
                feature_names=feature_names,
            )

            for row, vector in zip(test_rows, test_vectors):
                label = float(row.label)
                probability = model.predict_probability(vector)
                scenario_labels.append(label)
                scenario_probabilities.append(probability)
                all_rows.append(
                    _ModelRow(
                        model_run_id=resolved_run_id,
                        model_name="ablation_logistic",
                        model_version=model_version,
                        scenario_id=config.scenario_id,
                        include_features=config.include_features,
                        include_elo=config.include_elo,
                        include_market=config.include_market,
                        include_matchup_interactions=config.include_matchup_interactions,
                        window_type=window_type,
                        fold_index=fold.fold_index,
                        train_start_utc=fold.train_start_utc,
                        train_end_utc=fold.train_end_utc,
                        test_start_utc=fold.test_start_utc,
                        test_end_utc=fold.test_end_utc,
                        bout_id=row.bout_id,
                        event_id=row.event_id,
                        bout_datetime_utc=row.bout_datetime_utc,
                        fighter_id=row.fighter_id,
                        opponent_id=row.opponent_id,
                        label=label,
                        predicted_probability=probability,
                        feature_cutoff_utc=row.feature_cutoff_utc,
                    )
                )

        calibration = _build_calibration_summary(
            labels=scenario_labels,
            probabilities=scenario_probabilities,
            num_bins=calibration_bins,
        )
        scenario_results.append(
            AblationScenarioMetrics(
                config=config,
                fold_count=len(folds),
                predictions_count=len(scenario_labels),
                log_loss=_log_loss(scenario_labels, scenario_probabilities),
                brier_score=_brier_score(scenario_labels, scenario_probabilities),
                hit_rate=_hit_rate(scenario_labels, scenario_probabilities),
                calibration_ece=calibration.expected_calibration_error,
            )
        )

    _write_predictions_csv(output_path=predictions_output_path, rows=all_rows)
    _write_comparison_report(
        output_path=comparison_report_output_path,
        model_run_id=resolved_run_id,
        window_type=window_type,
        fold_count=len(folds),
        scenario_metrics=scenario_results,
    )
    _write_contribution_report(
        output_path=contribution_report_output_path,
        model_run_id=resolved_run_id,
        scenario_metrics=scenario_results,
    )

    return AblationExperimentResult(
        model_run_id=resolved_run_id,
        window_type=window_type,
        fold_count=len(folds),
        predictions_path=predictions_output_path,
        comparison_report_path=comparison_report_output_path,
        contribution_report_path=contribution_report_output_path,
        scenario_metrics=tuple(scenario_results),
    )


def _build_feature_schema(config: AblationConfig) -> tuple[str, ...]:
    names: list[str] = []
    if config.include_features:
        names.extend(TABULAR_FEATURE_COLUMNS)
    if config.include_elo:
        names.append("elo_diff")
    if config.include_market:
        names.append("market_implied_probability")

    if config.include_matchup_interactions:
        if config.include_features and config.include_elo:
            names.extend([f"x:{feature}:elo_diff" for feature in TABULAR_FEATURE_COLUMNS])
        if config.include_features and config.include_market:
            names.extend([f"x:{feature}:market_implied_probability" for feature in TABULAR_FEATURE_COLUMNS])
        if config.include_elo and config.include_market:
            names.append("x:elo_diff:market_implied_probability")

    if len(names) == 0:
        raise ValueError("Ablation config must include at least one base feature group.")
    return tuple(names)


def _vector_for_example(example: _BoutExample, *, feature_names: Sequence[str]) -> list[float | None]:
    base_values: dict[str, float | None] = {
        **{name: example.pre_fight_features.get(name) for name in TABULAR_FEATURE_COLUMNS},
        "elo_diff": example.elo_diff,
        "market_implied_probability": example.market_implied_probability,
    }
    output: list[float | None] = []
    for name in feature_names:
        if name in base_values:
            output.append(base_values[name])
            continue

        if name.startswith("x:"):
            parts = name.split(":")
            if len(parts) != 3:
                raise ValueError(f"Unexpected interaction feature name: {name}")
            left = base_values.get(parts[1])
            right = base_values.get(parts[2])
            output.append(_product_or_none(left, right))
            continue
        raise ValueError(f"Unknown feature name: {name}")
    return output


def _product_or_none(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left * right


def _fit_logistic_model(
    *,
    features: Sequence[Sequence[float | None]],
    labels: Sequence[float],
    feature_names: Sequence[str],
    learning_rate: float = 0.05,
    iterations: int = 600,
    l2_penalty: float = 0.001,
) -> _LogisticModel:
    if len(features) == 0:
        raise ValueError("Logistic model training requires at least one row.")
    if len(features) != len(labels):
        raise ValueError("Features and labels length mismatch.")
    dimension = len(feature_names)
    if dimension == 0:
        raise ValueError("At least one feature is required.")

    means: list[float] = []
    stds: list[float] = []
    for index in range(dimension):
        values = [row[index] for row in features if row[index] is not None]
        mean = sum(values) / float(len(values)) if len(values) > 0 else 0.0
        variance = (
            sum((value - mean) ** 2 for value in values) / float(len(values))
            if len(values) > 0
            else 0.0
        )
        std = math.sqrt(variance)
        means.append(mean)
        stds.append(std if std > 1e-8 else 1.0)

    weights = [0.0 for _ in range(dimension)]
    bias = 0.0
    row_count = float(len(features))

    for _ in range(iterations):
        grad_w = [0.0 for _ in range(dimension)]
        grad_b = 0.0
        for row_index, row in enumerate(features):
            transformed: list[float] = []
            for index in range(dimension):
                value = row[index]
                filled = means[index] if value is None else value
                transformed.append((filled - means[index]) / stds[index])

            prediction = _sigmoid(bias + sum(weights[i] * transformed[i] for i in range(dimension)))
            error = prediction - labels[row_index]
            grad_b += error
            for index in range(dimension):
                grad_w[index] += error * transformed[index]

        for index in range(dimension):
            grad_w[index] = (grad_w[index] / row_count) + (l2_penalty * weights[index])
            weights[index] -= learning_rate * grad_w[index]
        bias -= learning_rate * (grad_b / row_count)

    return _LogisticModel(
        feature_names=tuple(feature_names),
        means=tuple(means),
        stds=tuple(stds),
        weights=tuple(weights),
        bias=bias,
    )


def _write_predictions_csv(*, output_path: Path, rows: Sequence[_ModelRow]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "model_run_id",
        "model_name",
        "model_version",
        "scenario_id",
        "include_features",
        "include_elo",
        "include_market",
        "include_matchup_interactions",
        "window_type",
        "fold_index",
        "train_start_utc",
        "train_end_utc",
        "test_start_utc",
        "test_end_utc",
        "bout_id",
        "event_id",
        "bout_datetime_utc",
        "fighter_id",
        "opponent_id",
        "label",
        "predicted_probability",
        "feature_cutoff_utc",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model_run_id": row.model_run_id,
                    "model_name": row.model_name,
                    "model_version": row.model_version,
                    "scenario_id": row.scenario_id,
                    "include_features": int(row.include_features),
                    "include_elo": int(row.include_elo),
                    "include_market": int(row.include_market),
                    "include_matchup_interactions": int(row.include_matchup_interactions),
                    "window_type": row.window_type,
                    "fold_index": row.fold_index,
                    "train_start_utc": row.train_start_utc,
                    "train_end_utc": row.train_end_utc,
                    "test_start_utc": row.test_start_utc,
                    "test_end_utc": row.test_end_utc,
                    "bout_id": row.bout_id,
                    "event_id": row.event_id,
                    "bout_datetime_utc": row.bout_datetime_utc,
                    "fighter_id": row.fighter_id,
                    "opponent_id": row.opponent_id,
                    "label": f"{row.label:.6f}",
                    "predicted_probability": f"{row.predicted_probability:.8f}",
                    "feature_cutoff_utc": row.feature_cutoff_utc,
                }
            )


def _write_comparison_report(
    *,
    output_path: Path,
    model_run_id: str,
    window_type: WindowType,
    fold_count: int,
    scenario_metrics: Sequence[AblationScenarioMetrics],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranked = sorted(scenario_metrics, key=lambda row: row.log_loss)
    rank_by_id = {row.config.scenario_id: index + 1 for index, row in enumerate(ranked)}

    lines: list[str] = [
        "ablation_comparison_report",
        f"model_run_id={model_run_id}",
        f"window_type={window_type}",
        f"fold_count={fold_count}",
        f"scenarios={len(scenario_metrics)}",
        "scenario_metrics",
    ]
    for metrics in ranked:
        lines.append(
            f"rank={rank_by_id[metrics.config.scenario_id]} "
            f"scenario_id={metrics.config.scenario_id} "
            f"groups={','.join(metrics.config.active_groups())} "
            f"interactions={int(metrics.config.include_matchup_interactions)} "
            f"predictions={metrics.predictions_count} "
            f"log_loss={metrics.log_loss:.8f} "
            f"brier_score={metrics.brier_score:.8f} "
            f"hit_rate={metrics.hit_rate:.8f} "
            f"calibration_ece={metrics.calibration_ece:.8f}"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_contribution_report(
    *,
    output_path: Path,
    model_run_id: str,
    scenario_metrics: Sequence[AblationScenarioMetrics],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    by_id = {row.config.scenario_id: row for row in scenario_metrics}

    lines: list[str] = [
        "ablation_contribution_report",
        f"model_run_id={model_run_id}",
        "contribution_summary",
    ]

    def append_delta(name: str, baseline_id: str, augmented_id: str) -> None:
        baseline = by_id.get(baseline_id)
        augmented = by_id.get(augmented_id)
        if baseline is None or augmented is None:
            lines.append(
                f"contribution name={name} status=unavailable "
                f"baseline={baseline_id} augmented={augmented_id}"
            )
            return

        lines.append(
            f"contribution name={name} "
            f"baseline={baseline_id} augmented={augmented_id} "
            f"log_loss_improvement={(baseline.log_loss - augmented.log_loss):.8f} "
            f"brier_improvement={(baseline.brier_score - augmented.brier_score):.8f} "
            f"hit_rate_delta={(augmented.hit_rate - baseline.hit_rate):.8f} "
            f"calibration_ece_improvement={(baseline.calibration_ece - augmented.calibration_ece):.8f}"
        )

    append_delta(
        "features_given_elo_no_interactions",
        "elo_only_no_interactions",
        "features_elo_no_interactions",
    )
    append_delta(
        "elo_given_features_no_interactions",
        "features_only_no_interactions",
        "features_elo_no_interactions",
    )
    append_delta(
        "market_given_features_elo_no_interactions",
        "features_elo_no_interactions",
        "features_elo_market_no_interactions",
    )
    append_delta(
        "features_given_elo_with_interactions",
        "elo_only_with_interactions",
        "features_elo_with_interactions",
    )
    append_delta(
        "elo_given_features_with_interactions",
        "features_only_with_interactions",
        "features_elo_with_interactions",
    )
    append_delta(
        "market_given_features_elo_with_interactions",
        "features_elo_with_interactions",
        "features_elo_market_with_interactions",
    )
    append_delta(
        "interaction_terms_given_features_elo_market",
        "features_elo_market_no_interactions",
        "features_elo_market_with_interactions",
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_feature_rows(path: Path) -> dict[tuple[str, str], dict[str, float | None]]:
    by_key: dict[tuple[str, str], dict[str, float | None]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            bout_id = str(row.get("bout_id", "")).strip()
            fighter_id = str(row.get("fighter_id", "")).strip()
            if bout_id == "" or fighter_id == "":
                continue
            by_key[(bout_id, fighter_id)] = {
                name: _as_optional_float(row.get(name)) for name in TABULAR_FEATURE_COLUMNS
            }
    return by_key


def _load_elo_rows(path: Path) -> dict[tuple[str, str], float]:
    by_key: dict[tuple[str, str], float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            bout_id = str(row.get("bout_id", "")).strip()
            fighter_id = str(row.get("fighter_id", "")).strip()
            if bout_id == "" or fighter_id == "":
                continue
            pre_elo = _as_optional_float(row.get("pre_overall_elo"))
            if pre_elo is None:
                continue
            by_key[(bout_id, fighter_id)] = pre_elo
    return by_key


def _load_bout_rows(*, connection: sqlite3.Connection, promotion: str) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT
            b.bout_id,
            b.event_id,
            COALESCE(b.bout_order, 0) AS bout_order,
            COALESCE(b.bout_start_time_utc, e.event_date_utc) AS bout_datetime_utc,
            b.fighter_red_id,
            b.fighter_blue_id,
            b.winner_fighter_id
        FROM bouts AS b
        JOIN events AS e ON e.event_id = b.event_id
        WHERE UPPER(e.promotion) = UPPER(?)
        ORDER BY
            COALESCE(b.bout_start_time_utc, e.event_date_utc) ASC,
            e.event_date_utc ASC,
            COALESCE(b.bout_order, 0) ASC,
            b.bout_id ASC
        """,
        (promotion,),
    ).fetchall()


def _load_market_implied_probabilities(
    *,
    connection: sqlite3.Connection,
    promotion: str,
) -> dict[tuple[str, str], float]:
    rows = connection.execute(
        """
        SELECT
            b.bout_id,
            COALESCE(b.bout_start_time_utc, e.event_date_utc) AS bout_datetime_utc,
            m.selection_fighter_id,
            m.implied_probability,
            m.market_timestamp_utc
        FROM markets AS m
        JOIN bouts AS b ON b.bout_id = m.bout_id
        JOIN events AS e ON e.event_id = b.event_id
        WHERE UPPER(e.promotion) = UPPER(?)
          AND m.selection_fighter_id IS NOT NULL
          AND m.implied_probability IS NOT NULL
        ORDER BY
            b.bout_id ASC,
            m.selection_fighter_id ASC,
            m.market_timestamp_utc ASC,
            m.market_id ASC
        """,
        (promotion,),
    ).fetchall()

    chosen: dict[tuple[str, str], tuple[str, list[float]]] = {}
    for row in rows:
        bout_id = str(row["bout_id"])
        fighter_id = str(row["selection_fighter_id"])
        bout_datetime = str(row["bout_datetime_utc"])
        timestamp = str(row["market_timestamp_utc"])
        implied = _as_optional_float(row["implied_probability"])
        if implied is None:
            continue
        if timestamp > bout_datetime:
            continue
        key = (bout_id, fighter_id)
        existing = chosen.get(key)
        if existing is None or timestamp > existing[0]:
            chosen[key] = (timestamp, [implied])
            continue
        if timestamp == existing[0]:
            existing[1].append(implied)

    return {
        key: sum(values) / float(len(values))
        for key, (_, values) in chosen.items()
        if len(values) > 0
    }


def _build_examples(
    *,
    bout_rows: Sequence[sqlite3.Row],
    features_by_key: dict[tuple[str, str], dict[str, float | None]],
    elo_by_key: dict[tuple[str, str], float],
    market_by_key: dict[tuple[str, str], float],
) -> list[_BoutExample]:
    examples: list[_BoutExample] = []
    for row in bout_rows:
        bout_id = str(row["bout_id"])
        event_id = str(row["event_id"])
        bout_datetime_utc = str(row["bout_datetime_utc"])
        bout_order = int(row["bout_order"])
        red_id = str(row["fighter_red_id"])
        blue_id = str(row["fighter_blue_id"])
        winner_id = _clean_text(row["winner_fighter_id"])
        red_label = _resolve_label(winner_id=winner_id, fighter_id=red_id, opponent_id=blue_id)
        blue_label = _resolve_label(winner_id=winner_id, fighter_id=blue_id, opponent_id=red_id)

        examples.append(
            _BoutExample(
                bout_id=bout_id,
                event_id=event_id,
                bout_datetime_utc=bout_datetime_utc,
                bout_order=bout_order,
                fighter_id=red_id,
                opponent_id=blue_id,
                label=red_label,
                feature_cutoff_utc=bout_datetime_utc,
                pre_fight_features=features_by_key.get((bout_id, red_id), {}),
                pre_overall_elo=elo_by_key.get((bout_id, red_id)),
                opponent_pre_overall_elo=elo_by_key.get((bout_id, blue_id)),
                market_implied_probability=market_by_key.get((bout_id, red_id)),
            )
        )
        examples.append(
            _BoutExample(
                bout_id=bout_id,
                event_id=event_id,
                bout_datetime_utc=bout_datetime_utc,
                bout_order=bout_order,
                fighter_id=blue_id,
                opponent_id=red_id,
                label=blue_label,
                feature_cutoff_utc=bout_datetime_utc,
                pre_fight_features=features_by_key.get((bout_id, blue_id), {}),
                pre_overall_elo=elo_by_key.get((bout_id, blue_id)),
                opponent_pre_overall_elo=elo_by_key.get((bout_id, red_id)),
                market_implied_probability=market_by_key.get((bout_id, blue_id)),
            )
        )

    return examples


def _build_timeline(*, examples: Sequence[_BoutExample]) -> list[_BoutTimelineEntry]:
    by_bout: dict[str, tuple[str, int]] = {}
    for row in examples:
        if row.label is None:
            continue
        existing = by_bout.get(row.bout_id)
        candidate = (row.bout_datetime_utc, row.bout_order)
        if existing is None or candidate < existing:
            by_bout[row.bout_id] = candidate
    timeline = [
        _BoutTimelineEntry(
            bout_id=bout_id,
            bout_datetime_utc=bout_datetime_utc,
            bout_order=bout_order,
        )
        for bout_id, (bout_datetime_utc, bout_order) in by_bout.items()
    ]
    timeline.sort(key=lambda row: (_parse_utc(row.bout_datetime_utc), row.bout_order, row.bout_id))
    return timeline


def _resolve_label(*, winner_id: str | None, fighter_id: str, opponent_id: str) -> float | None:
    if winner_id is None:
        return None
    if winner_id == fighter_id:
        return 1.0
    if winner_id == opponent_id:
        return 0.0
    return None


@dataclass(frozen=True)
class _CalibrationSummary:
    expected_calibration_error: float


def _build_calibration_summary(
    *,
    labels: Sequence[float],
    probabilities: Sequence[float],
    num_bins: int,
) -> _CalibrationSummary:
    if len(labels) == 0 or len(probabilities) == 0:
        raise ValueError("Calibration summary requires at least one prediction.")
    if len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities length mismatch.")

    bucket_probs: list[list[float]] = [[] for _ in range(num_bins)]
    bucket_labels: list[list[float]] = [[] for _ in range(num_bins)]
    for label, probability in zip(labels, probabilities):
        p = _clamp_probability(probability)
        index = min(int(p * num_bins), num_bins - 1)
        bucket_probs[index].append(p)
        bucket_labels[index].append(label)

    ece = 0.0
    total_count = len(probabilities)
    for index in range(num_bins):
        probs = bucket_probs[index]
        labels_for_bin = bucket_labels[index]
        count = len(probs)
        if count == 0:
            continue
        avg_probability = sum(probs) / float(count)
        observed_rate = sum(labels_for_bin) / float(count)
        abs_error = abs(avg_probability - observed_rate)
        ece += (count / float(total_count)) * abs_error
    return _CalibrationSummary(expected_calibration_error=ece)


def _log_loss(labels: Sequence[float], probabilities: Sequence[float]) -> float:
    if len(labels) == 0 or len(probabilities) == 0:
        raise ValueError("log_loss requires at least one prediction.")
    if len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities length mismatch.")
    total = 0.0
    for label, probability in zip(labels, probabilities):
        p = _clamp_probability(probability)
        total += -(label * math.log(p) + ((1.0 - label) * math.log(1.0 - p)))
    return total / float(len(labels))


def _brier_score(labels: Sequence[float], probabilities: Sequence[float]) -> float:
    if len(labels) == 0 or len(probabilities) == 0:
        raise ValueError("brier_score requires at least one prediction.")
    if len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities length mismatch.")
    total = 0.0
    for label, probability in zip(labels, probabilities):
        p = _clamp_probability(probability)
        total += (p - label) ** 2
    return total / float(len(labels))


def _hit_rate(labels: Sequence[float], probabilities: Sequence[float]) -> float:
    if len(labels) == 0 or len(probabilities) == 0:
        raise ValueError("hit_rate requires at least one prediction.")
    if len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities length mismatch.")
    hits = 0
    for label, probability in zip(labels, probabilities):
        predicted = 1.0 if probability >= 0.5 else 0.0
        if predicted == label:
            hits += 1
    return hits / float(len(labels))


def _as_optional_float(value: object | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _clean_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text != "" else None


def _clamp_probability(value: float) -> float:
    if value < 1e-6:
        return 1e-6
    if value > 1.0 - 1e-6:
        return 1.0 - 1e-6
    return value


def _sigmoid(z: float) -> float:
    if z >= 0:
        exp_value = math.exp(-z)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(z)
    return exp_value / (1.0 + exp_value)


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _build_model_run_id() -> str:
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return f"ablation_{timestamp}"
