"""Chronology-safe baseline model training and prediction pipeline."""

from __future__ import annotations

import csv
import hashlib
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from features import generate_pre_fight_features
from ratings import EloConfig, generate_elo_ratings_history

TABULAR_FEATURE_COLUMNS: tuple[str, ...] = (
    "fighter_age_years",
    "height_diff_cm",
    "reach_diff_cm",
    "days_since_last_fight",
    "prior_fight_count",
    "recent_form_win_rate_l3",
    "recent_form_win_rate_l5",
    "opponent_strength_avg_l3",
    "opponent_strength_avg_l5",
    "finish_rate_all",
    "decision_rate_all",
    "sig_strikes_landed_per_min_all",
    "sig_striking_accuracy_all",
    "sig_striking_defense_all",
    "takedown_attempts_per_15m_all",
    "takedown_accuracy_all",
    "takedown_defense_all",
    "submission_attempts_per_15m_all",
    "avg_fight_duration_seconds_all",
    "rematch_flag",
    "weight_class_experience",
)


@dataclass(frozen=True)
class BoutExample:
    """Single fighter-side bout example used for model training/scoring."""

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
class TrainTestSplit:
    """Chronological bout split boundary metadata."""

    train_bout_ids: set[str]
    test_bout_ids: set[str]
    train_start_utc: str
    train_end_utc: str
    test_start_utc: str
    test_end_utc: str


@dataclass(frozen=True)
class CoverageMetric:
    """Model-level scoring coverage summary."""

    model_name: str
    scored_rows: int
    eligible_rows: int
    coverage_rate: float


@dataclass(frozen=True)
class BaselineRunResult:
    """Outcome summary for baseline modeling run."""

    model_run_id: str
    predictions_path: Path
    report_path: Path
    prediction_count: int
    scored_bout_count: int
    train_test_split: TrainTestSplit
    coverage_metrics: tuple[CoverageMetric, ...]
    feature_coverage: dict[str, float]


@dataclass(frozen=True)
class ScoredPrediction:
    """Single scored prediction row for persistence."""

    model_name: str
    model_version: str
    model_run_id: str
    bout_id: str
    event_id: str
    bout_datetime_utc: str
    fighter_id: str
    opponent_id: str
    split: str
    label: float | None
    predicted_probability: float
    feature_cutoff_utc: str


@dataclass
class LogisticModel:
    """Minimal logistic regression model with feature scaling."""

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
        return _sigmoid(z)


@dataclass(frozen=True)
class TreeStump:
    """Simple one-level decision tree classifier baseline."""

    feature_index: int
    threshold: float
    left_probability: float
    right_probability: float

    def predict_probability(self, feature_values: Sequence[float]) -> float:
        value = feature_values[self.feature_index]
        if value <= self.threshold:
            return self.left_probability
        return self.right_probability


def run_baseline_models(
    *,
    db_path: Path,
    promotion: str,
    train_fraction: float,
    model_version: str,
    features_output_path: Path,
    ratings_output_path: Path,
    predictions_output_path: Path,
    report_output_path: Path,
    model_run_id: str | None = None,
) -> BaselineRunResult:
    """Train chronology-safe baseline models and persist out-of-sample predictions."""

    if train_fraction <= 0.0 or train_fraction >= 1.0:
        raise ValueError("train_fraction must be between 0 and 1 (exclusive).")

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
    split = _chronological_train_test_split(examples=examples, train_fraction=train_fraction)

    train_rows = [row for row in examples if row.bout_id in split.train_bout_ids and row.label is not None]
    test_rows = [row for row in examples if row.bout_id in split.test_bout_ids]

    elo_train = [row for row in train_rows if row.elo_diff is not None]
    elo_model = _fit_logistic_model(
        features=[[float(row.elo_diff)] for row in elo_train],
        labels=[float(row.label) for row in elo_train],
        feature_names=("elo_diff",),
    )

    tabular_train_rows = [row for row in train_rows if _has_any_tabular_value(row)]
    tabular_train_vectors = [_tabular_vector(row, TABULAR_FEATURE_COLUMNS) for row in tabular_train_rows]
    tabular_model = _fit_logistic_model(
        features=tabular_train_vectors,
        labels=[float(row.label) for row in tabular_train_rows],
        feature_names=TABULAR_FEATURE_COLUMNS,
    )

    tree_model = _fit_tree_stump(
        features=tabular_train_vectors,
        labels=[float(row.label) for row in tabular_train_rows],
    )

    resolved_model_run_id = model_run_id or _build_model_run_id()
    predictions: list[ScoredPrediction] = []
    model_coverage_counts: dict[str, int] = {
        "market_implied_probability": 0,
        "elo_logistic": 0,
        "tabular_logistic": 0,
        "tabular_tree_stump": 0,
    }
    eligible_rows = len(test_rows)

    for row in test_rows:
        market_prob = row.market_implied_probability
        if market_prob is not None:
            predictions.append(
                _to_prediction(
                    row=row,
                    model_name="market_implied_probability",
                    model_version=model_version,
                    model_run_id=resolved_model_run_id,
                    probability=market_prob,
                )
            )
            model_coverage_counts["market_implied_probability"] += 1

        if row.elo_diff is not None:
            elo_prob = elo_model.predict_probability([float(row.elo_diff)])
            predictions.append(
                _to_prediction(
                    row=row,
                    model_name="elo_logistic",
                    model_version=model_version,
                    model_run_id=resolved_model_run_id,
                    probability=elo_prob,
                )
            )
            model_coverage_counts["elo_logistic"] += 1

        if _has_any_tabular_value(row):
            vector = _tabular_vector(row, TABULAR_FEATURE_COLUMNS)
            tabular_prob = tabular_model.predict_probability(vector)
            tree_prob = tree_model.predict_probability(_impute_with_means(vector, tabular_model.means))
            predictions.append(
                _to_prediction(
                    row=row,
                    model_name="tabular_logistic",
                    model_version=model_version,
                    model_run_id=resolved_model_run_id,
                    probability=tabular_prob,
                )
            )
            predictions.append(
                _to_prediction(
                    row=row,
                    model_name="tabular_tree_stump",
                    model_version=model_version,
                    model_run_id=resolved_model_run_id,
                    probability=tree_prob,
                )
            )
            model_coverage_counts["tabular_logistic"] += 1
            model_coverage_counts["tabular_tree_stump"] += 1

    _persist_predictions_csv(output_path=predictions_output_path, predictions=predictions)
    _persist_predictions_db(db_path=db_path, predictions=predictions)

    feature_coverage = _compute_feature_coverage(rows=test_rows, feature_names=TABULAR_FEATURE_COLUMNS)
    coverage_metrics = tuple(
        CoverageMetric(
            model_name=model_name,
            scored_rows=scored,
            eligible_rows=eligible_rows,
            coverage_rate=0.0 if eligible_rows == 0 else scored / eligible_rows,
        )
        for model_name, scored in model_coverage_counts.items()
    )
    scored_bouts = {prediction.bout_id for prediction in predictions}
    _write_report(
        output_path=report_output_path,
        split=split,
        scored_bout_count=len(scored_bouts),
        coverage_metrics=coverage_metrics,
        feature_coverage=feature_coverage,
    )

    return BaselineRunResult(
        model_run_id=resolved_model_run_id,
        predictions_path=predictions_output_path,
        report_path=report_output_path,
        prediction_count=len(predictions),
        scored_bout_count=len(scored_bouts),
        train_test_split=split,
        coverage_metrics=coverage_metrics,
        feature_coverage=feature_coverage,
    )


def _load_feature_rows(path: Path) -> dict[tuple[str, str], dict[str, float | None]]:
    by_key: dict[tuple[str, str], dict[str, float | None]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            bout_id = row.get("bout_id", "").strip()
            fighter_id = row.get("fighter_id", "").strip()
            if bout_id == "" or fighter_id == "":
                continue
            features = {name: _as_optional_float(row.get(name)) for name in TABULAR_FEATURE_COLUMNS}
            by_key[(bout_id, fighter_id)] = features
    return by_key


def _load_elo_rows(path: Path) -> dict[tuple[str, str], float]:
    by_key: dict[tuple[str, str], float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            bout_id = row.get("bout_id", "").strip()
            fighter_id = row.get("fighter_id", "").strip()
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
            b.bout_order,
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
) -> list[BoutExample]:
    examples: list[BoutExample] = []
    for row in bout_rows:
        bout_id = str(row["bout_id"])
        event_id = str(row["event_id"])
        bout_datetime_utc = str(row["bout_datetime_utc"])
        bout_order = int(row["bout_order"]) if row["bout_order"] is not None else 0
        red_id = str(row["fighter_red_id"])
        blue_id = str(row["fighter_blue_id"])
        winner_id = _clean_text(row["winner_fighter_id"])

        red_label = _resolve_label(winner_id=winner_id, fighter_id=red_id, opponent_id=blue_id)
        blue_label = _resolve_label(winner_id=winner_id, fighter_id=blue_id, opponent_id=red_id)

        examples.append(
            BoutExample(
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
            BoutExample(
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


def _resolve_label(*, winner_id: str | None, fighter_id: str, opponent_id: str) -> float | None:
    if winner_id is None:
        return None
    if winner_id == fighter_id:
        return 1.0
    if winner_id == opponent_id:
        return 0.0
    return None


def _chronological_train_test_split(*, examples: Sequence[BoutExample], train_fraction: float) -> TrainTestSplit:
    bout_timeline: list[tuple[str, str, int]] = []
    seen_bout_ids: set[str] = set()
    for row in examples:
        if row.label is None:
            continue
        if row.bout_id in seen_bout_ids:
            continue
        seen_bout_ids.add(row.bout_id)
        bout_timeline.append((row.bout_datetime_utc, row.bout_order, row.bout_id))

    bout_timeline.sort(key=lambda item: (item[0], item[1], item[2]))
    if len(bout_timeline) < 2:
        raise ValueError("Need at least two completed bouts for chronological train/test split.")

    train_bout_count = int(len(bout_timeline) * train_fraction)
    train_bout_count = max(1, min(train_bout_count, len(bout_timeline) - 1))

    train = bout_timeline[:train_bout_count]
    test = bout_timeline[train_bout_count:]

    return TrainTestSplit(
        train_bout_ids={entry[2] for entry in train},
        test_bout_ids={entry[2] for entry in test},
        train_start_utc=train[0][0],
        train_end_utc=train[-1][0],
        test_start_utc=test[0][0],
        test_end_utc=test[-1][0],
    )


def _fit_logistic_model(
    *,
    features: Sequence[Sequence[float | None]],
    labels: Sequence[float],
    feature_names: Iterable[str],
    learning_rate: float = 0.05,
    iterations: int = 600,
    l2_penalty: float = 0.001,
) -> LogisticModel:
    if len(features) == 0:
        raise ValueError("Logistic model training requires at least one row.")
    if len(features) != len(labels):
        raise ValueError("Features and labels length mismatch.")

    feature_name_tuple = tuple(feature_names)
    dimension = len(feature_name_tuple)
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

    return LogisticModel(
        feature_names=feature_name_tuple,
        means=tuple(means),
        stds=tuple(stds),
        weights=tuple(weights),
        bias=bias,
    )


def _fit_tree_stump(*, features: Sequence[Sequence[float | None]], labels: Sequence[float]) -> TreeStump:
    if len(features) == 0 or len(labels) == 0:
        raise ValueError("Tree stump training requires non-empty data.")
    if len(features) != len(labels):
        raise ValueError("Features and labels length mismatch.")

    matrix = _impute_matrix(features)
    dimension = len(matrix[0])

    best_loss = float("inf")
    best_stump: TreeStump | None = None

    for feature_index in range(dimension):
        sorted_values = sorted({row[feature_index] for row in matrix})
        if len(sorted_values) == 1:
            threshold_candidates = [sorted_values[0]]
        else:
            threshold_candidates = [
                (sorted_values[idx] + sorted_values[idx + 1]) / 2.0
                for idx in range(len(sorted_values) - 1)
            ]

        for threshold in threshold_candidates:
            left_labels = [labels[i] for i, row in enumerate(matrix) if row[feature_index] <= threshold]
            right_labels = [labels[i] for i, row in enumerate(matrix) if row[feature_index] > threshold]
            if len(left_labels) == 0 or len(right_labels) == 0:
                continue
            left_prob = _smoothed_mean(left_labels)
            right_prob = _smoothed_mean(right_labels)
            loss = 0.0
            for row_index, row in enumerate(matrix):
                probability = left_prob if row[feature_index] <= threshold else right_prob
                y = labels[row_index]
                p = _clamp_probability(probability)
                loss += -(y * math.log(p) + ((1.0 - y) * math.log(1.0 - p)))
            if loss < best_loss:
                best_loss = loss
                best_stump = TreeStump(
                    feature_index=feature_index,
                    threshold=threshold,
                    left_probability=left_prob,
                    right_probability=right_prob,
                )

    if best_stump is not None:
        return best_stump

    base_prob = _smoothed_mean(labels)
    return TreeStump(feature_index=0, threshold=0.0, left_probability=base_prob, right_probability=base_prob)


def _smoothed_mean(values: Sequence[float], alpha: float = 1.0) -> float:
    positives = sum(values)
    total = float(len(values))
    return (positives + alpha) / (total + (2.0 * alpha))


def _impute_matrix(features: Sequence[Sequence[float | None]]) -> list[list[float]]:
    if len(features) == 0:
        return []
    dimension = len(features[0])
    means: list[float] = []
    for index in range(dimension):
        values = [row[index] for row in features if row[index] is not None]
        means.append(sum(values) / float(len(values)) if len(values) > 0 else 0.0)

    matrix: list[list[float]] = []
    for row in features:
        matrix.append([means[i] if row[i] is None else float(row[i]) for i in range(dimension)])
    return matrix


def _impute_with_means(values: Sequence[float | None], means: Sequence[float]) -> list[float]:
    return [means[i] if value is None else float(value) for i, value in enumerate(values)]


def _tabular_vector(example: BoutExample, columns: Sequence[str]) -> list[float | None]:
    return [example.pre_fight_features.get(column) for column in columns]


def _has_any_tabular_value(example: BoutExample) -> bool:
    for column in TABULAR_FEATURE_COLUMNS:
        if example.pre_fight_features.get(column) is not None:
            return True
    return False


def _to_prediction(
    *,
    row: BoutExample,
    model_name: str,
    model_version: str,
    model_run_id: str,
    probability: float,
) -> ScoredPrediction:
    return ScoredPrediction(
        model_name=model_name,
        model_version=model_version,
        model_run_id=model_run_id,
        bout_id=row.bout_id,
        event_id=row.event_id,
        bout_datetime_utc=row.bout_datetime_utc,
        fighter_id=row.fighter_id,
        opponent_id=row.opponent_id,
        split="test",
        label=row.label,
        predicted_probability=_clamp_probability(probability),
        feature_cutoff_utc=row.feature_cutoff_utc,
    )


def _persist_predictions_csv(*, output_path: Path, predictions: Sequence[ScoredPrediction]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "model_run_id",
        "model_name",
        "model_version",
        "bout_id",
        "event_id",
        "bout_datetime_utc",
        "fighter_id",
        "opponent_id",
        "split",
        "label",
        "predicted_probability",
        "feature_cutoff_utc",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for prediction in predictions:
            writer.writerow(
                {
                    "model_run_id": prediction.model_run_id,
                    "model_name": prediction.model_name,
                    "model_version": prediction.model_version,
                    "bout_id": prediction.bout_id,
                    "event_id": prediction.event_id,
                    "bout_datetime_utc": prediction.bout_datetime_utc,
                    "fighter_id": prediction.fighter_id,
                    "opponent_id": prediction.opponent_id,
                    "split": prediction.split,
                    "label": "" if prediction.label is None else f"{prediction.label:.6f}",
                    "predicted_probability": f"{prediction.predicted_probability:.8f}",
                    "feature_cutoff_utc": prediction.feature_cutoff_utc,
                }
            )


def _persist_predictions_db(*, db_path: Path, predictions: Sequence[ScoredPrediction]) -> None:
    now = _utc_now()
    records: list[tuple[object, ...]] = []
    for prediction in predictions:
        prediction_type = f"win_probability:{prediction.model_name}"
        source_record_id = (
            f"{prediction.model_run_id}:{prediction.model_name}:{prediction.bout_id}:{prediction.fighter_id}:{prediction_type}"
        )
        model_prediction_id = hashlib.sha256(source_record_id.encode("utf-8")).hexdigest()[:32]
        source_payload_sha = hashlib.sha256(
            (
                f"{prediction.model_name}|{prediction.model_version}|{prediction.bout_id}|"
                f"{prediction.fighter_id}|{prediction.predicted_probability:.8f}"
            ).encode("utf-8")
        ).hexdigest()
        records.append(
            (
                model_prediction_id,
                prediction.bout_id,
                prediction.fighter_id,
                prediction.model_name,
                prediction.model_version,
                prediction.model_run_id,
                prediction_type,
                prediction.predicted_probability,
                now,
                prediction.feature_cutoff_utc,
                "mma_elo_baselines",
                source_record_id,
                None,
                source_payload_sha,
                now,
                now,
            )
        )

    with sqlite3.connect(db_path) as connection:
        connection.executemany(
            """
            INSERT OR REPLACE INTO model_predictions (
                model_prediction_id,
                bout_id,
                target_fighter_id,
                model_name,
                model_version,
                model_run_id,
                prediction_type,
                predicted_probability,
                generated_at_utc,
                feature_cutoff_utc,
                source_system,
                source_record_id,
                source_updated_at_utc,
                source_payload_sha256,
                created_at_utc,
                updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        connection.commit()


def _compute_feature_coverage(*, rows: Sequence[BoutExample], feature_names: Iterable[str]) -> dict[str, float]:
    row_count = len(rows)
    coverage: dict[str, float] = {}
    for feature_name in feature_names:
        present = 0
        for row in rows:
            if row.pre_fight_features.get(feature_name) is not None:
                present += 1
        coverage[feature_name] = 0.0 if row_count == 0 else present / row_count
    return coverage


def _write_report(
    *,
    output_path: Path,
    split: TrainTestSplit,
    scored_bout_count: int,
    coverage_metrics: Sequence[CoverageMetric],
    feature_coverage: dict[str, float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "baseline_report",
        f"bouts_scored={scored_bout_count}",
        "train_test_boundaries",
        f"train_start_utc={split.train_start_utc}",
        f"train_end_utc={split.train_end_utc}",
        f"test_start_utc={split.test_start_utc}",
        f"test_end_utc={split.test_end_utc}",
        "model_coverage",
    ]
    for metric in coverage_metrics:
        lines.append(
            f"{metric.model_name}: scored_rows={metric.scored_rows} "
            f"eligible_rows={metric.eligible_rows} coverage={metric.coverage_rate:.6f}"
        )
    lines.append("feature_coverage")
    for feature_name in sorted(feature_coverage):
        lines.append(f"{feature_name}: coverage={feature_coverage[feature_name]:.6f}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _build_model_run_id() -> str:
    timestamp = _utc_now()
    digest = hashlib.sha256(timestamp.encode("utf-8")).hexdigest()[:10]
    return f"baselines_{timestamp}_{digest}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
