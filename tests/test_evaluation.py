from copy import deepcopy
from race_strategy.evaluate import evaluate_reports, merge_observations
from tests.test_report import report
from tests.test_weekend import lap


def test_evaluation_cannot_mutate_frozen_forecast_and_counts_unknowns():
    frozen=report(); before=deepcopy(frozen)
    rows=[lap(year=2026, round=3, session="R"), lap(year=2026, round=3, session="R", driver="XYZ")]
    result=evaluate_reports([frozen], rows)
    assert result["eligible_laps"] == 2
    assert result["predicted_laps"] == 1
    assert result["unsupported_laps"] == 1
    rows[0]["seconds"]+=20
    assert evaluate_reports([frozen], rows)["mae_seconds"] != result["mae_seconds"]
    assert frozen == before


def test_replaying_twice_does_not_duplicate_observations():
    row=lap(year=2026, round=3, session="R", revision="a")
    assert len(merge_observations([row], [row])) == 1
    corrected=row | dict(seconds=91., revision="b")
    assert merge_observations([row], [corrected]) == [corrected]
