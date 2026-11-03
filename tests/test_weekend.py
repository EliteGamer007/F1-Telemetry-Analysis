import pytest
from race_strategy.regimes import era_for, eligible_events
from race_strategy.features import clean_laps, summarize_stints
from race_strategy.weekend import validate_bundle


def lap(**updates):
    row = dict(driver="NOR", session="FP2", stint=1, compound="MEDIUM", lap=5,
               tyre_age=3, seconds=90., accurate=True, deleted=False, pit_in=False,
               pit_out=False, track_status="1", fresh=True)
    return row | updates


def test_regime_and_chronology_are_hard_boundaries():
    events = [dict(year=2025, round=24, date="2025-12-07T12:00:00Z"),
              dict(year=2026, round=1, date="2026-03-08T12:00:00Z"),
              dict(year=2026, round=8, date="2026-06-07T12:00:00Z")]
    assert era_for(2026) != era_for(2025)
    assert eligible_events(events, 2026, "2026-05-01T00:00:00Z") == [events[1]]


def test_clean_laps_exclude_non_racing_and_unknown_quality():
    rows = [lap(), lap(pit_in=True), lap(track_status="14"), lap(deleted=True),
            lap(accurate=None), lap(compound="INTERMEDIATE"), lap(seconds=None)]
    assert clean_laps(rows) == [rows[0]]


def test_same_compound_sets_are_distinct_and_short_runs_flagged():
    rows = [lap(lap=i, tyre_age=i, stint=1) for i in range(1, 7)]
    rows += [lap(lap=i+10, tyre_age=i, stint=2) for i in range(1, 4)]
    stints = summarize_stints(rows)
    assert len(stints) == 2
    assert [s["samples"] for s in stints] == [6, 3]
    assert [s["long_run"] for s in stints] == [True, False]


def test_unknown_deleted_status_stays_unknown_for_evidence_label():
    row = lap(deleted=None)
    assert clean_laps([row])[0]["deleted"] is None


def test_target_race_cannot_enter_bundle():
    bundle = dict(year=2026, round=4, cutoff="2026-04-01T12:00:00Z",
                  sessions=[dict(code="R", end="2026-04-01T10:00:00Z")], laps=[])
    with pytest.raises(ValueError, match="practice or qualifying"):
        validate_bundle(bundle)


def test_future_session_rejected():
    bundle = dict(year=2026, round=4, cutoff="2026-04-01T12:00:00Z",
                  sessions=[dict(code="Q", end="2026-04-02T10:00:00Z")], laps=[])
    with pytest.raises(ValueError, match="cutoff"):
        validate_bundle(bundle)
