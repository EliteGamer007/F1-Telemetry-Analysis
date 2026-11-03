from copy import deepcopy
from race_strategy.report import build_report, validate_report
from race_strategy.pace import fit_prior, fit_weekend
from race_strategy.planner import EventPolicy
from tests.test_pace import bundle, training_event


def report():
    b = bundle() | dict(era="2026+", date="2026-03-29", source_hash="fixture", missing_sessions=[], metadata={})
    model = fit_weekend(b, fit_prior([training_event()], b))
    return build_report(b, model, EventPolicy(laps=12, min_stint=2))


def test_report_schema_provenance_and_prepared_scenarios():
    result=report()
    validate_report(result)
    assert result["manifest"]["kind"] == "historical reconstruction"
    assert result["manifest"]["training_events"] == ["2026-1"]
    assert len(result["drivers"]["NOR"]["scenarios"]) == 9
    assert result["drivers"]["NOR"]["curves"]["HARD"][0]["p10"] < result["drivers"]["NOR"]["curves"]["HARD"][0]["p90"]


def test_report_missing_evidence_never_invents_a_plan():
    b=bundle() | dict(era="2026+", date="2026-03-29", source_hash="fixture", missing_sessions=[], metadata={})
    b["laps"]=[r for r in b["laps"] if r["session"]=="Q"]
    model=fit_weekend(b, fit_prior([], b))
    result=build_report(b, model, EventPolicy(laps=12))
    assert result["drivers"]["NOR"]["scenarios"]["base:base"] == []
    assert result["drivers"]["NOR"]["status"] == "unavailable"
