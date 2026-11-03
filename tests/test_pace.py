import copy
import numpy as np
import pytest
from race_strategy.pace import fit_prior, fit_weekend, predict_lap
from tests.test_weekend import lap


def training_event():
    rows = []
    for driver, delta in [("NOR", -0.4), ("LEC", 0.4)]:
        for code in ["FP2", "R", "Q"]:
            for stint, compound in enumerate(["SOFT", "MEDIUM", "HARD"]):
                for age in range(1, 11):
                    n = stint * 10 + age
                    seconds = 90 + delta + .08 * age - .02 * n + stint * .5
                    rows.append(lap(driver=driver, session=code, stint=stint+1, compound=compound,
                                    lap=n, tyre_age=age, seconds=seconds+(2 if code=="FP2" else -3 if code=="Q" else 0)))
    return dict(year=2026, round=1, date="2026-03-08T12:00:00Z", location="Melbourne", laps=rows)


def bundle():
    event = training_event()
    return dict(year=2026, round=3, cutoff="2026-03-28T09:00:00Z", event="Japanese Grand Prix",
                location="Suzuka", sessions=[dict(code="Q", end="2026-03-28T08:00:00Z")],
                laps=[r for r in event["laps"] if r["session"] != "R"],
                drivers=[dict(code="NOR"), dict(code="LEC")])


def test_model_reproducible_and_uncertainty_does_not_disappear():
    target = bundle()
    p = fit_prior([training_event()], target)
    m = fit_weekend(target, p)
    assert m == fit_weekend(target, p)
    prediction = predict_lap(m, "NOR", "MEDIUM", 10, 25)
    assert 75 < prediction["median"] < 110
    assert prediction["p10"] < prediction["median"] < prediction["p90"]
    assert m["drivers"]["NOR"]["pace_sigma"] >= .8
    assert p["training_events"] == ["2026-1"]


def test_missing_compound_is_prior_not_weekend_measurement():
    target = bundle()
    target["laps"] = [r for r in target["laps"] if r["compound"] != "HARD"]
    model = fit_weekend(target, fit_prior([training_event()], target))
    assert model["drivers"]["NOR"]["compounds"]["HARD"]["practice_samples"] == 0
    assert model["drivers"]["NOR"]["compounds"]["HARD"]["source"] == "historical prior"


def test_no_history_has_explicit_scenario_only_status():
    model = fit_weekend(bundle(), fit_prior([], bundle()))
    assert model["support"] == "scenario-only"
    assert model["prior"]["training_laps"] == 0


def test_future_training_rejected_in_model_even_if_acquisition_bypassed():
    future = copy.deepcopy(training_event())
    future["date"] = "2026-04-10T12:00:00Z"
    with pytest.raises(ValueError, match="chronological"):
        fit_prior([future], bundle())
