import pytest
from race_strategy.planner import EventPolicy, enumerate_plans, rank_strategies, validate_plan
from race_strategy.pace import fit_prior, fit_weekend
from tests.test_pace import bundle, training_event


def test_enumeration_full_distance_inventory_and_two_compounds():
    policy = EventPolicy(laps=8, pit_loss=20, inventory={"SOFT":1,"MEDIUM":1,"HARD":1}, min_stint=2)
    plans = enumerate_plans(policy)
    assert plans
    for p in plans:
        validate_plan(p, policy)
        assert sum(p["lengths"]) == 8
        assert len(set(p["compounds"])) >= 2
        assert len(p["compounds"]) == len(p["pits"])+1
        assert len(set(p["compounds"])) == len(p["compounds"])


def test_illegal_plan_rejected():
    with pytest.raises(ValueError):
        validate_plan(dict(compounds=["SOFT", "HARD"], pits=[0], lengths=[0, 8]), EventPolicy(laps=8))


def test_ranking_reproducible_distinct_and_paired():
    model = fit_weekend(bundle(), fit_prior([training_event()], bundle()))
    result = rank_strategies(model, "NOR", EventPolicy(laps=12, min_stint=2), seed=7)
    assert result == rank_strategies(model, "NOR", EventPolicy(laps=12, min_stint=2), seed=7)
    assert len({tuple(s["compounds"]) for s in result}) == 3
    assert result[0]["delta"] == 0
    assert result[0]["p10"] < result[0]["median"] < result[0]["p90"]
    assert all(s["p10_delta"] <= s["p90_delta"] for s in result)


def test_high_pitloss_favors_one_stop_with_zero_degradation():
    model = fit_weekend(bundle(), fit_prior([training_event()], bundle()))
    for c in model["drivers"]["NOR"]["compounds"].values():
        c["slope"] = 0
        c["offset"] = 0
    result = rank_strategies(model, "NOR", EventPolicy(laps=12, pit_loss=100, min_stint=2))
    assert len(result[0]["pits"]) == 1
