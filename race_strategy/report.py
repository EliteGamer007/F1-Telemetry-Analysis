from dataclasses import asdict
from datetime import datetime, timezone
import math
from .pace import predict_lap
from .features import COMPOUNDS
from .planner import rank_strategies
from .weekend import validate_bundle, digest

DEGRADATION = {"low":.65, "base":1., "high":1.35}
PIT_LOSS = {"low":-3., "base":0., "high":3.}


def build_report(bundle, model, policy, seed=42):
    validate_bundle(bundle)
    drivers={}
    for info in bundle["drivers"]:
        code=info["code"]; d=model["drivers"][code]
        scenarios={f"{deg}:{pit}":rank_strategies(model,code,policy,seed,scale,loss)
                   for deg,scale in DEGRADATION.items() for pit,loss in PIT_LOSS.items()}
        curves={c:[dict(age=age, **prediction) for age in range(1,min(35,policy.laps)+1)
                   if (prediction:=predict_lap(model,code,c,age,age)) is not None] for c in COMPOUNDS}
        drivers[code]=info | dict(status=d["status"], practice_samples=d["practice_samples"],
                                  compounds=d["compounds"], qualifying_delta=d["qualifying_delta"],
                                  scenarios=scenarios, curves=curves,
                                  evidence=[f"{d['practice_samples']} usable laps in runs of at least five timed laps",
                                            f"Historical model: {len(model['prior']['training_events'])} earlier same-era events",
                                            "Practice programmes, fuel loads and traffic remain unknown"],
                                  stints=[s for s in model["stints"] if s["driver"]==code])
    result=dict(schema_version=1, year=bundle["year"], round=bundle["round"], event=bundle["event"],
                location=bundle["location"], date=bundle["date"], support=model["support"],
                manifest=dict(kind="historical reconstruction", cutoff=bundle["cutoff"],
                              cutoff_method="Scheduled session start + observed duration + 15-minute buffer",
                              generated_at=datetime.now(timezone.utc).isoformat(), era=bundle["era"],
                              model_version=model["version"], source_hash=bundle["source_hash"],
                              model_hash=digest(model), training_hash=model["prior"]["source_hash"],
                              training_events=model["prior"]["training_events"], metadata=bundle["metadata"]),
                coverage=dict(sessions=bundle["sessions"],missing=bundle["missing_sessions"],
                              training_laps=model["prior"]["training_laps"],
                              calibration_events=model["prior"]["correction_events"]),
                policy=asdict(policy), assumptions=model["assumptions"]+[policy.inventory_source,policy.pit_source],
                uncertainty=model["prior"]["calibration"], drivers=drivers,
                model=model, scenario_options=dict(degradation=DEGRADATION,pit_loss=PIT_LOSS),
                provenance="FastF1 timing observations from the owner's local cache; independent educational project")
    if model["prior"]["correction_events"]==0:
        result["assumptions"].append("No historical practice-to-race calibration available: absolute race times are scenario illustrations")
    validate_report(result)
    return result


def validate_report(report):
    if report.get("schema_version")!=1 or not report.get("drivers") or not report.get("manifest",{}).get("source_hash"):
        raise ValueError("Incomplete report")
    from .regimes import utc
    utc(report["manifest"]["cutoff"])
    for driver in report["drivers"].values():
        if set(driver["scenarios"]) != {f"{d}:{p}" for d in DEGRADATION for p in PIT_LOSS}:
            raise ValueError("Incomplete prepared scenarios")
    def finite(value):
        if isinstance(value,float) and not math.isfinite(value):
            raise ValueError("Report contains non-finite values")
        if isinstance(value,dict):
            for v in value.values(): finite(v)
        elif isinstance(value,list):
            for v in value: finite(v)
    finite(report)
    return report
