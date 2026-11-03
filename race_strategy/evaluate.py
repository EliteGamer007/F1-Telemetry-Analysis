"""Read-only evaluation. These labels NEVER flow back into a frozen report."""
from collections import defaultdict
import numpy as np
from .features import clean_laps
from .pace import predict_lap


def merge_observations(existing, incoming):
    observations={}
    for row in [*existing,*incoming]:
        key=tuple(row[k] for k in ("year","round","session","driver","lap"))
        observations[key]=row
    return list(observations.values())


def evaluate_reports(reports, heldout_laps):
    lookup={(r["year"],r["round"]):r for r in reports}
    errors=[]; coverage=[]; widths=[]; records=[]; prior_errors=[]; practice_errors=[]
    slices=defaultdict(list)
    eligible=clean_laps([r for r in merge_observations([],heldout_laps) if r["session"]=="R" and (r["year"],r["round"]) in lookup])
    for lap in eligible:
        report=lookup[lap["year"],lap["round"]]; m=report["model"]
        if lap["driver"] not in m["drivers"]:
            continue
        pred=predict_lap(m,lap["driver"],lap["compound"],lap["tyre_age"],lap["lap"])
        if pred is None:
            continue
        error=pred["median"]-lap["seconds"]
        errors.append(abs(error)); coverage.append(pred["p10"]<=lap["seconds"]<=pred["p90"])
        widths.append(pred["p90"]-pred["p10"])
        # Historical effects, event scale from practice, no driver-specific practice shrinkage.
        prior=m["prior"]; coef=prior["coefficients"]; ci=("SOFT","MEDIUM","HARD").index(lap["compound"])
        hist=m["event_practice_base"]+prior["race_practice_offset"]+coef[ci]*lap["tyre_age"]+coef[3]*lap["lap"]+(coef[4+ci] if ci<2 else 0.)
        prior_errors.append(abs(hist-lap["seconds"]))
        practice_errors.append(abs(m["event_practice_base"]-lap["seconds"]))
        slices[f"{lap['year']}-{lap['round']}:{lap['compound']}"] .append(abs(error))
        records.append({k:lap[k] for k in ("year","round","session","driver","lap","stint","compound","tyre_age","seconds")} |
                       dict(predicted=pred["median"], error=round(error,4), model_hash=report["manifest"]["model_hash"],
                            revision=lap.get("revision","cached-source-v1"),cutoff=report["manifest"]["cutoff"]))
    # Five-lap contiguous windows only, never concatenate separated clean laps.
    groups=defaultdict(list)
    for record in records: groups[record["year"],record["round"],record["driver"],record["stint"]].append(record)
    stint_errors=[]
    for rows in groups.values():
        rows.sort(key=lambda r:r["lap"])
        for i in range(0,len(rows)-4,5):
            window=rows[i:i+5]
            if window[-1]["lap"]-window[0]["lap"]==4:
                stint_errors.append(abs(sum(r["error"] for r in window))/5)
    def avg(values): return round(float(np.mean(values)),3) if values else None
    return dict(eligible_laps=len(eligible),predicted_laps=len(errors),unsupported_laps=len(eligible)-len(errors),
                mae_seconds=avg(errors), historical_effects_baseline_mae=avg(prior_errors), practice_baseline_mae=avg(practice_errors),
                interval_coverage=avg(coverage),mean_interval_width=avg(widths), short_stint_mae=avg(stint_errors),
                short_stint_windows=len(stint_errors),
                slices={k:dict(laps=len(v),mae_seconds=avg(v)) for k,v in slices.items()},
                records=records, protocol="Frozen chronological forecasts, observed dry green accurate laps; pilot only, no strategy causal claims")
