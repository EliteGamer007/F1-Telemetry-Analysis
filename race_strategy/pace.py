"""Small ridge baseline with explicit prior shrinkage and correlated uncertainty.

Age and race progression are confounded; estimates are empirical pace trends,
not measurements of tyre wear or fuel. All training events precede the cutoff.
"""
from collections import defaultdict
import numpy as np
from .features import COMPOUNDS, clean_laps, summarize_stints
from .regimes import eligible_events
from .weekend import validate_bundle, digest
from . import MODEL_VERSION

DEFAULT_SLOPES = np.array([.075, .05, .03])
DEFAULT_OFFSETS = np.array([-.6, -.3, 0.])


def _design(rows):
    return np.array([[r["tyre_age"] * (r["compound"] == c) for c in COMPOUNDS]
                     + [r["lap"], float(r["compound"] == "SOFT"), float(r["compound"] == "MEDIUM")]
                    for r in rows], dtype=float)


def fit_prior(events, target):
    validate_bundle(target)
    eligible = eligible_events(events, target["year"], target["cutoff"])
    if len(eligible) != len(events) or any((e["year"], e["round"]) == (target["year"], target["round"]) for e in events):
        raise ValueError("Training violates chronological same-era boundary")
    keys = [(e["year"], e["round"]) for e in events]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate training event")
    groups = defaultdict(list)
    for event in events:
        for row in clean_laps(event["laps"]):
            if row["session"] == "R":
                groups[event["year"], event["round"], row["driver"]].append(row)
    xs, ys = [], []
    for rows in groups.values():
        if len(rows) < 8:
            continue
        x = _design(rows)
        y = np.array([r["seconds"] for r in rows])
        # Remove driver/event intercept before fitting common empirical effects.
        xs.append(x - x.mean(axis=0))
        ys.append(y - y.mean())
    defaults = np.r_[DEFAULT_SLOPES, -.025, DEFAULT_OFFSETS[:2]]
    residual_sigma = 1.5
    coef = defaults.copy()
    if xs:
        x, y = np.vstack(xs), np.concatenate(ys)
        penalty = np.diag([100, 100, 100, 200, 80, 80])
        coef = np.linalg.solve(x.T @ x + penalty, x.T @ y + penalty @ defaults)
        # Conservative structural bounds; flags/model card expose this assumption.
        coef[:3] = np.clip(coef[:3], .005, .25)
        coef[3] = np.clip(coef[3], -.08, .02)
        coef[4:] = np.clip(coef[4:], -2., 1.)
        residual_sigma = max(.6, float(np.quantile(np.abs(y - x @ coef), .68)))
    corrections, qx, qy, event_offsets = [], [], [], []
    for event in events:
        race = [r for r in clean_laps(event["laps"]) if r["session"] == "R"]
        practice = [r for r in event["laps"] if r["session"].startswith("FP")]
        runs = [r for r in summarize_stints(practice) if r["long_run"]]
        def adjusted_run(r):
            i = COMPOUNDS.index(r["compound"])
            offset = coef[4+i] if i < 2 else 0.
            return r["median_seconds"] - coef[i]*(r["age_min"]+r["age_max"])/2 - offset
        if race and runs:
            normalized = [r["seconds"] - _design([r])[0] @ coef for r in race]
            corrections.append(float(np.median(normalized)-np.median([adjusted_run(r) for r in runs])))
        dpace, qpace = {}, {}
        for driver in {r["driver"] for r in race}:
            laps = [r for r in race if r["driver"] == driver]
            dpace[driver] = float(np.median([r["seconds"] - _design([r])[0] @ coef for r in laps]))
            qual = [r["seconds"] for r in clean_laps(event["laps"]) if r["driver"]==driver and r["session"]=="Q"]
            if qual:
                qpace[driver] = min(qual)
        if qpace:
            qmedian, rmedian = np.median(list(qpace.values())), np.median(list(dpace.values()))
            for driver in qpace:
                qx.append(qpace[driver]-qmedian)
                qy.append(dpace[driver]-rmedian)
            event_offsets.append(float(np.std(list(dpace.values()))))
    weight = float(np.clip(np.dot(qx, qy)/(np.dot(qx, qx)+10), 0, 1.2)) if qx else 0.
    support = {c: sum(r["compound"]==c for rows in groups.values() for r in rows) for c in COMPOUNDS}
    return dict(version=MODEL_VERSION, coefficients=[float(c) for c in coef],
                training_events=[f"{e['year']}-{e['round']}" for e in events],
                training_laps=sum(len(v) for v in groups.values()), compound_samples=support,
                source_hash=digest(events), cutoff=target["cutoff"],
                race_practice_offset=float(np.median(corrections)) if corrections else 0.,
                correction_events=len(corrections), qualifying_weight=weight,
                residual_sigma=residual_sigma,
                offset_sigma=max(.8, float(np.std(corrections))) if corrections else 2.,
                calibration="Uncalibrated 10–90% model range; not a guaranteed prediction interval")


def fit_weekend(bundle, prior):
    validate_bundle(bundle)
    coef = prior["coefficients"]
    runs = summarize_stints([r for r in bundle["laps"] if r["session"].startswith("FP")])
    longruns = [r for r in runs if r["long_run"]]
    clean = clean_laps(bundle["laps"])
    qualifying = {d["code"]: min([r["seconds"] for r in clean if r["session"]=="Q" and r["driver"]==d["code"]], default=None)
                  for d in bundle["drivers"]}
    qvalues = [v for v in qualifying.values() if v is not None]
    qmedian = float(np.median(qvalues)) if qvalues else None
    def adjusted(r):
        i = COMPOUNDS.index(r["compound"])
        return r["median_seconds"] - coef[i]*(r["age_min"]+r["age_max"])/2 - (coef[4+i] if i<2 else 0.)
    event_base = float(np.median([adjusted(r) for r in longruns])) if longruns else None
    drivers = {}
    for driver in bundle["drivers"]:
        code = driver["code"]
        own = [r for r in longruns if r["driver"]==code]
        samples = sum(r["samples"] for r in own)
        qdelta = qualifying[code]-qmedian if qualifying[code] is not None and qmedian is not None else 0.
        inferred = event_base + prior["race_practice_offset"] + prior["qualifying_weight"]*qdelta if event_base else None
        share = min(.65, samples/(samples+30))
        base = ((1-share)*inferred + share*(float(np.median([adjusted(r) for r in own]))+prior["race_practice_offset"])) if own and inferred else inferred
        compounds = {}
        for i, c in enumerate(COMPOUNDS):
            observed = [r for r in own if r["compound"]==c and r["trend"] is not None]
            n = sum(r["samples"] for r in observed)
            alpha = min(.5, n/(n+40))
            # Subtract historical race-progression effect only as an empirical correction.
            trend = float(np.median([r["trend"] for r in observed])) - coef[3] if observed else coef[i]
            slope = float(np.clip((1-alpha)*coef[i] + alpha*trend, .005, .3))
            compounds[c] = dict(offset=coef[4+i] if i<2 else 0., slope=slope,
                                slope_sigma=max(.018, abs(slope)*(.35 if n else .65)),
                                practice_samples=n, historical_samples=prior["compound_samples"][c],
                                source="practice + historical prior" if n else "historical prior" if prior["compound_samples"][c] else "scenario assumption")
        drivers[code] = dict(base_seconds=base, progression=coef[3], compounds=compounds,
                             practice_samples=samples, qualifying_delta=qdelta if qualifying[code] else None,
                             pace_sigma=max(.8, prior["offset_sigma"]+(0 if samples>=15 else .8)),
                             lap_sigma=prior["residual_sigma"],
                             status="supported" if base and samples>=5 and prior["training_laps"] else "sparse" if base else "unavailable")
    support = "scenario-only" if not prior["training_laps"] or not prior["correction_events"] else "limited" if len(prior["training_events"])<3 else "experimental"
    return dict(version=MODEL_VERSION, support=support, prior=prior, drivers=drivers, stints=runs,
                event_practice_base=event_base, baseline=event_base,
                assumptions=["Dry, uninterrupted race; free air; no traffic or finishing-position simulation",
                             "Practice fuel loads and traffic are unknown; trends are inferred, not measured tyre wear",
                             "Deleted-lap status is unavailable in some cached sessions; known deletions are excluded, other timing checks still apply",
                             "Soft/Medium/Hard labels are event-relative; cross-event tyre nomination is unknown",
                             "All ranges are model scenarios and have not been calibrated for predictive coverage"])


def predict_lap(model, driver, compound, tyre_age, race_lap):
    d = model["drivers"][driver]
    if d["base_seconds"] is None or compound not in d["compounds"]:
        return None
    c = d["compounds"][compound]
    middle = d["base_seconds"]+c["offset"]+c["slope"]*tyre_age+d["progression"]*race_lap
    sigma = np.sqrt(d["pace_sigma"]**2 + d["lap_sigma"]**2 + (tyre_age*c["slope_sigma"])**2)
    return dict(median=round(middle, 3), p10=round(middle-1.28155*sigma, 3), p90=round(middle+1.28155*sigma, 3))
