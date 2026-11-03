"""Conservative timing filters. Green flags do not prove absence of traffic."""
from collections import defaultdict
import math
import numpy as np

COMPOUNDS = ("SOFT", "MEDIUM", "HARD")


def clean_laps(rows):
    return [r for r in rows if r.get("compound") in COMPOUNDS
            and r.get("accurate") is True and r.get("deleted") is not True
            and r.get("pit_in") is False and r.get("pit_out") is False
            and str(r.get("track_status")) == "1"
            and isinstance(r.get("seconds"), (float, int))
            and math.isfinite(r["seconds"]) and 30 < r["seconds"] < 240
            and isinstance(r.get("tyre_age"), (float, int))
            and math.isfinite(r["tyre_age"]) and r["tyre_age"] > 0]


def summarize_stints(rows, minimum=5):
    groups = defaultdict(list)
    for row in clean_laps(rows):
        groups[row["driver"], row["session"], row["stint"], row["compound"]].append(row)
    result = []
    for (driver, session, stint, compound), laps in sorted(groups.items()):
        ages = np.array([r["tyre_age"] for r in laps])
        times = np.array([r["seconds"] for r in laps])
        # A robust within-run filter removes obvious slow preparation/traffic laps.
        median = np.median(times)
        mad = np.median(np.abs(times - median))
        mask = np.abs(times - median) <= max(2.5, 3 * 1.4826 * mad)
        ages, times = ages[mask], times[mask]
        usable = [r for r, keep in zip(laps, mask) if keep]
        if not usable:
            continue
        trend = float(np.polyfit(ages, times, 1)[0]) if len(set(ages)) >= 3 else None
        result.append(dict(driver=driver, session=session, stint=int(stint), compound=compound,
                           samples=len(usable), age_min=float(ages.min()), age_max=float(ages.max()),
                           start_lap=min(r["lap"] for r in usable), end_lap=max(r["lap"] for r in usable),
                           fresh=usable[0].get("fresh"), median_seconds=round(float(np.median(times)), 3),
                           trend=round(trend, 4) if trend is not None else None,
                           long_run=len(usable) >= minimum, traffic="not independently measured"))
    return result
