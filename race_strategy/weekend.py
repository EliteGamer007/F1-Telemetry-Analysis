"""FastF1 is imported only for offline acquisition, never by the public API."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from .regimes import utc, era_for, eligible_events


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def validate_bundle(bundle):
    cutoff = utc(bundle["cutoff"])
    allowed = {"FP1", "FP2", "FP3", "Q"}
    if any(s["code"] not in allowed for s in bundle["sessions"]):
        raise ValueError("Forecast inputs must be practice or qualifying only")
    if any(utc(s["end"]) > cutoff for s in bundle["sessions"]):
        raise ValueError("Session extends beyond forecast cutoff")
    if any(lap["session"] not in allowed for lap in bundle["laps"]):
        raise ValueError("Lap inputs must be practice or qualifying only")
    return bundle


def normalize_session(session, code):
    import pandas as pd
    rows = []
    def number(value):
        return None if pd.isna(value) else float(value)
    for _, lap in session.laps.iterrows():
        if pd.isna(lap.get("Stint")) or pd.isna(lap.get("LapNumber")):
            continue
        seconds = lap["LapTime"].total_seconds() if pd.notna(lap["LapTime"]) else None
        rows.append(dict(driver=str(lap["Driver"]), session=code, lap=int(lap["LapNumber"]),
                         stint=int(lap["Stint"]), compound=str(lap["Compound"]),
                         tyre_age=number(lap["TyreLife"]), seconds=seconds,
                         accurate=None if pd.isna(lap["IsAccurate"]) else bool(lap["IsAccurate"]),
                         deleted=None if pd.isna(lap.get("Deleted")) else bool(lap["Deleted"]),
                         pit_in=bool(pd.notna(lap["PitInTime"])), pit_out=bool(pd.notna(lap["PitOutTime"])),
                         track_status=str(lap["TrackStatus"]),
                         fresh=None if pd.isna(lap["FreshTyre"]) else bool(lap["FreshTyre"])))
    return rows


def load_session(year, round_number, code, cache="cache", cache_only=False):
    import fastf1
    Path(cache).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(cache)
    session = fastf1.get_session(year, round_number, code)
    if cache_only:
        folder = Path(cache).joinpath(*session.api_path.strip("/").split("/")[1:])
        if not (folder / "_extended_timing_data.ff1pkl").exists():
            raise ValueError("Timing input not in the local cache; archive authentication required")
    fastf1.Cache.offline_mode(cache_only)
    session.load(telemetry=False, weather=True, messages=False)
    return session


def prepare_weekend(year, round_number, cache="cache", cache_only=False):
    import pandas as pd
    import fastf1
    event = fastf1.get_event(year, round_number)
    if event["EventFormat"] not in {"conventional", "sprint_qualifying", "sprint_shootout"}:
        raise ValueError("Unsupported weekend format")
    # Sprint-weekend GRAND PRIX only; never predict a sprint or use its race labels.
    codes = ("FP1", "FP2", "FP3", "Q") if event["EventFormat"]=="conventional" else ("FP1", "Q")
    sessions, rows, missing = [], [], []
    driver_rows = None
    for code in codes:
        try:
            s = load_session(year, round_number, code, cache, cache_only)
            normalized = normalize_session(s, code)
            # Session timing is elapsed since t0. Never infer the cutoff from race results.
            if not normalized:
                raise ValueError("No timing observations available")
            # t0_date depends on car/position telemetry, intentionally not loaded.
            # Use scheduled start plus observed elapsed duration and a 15-minute buffer.
            end = pd.Timestamp(s.date) + (s.laps["Time"].max() - s.session_start_time) + pd.Timedelta(minutes=15)
            end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
            try:
                weather = s.weather_data
            except fastf1.core.DataNotLoadedError:
                weather = pd.DataFrame()
            sessions.append(dict(code=code, end=end.isoformat(), laps=len(normalized),
                                 deletion_checks_available=all(r["deleted"] is not None for r in normalized),
                                 rainfall_observed=bool(weather["Rainfall"].any()) if len(weather) else None,
                                 track_temp=float(weather["TrackTemp"].median()) if len(weather) and weather["TrackTemp"].notna().any() else None))
            rows.extend(normalized)
            driver_rows = s.results
        except Exception as exc:
            missing.append(dict(code=code, reason=type(exc).__name__))
    if not sessions:
        raise ValueError("No practice or qualifying timing available")
    cutoff = max(s["end"] for s in sessions)
    drivers = [dict(code=str(r.Abbreviation), name=str(r.FullName), team=str(r.TeamName),
                    color="#" + str(r.TeamColor or "888888"),
                    qualifying_position=int(r.Position) if sessions[-1]["code"]=="Q" and pd.notna(r.Position) else None)
               for _, r in driver_rows.iterrows()]
    bundle = dict(year=year, round=round_number, event=str(event.EventName),
                  location=str(event.Location), date=str(event.EventDate.date()), format=str(event.EventFormat),
                  cutoff=cutoff, generated_at=datetime.now(timezone.utc).isoformat(), era=era_for(year),
                  sessions=sessions, missing_sessions=missing, laps=rows, drivers=drivers,
                  metadata=dict(layout="unverified — event-specific; no historical geometry transfer",
                                tyre_nomination="unknown", tyre_construction="unknown",
                                deletion_checks="Unknown where race-control messages are absent; known deleted laps excluded"))
    validate_bundle(bundle)
    bundle["source_hash"] = digest(bundle)
    return bundle


def prepare_training_table(event_keys, target, cache="cache", cache_only=False):
    """Prepare historical events separately; reject future/cross-era/target rows."""
    output = []
    for year, round_number in event_keys:
        if (year, round_number) == (target["year"], target["round"]):
            raise ValueError("Target event cannot train its own forecast")
        # Schedule date checked BEFORE loading historical race observations.
        import fastf1
        e = fastf1.get_event(year, round_number)
        date = e["Session5DateUtc"].isoformat() + "+00:00"
        info = dict(year=year, round=round_number, date=date)
        if not eligible_events([info], target["year"], target["cutoff"]):
            raise ValueError("Training event outside chronological same-era boundary")
        try:
            b = prepare_weekend(year, round_number, cache, cache_only)
            previous_laps = b["laps"]
        except ValueError:
            previous_laps = []
        race = load_session(year, round_number, "R", cache, cache_only)
        historical = previous_laps + normalize_session(race, "R")
        output.append(dict(**info, event=str(e.EventName), location=str(e.Location),
                           laps=historical, source_hash=digest(historical)))
    return output
