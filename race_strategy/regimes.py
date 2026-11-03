from datetime import datetime, timezone


def utc(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamp must include a timezone")
    return result.astimezone(timezone.utc)


def era_for(year):
    if year >= 2026:
        return "2026+"
    if year >= 2022:
        return "2022-2025"
    return "pre-2022"


def eligible_events(events, year, cutoff):
    return [e for e in events if era_for(e["year"]) == era_for(year)
            and utc(e["date"]) < utc(cutoff)]
