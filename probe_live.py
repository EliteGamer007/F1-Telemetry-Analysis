"""
Live data probe for Monaco 2026.
Tests FastF1 partial-session load AND OpenF1 REST API.
Run this while the race is live.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import time
import urllib.request
import urllib.error
import json
import datetime

print("=" * 60)
print("F1 LIVE DATA PROBE — Monaco 2026 Race")
print(f"Local time: {datetime.datetime.now().strftime('%H:%M:%S')}")
print("=" * 60)

# ── 1. OpenF1 REST — instant, no auth needed ─────────────────────
OPENF1 = "https://api.openf1.org/v1"

def openf1_get(endpoint, params=""):
    url = f"{OPENF1}/{endpoint}?{params}"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
        return data, None
    except Exception as e:
        return None, str(e)

print("\n[1] OpenF1 — fetching latest session info...")
sessions, err = openf1_get("sessions", "year=2026&circuit_short_name=monaco")
if err:
    print(f"  ✗ Error: {err}")
elif sessions:
    sess = sessions[-1]
    print(f"  ✓ Latest session: {sess.get('session_name')} | {sess.get('date_start')} → {sess.get('date_end')}")
    print(f"    Status: {sess.get('session_status','?')} | Key: {sess.get('session_key','?')}")
    
    session_key = sess.get("session_key")

    print(f"\n[2] OpenF1 — fetching LIVE driver positions (session_key={session_key})...")
    t0 = time.time()
    positions, err2 = openf1_get("location", f"session_key={session_key}&driver_number=1")
    latency = time.time() - t0

    if err2:
        print(f"  ✗ Error: {err2}")
    elif positions:
        last = positions[-1]
        data_ts = last.get("date", "?")
        now_utc = datetime.datetime.utcnow()
        print(f"  ✓ Latest position for #1 VER:")
        print(f"    Data timestamp (UTC): {data_ts}")
        print(f"    Current time  (UTC):  {now_utc.strftime('%Y-%m-%dT%H:%M:%S')}")
        print(f"    API response latency: {latency:.2f}s")
        # Parse delay
        try:
            dt = datetime.datetime.fromisoformat(data_ts.replace("Z",""))
            delay = (now_utc - dt).total_seconds()
            print(f"    ⏱  Data delay: {delay:.1f}s ({delay/60:.1f} min)")
            if delay < 120:
                print("    🟢 EXCELLENT — sub 2-minute delay, real-time usable!")
            elif delay < 300:
                print("    🟡 GOOD — 2-5 minute delay, watchalong viable")
            else:
                print("    🔴 HIGH delay — not suitable for live watchalong")
        except Exception as e:
            print(f"    Could not parse delay: {e}")
    else:
        print("  No position data returned (race may not have started yet)")

    print(f"\n[3] OpenF1 — timing / lap data freshness...")
    laps, err3 = openf1_get("laps", f"session_key={session_key}&driver_number=1")
    if laps:
        last_lap = laps[-1]
        print(f"  ✓ Latest lap for #1: Lap {last_lap.get('lap_number')} | {last_lap.get('lap_duration')}s")
        print(f"    Date start: {last_lap.get('date_start')}")
    else:
        print(f"  No lap data yet: {err3}")

    print(f"\n[4] OpenF1 — race control messages (flags, SC)...")
    rc, err4 = openf1_get("race_control", f"session_key={session_key}")
    if rc:
        print(f"  ✓ {len(rc)} race control messages. Latest:")
        for msg in rc[-3:]:
            print(f"    [{msg.get('date','?')}] {msg.get('flag','?')} — {msg.get('message','?')[:80]}")
    else:
        print(f"  No RC messages: {err4}")

    print(f"\n[5] OpenF1 — pit stops...")
    pits, _ = openf1_get("pit", f"session_key={session_key}")
    if pits:
        print(f"  ✓ {len(pits)} pit stops recorded so far")
    else:
        print("  No pit data yet")
else:
    print("  No Monaco 2026 sessions found. Race may not be indexed yet.")

# ── 2. FastF1 probe ──────────────────────────────────────────────
print("\n[6] FastF1 — attempting partial live session load...")
try:
    import fastf1
    fastf1.Cache.enable_cache('./cache')
    t0 = time.time()
    session = fastf1.get_session(2026, 'Monaco', 'R')
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    elapsed = time.time() - t0
    laps = session.laps
    print(f"  ✓ FastF1 loaded in {elapsed:.1f}s")
    print(f"  Total laps in cache: {len(laps)}")
    if not laps.empty:
        latest = laps.sort_values('Time').iloc[-1]
        print(f"  Latest lap: {latest.get('Driver')} L{int(latest.get('LapNumber',0))} @ {latest.get('Time')}")
    else:
        print("  No lap data yet from FastF1")
except Exception as e:
    print(f"  ✗ FastF1 error: {e}")

print("\n" + "=" * 60)
print("VERDICT")
print("=" * 60)
print("""
FastF1 livetiming module: RECORD-ONLY during session, process after.
  → NOT suitable for real-time watchalong directly.

OpenF1 REST API:
  → Free tier: historical + current session data (polling)
  → Sponsor tier: WebSocket/MQTT true real-time streaming
  → Car location at 3.7 Hz, timing every ~4 seconds
  → The delay seen above tells us if it's watchalong-viable.

RECOMMENDATION:
  Use OpenF1 REST (free) with polling every 2-4 seconds for
  live positions, timing, and flag data during a race.
  This gives a live experience with minimal latency.
""")
