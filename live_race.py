"""
Live Race Polling via FastF1 repeated partial loads.

Strategy:
- FastF1 CAN load partial session data mid-race (with errors/warnings)
- We suppress warnings and poll every 45 seconds
- Each poll fetches the latest lap data — typically 1-2 laps behind
- This data is normalized into exactly the same format our existing API uses
- Exposes /api/live/session and /api/live/poll endpoints

Run: python live_race.py
"""
import warnings
warnings.filterwarnings('ignore')

import fastf1
import pandas as pd
import numpy as np
import time
import os
import json
import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

logging.getLogger('fastf1').setLevel(logging.ERROR)

CACHE_DIR = './cache_live'
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

app = FastAPI(title="F1 Live Race Poller", version="1.0.0")
app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global live state
live_state: Dict[str, Any] = {
    "session_info": None,      # year, gp, session_code
    "last_poll": None,          # datetime of last successful poll
    "poll_count": 0,
    "leaderboard": [],          # same format as /api/session
    "lap_history": {},          # per driver
    "track_status": [],
    "total_laps": None,
    "is_polling": False,
    "error": None,
    "laps_completed": 0,        # highest lap number seen
}

TEAM_COLORS = {
    'Red Bull Racing': '#3671C6',
    'Ferrari': '#E80020',
    'Mercedes': '#27F4D2',
    'McLaren': '#FF8000',
    'Aston Martin': '#229971',
    'Alpine': '#FF87BC',
    'Williams': '#64C4FF',
    'RB': '#6692FF',
    'Kick Sauber': '#52E252',
    'Haas F1 Team': '#B6BABD',
}

def lighten_color(hex_color: str, factor: float = 0.45) -> str:
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f'#{int(r+(255-r)*factor):02x}{int(g+(255-g)*factor):02x}{int(b+(255-b)*factor):02x}'

def fmt_laptime(lt) -> str:
    try:
        if pd.isna(lt): return 'N/A'
        t = lt.total_seconds()
        return f"{int(t//60)}:{t%60:06.3f}"
    except: return 'N/A'

def fmt_sector(s) -> str:
    try:
        if pd.isna(s): return 'N/A'
        return f"{s.total_seconds():.3f}"
    except: return 'N/A'

def do_poll(year: int, gp: str, session_code: str) -> Dict:
    """One polling cycle — load session, extract latest lap data."""
    try:
        session = fastf1.get_session(year, gp, session_code)
        # Load only lap data (no telemetry) to keep it fast
        session.load(laps=True, telemetry=False, weather=False, messages=False)
    except Exception as e:
        return {"error": str(e), "laps": pd.DataFrame()}

    laps = session.laps
    if laps is None or laps.empty:
        return {"error": "No lap data", "laps": laps}

    # Build driver color map
    drivers = sorted(laps['Driver'].dropna().unique())
    team_drivers: Dict[str, list] = {}
    color_map = {}
    for drv in drivers:
        try:
            info = session.get_driver(drv)
            team = info.get('TeamName', 'Unknown')
            team_drivers.setdefault(team, []).append(drv)
        except: pass
    for team, drvs in team_drivers.items():
        primary = TEAM_COLORS.get(team, '#888888')
        for i, drv in enumerate(drvs):
            color_map[drv] = primary if i == 0 else lighten_color(primary, 0.45)

    # Build leaderboard (best lap per driver)
    leaderboard = []
    lap_history = {}

    for drv in drivers:
        try:
            drv_laps = laps.pick_drivers(drv)
            if drv_laps.empty: continue
            info = session.get_driver(drv)
            abbr = info.get('Abbreviation', drv)
            team = info.get('TeamName', 'Unknown')

            # Best lap
            accurate = drv_laps[drv_laps['IsAccurate'] == True]
            best = drv_laps.pick_fastest() if accurate.empty else accurate.loc[accurate['LapTime'].idxmin()]

            lt = best.get('LapTime')
            lt_sec = lt.total_seconds() if pd.notna(lt) else float('inf')
            compound = str(best.get('Compound', 'UNKNOWN')).upper()
            tyre_life = int(best.get('TyreLife', 0)) if pd.notna(best.get('TyreLife')) else 0
            s1 = best.get('Sector1Time', pd.NaT)
            s2 = best.get('Sector2Time', pd.NaT)
            s3 = best.get('Sector3Time', pd.NaT)
            last_pos = int(drv_laps.iloc[-1].get('Position', 0)) if pd.notna(drv_laps.iloc[-1].get('Position')) else 0

            leaderboard.append({
                'driver': abbr,
                'team': team,
                'lapTime': lt_sec,
                'bestLapStr': fmt_laptime(lt),
                'compound': compound if pd.notna(compound) else 'UNKNOWN',
                'tyreLife': tyre_life,
                'S1': fmt_sector(s1),
                'S2': fmt_sector(s2),
                'S3': fmt_sector(s3),
                'S1_sec': s1.total_seconds() if pd.notna(s1) else 9999,
                'S2_sec': s2.total_seconds() if pd.notna(s2) else 9999,
                'S3_sec': s3.total_seconds() if pd.notna(s3) else 9999,
                'color': color_map.get(abbr, '#888888'),
                'position': last_pos,
                'status': 'Finished',
                'gap': '',
            })

            # Build per-lap history
            history = []
            for _, lap_row in drv_laps.iterrows():
                lap_lt = lap_row.get('LapTime')
                history.append({
                    'lap': int(lap_row.get('LapNumber', 0)) if pd.notna(lap_row.get('LapNumber')) else 0,
                    'position': int(lap_row.get('Position', 0)) if pd.notna(lap_row.get('Position')) else 0,
                    'lapTime': lap_lt.total_seconds() if pd.notna(lap_lt) else None,
                    'lapTimeStr': fmt_laptime(lap_lt),
                    'compound': str(lap_row.get('Compound', 'UNKNOWN')).upper(),
                    'tyreLife': int(lap_row.get('TyreLife', 0)) if pd.notna(lap_row.get('TyreLife')) else 0,
                    'S1': fmt_sector(lap_row.get('Sector1Time', pd.NaT)),
                    'S2': fmt_sector(lap_row.get('Sector2Time', pd.NaT)),
                    'S3': fmt_sector(lap_row.get('Sector3Time', pd.NaT)),
                    'pitInTrel': None, 'pitOutTrel': None, 'trel': 0,
                })
            lap_history[abbr] = history
        except Exception:
            continue

    # Sort by position, then by lap time
    leaderboard.sort(key=lambda x: (x['position'] if x['position'] else 999, x['lapTime']))
    for i, row in enumerate(leaderboard):
        row['position'] = i + 1
        if i == 0: row['gap'] = ''
        else:
            diff = row['lapTime'] - leaderboard[0]['lapTime']
            row['gap'] = f'+{diff:.3f}' if diff < float('inf') else '—'

    # Total laps
    total_laps = None
    try:
        total_laps = int(session.total_laps) if session.total_laps else None
    except: pass

    max_lap = int(laps['LapNumber'].max()) if not laps.empty and pd.notna(laps['LapNumber'].max()) else 0

    return {
        "leaderboard": leaderboard,
        "lapHistory": lap_history,
        "totalLaps": total_laps,
        "lapsCompleted": max_lap,
        "polledAt": datetime.now(timezone.utc).isoformat(),
        "error": None,
    }


def background_poller():
    """Runs in a background thread, polls every 45 seconds."""
    while live_state["is_polling"]:
        info = live_state.get("session_info")
        if not info:
            time.sleep(5)
            continue
        try:
            print(f"[Live Poll #{live_state['poll_count']+1}] {info['year']} {info['gp']} {info['session_code']}...")
            t0 = time.time()
            result = do_poll(info['year'], info['gp'], info['session_code'])
            elapsed = time.time() - t0

            if result.get("error"):
                live_state["error"] = result["error"]
                print(f"  Poll error: {result['error']}")
            else:
                live_state["leaderboard"] = result["leaderboard"]
                live_state["lap_history"] = result["lapHistory"]
                live_state["total_laps"] = result["totalLaps"]
                live_state["laps_completed"] = result["lapsCompleted"]
                live_state["last_poll"] = result["polledAt"]
                live_state["poll_count"] += 1
                live_state["error"] = None
                print(f"  OK in {elapsed:.1f}s — {len(result['leaderboard'])} drivers, lap {result['lapsCompleted']}/{result['totalLaps']}")
        except Exception as e:
            live_state["error"] = str(e)
            print(f"  Unhandled error: {e}")

        time.sleep(45)  # poll every 45 seconds


@app.post("/api/live/start")
def start_live(year: int, gp: str, session_code: str = 'R'):
    """Begin polling a live/recent session."""
    live_state["session_info"] = {"year": year, "gp": gp, "session_code": session_code}
    live_state["poll_count"] = 0
    live_state["error"] = None
    if not live_state["is_polling"]:
        live_state["is_polling"] = True
        t = threading.Thread(target=background_poller, daemon=True)
        t.start()
    return {"status": "started", "session": live_state["session_info"]}


@app.post("/api/live/stop")
def stop_live():
    live_state["is_polling"] = False
    live_state["session_info"] = None
    return {"status": "stopped"}


@app.get("/api/live/state")
def get_live_state():
    """Returns current live leaderboard — same shape as /api/session."""
    return {
        "leaderboard": live_state["leaderboard"],
        "lapHistory": live_state["lap_history"],
        "totalLaps": live_state["total_laps"],
        "lapsCompleted": live_state["laps_completed"],
        "lastPoll": live_state["last_poll"],
        "pollCount": live_state["poll_count"],
        "isPolling": live_state["is_polling"],
        "sessionInfo": live_state["session_info"],
        "error": live_state["error"],
        # Stub these out — telemetry not available live
        "telemetry": {},
        "fastestLapTelemetry": {},
        "driverColors": {row['driver']: {"color": row['color'], "team": row['team']}
                         for row in live_state["leaderboard"]},
        "maxDuration": 0,
        "isRace": True,
        "trackStatusEvents": [],
        "practiceDriverData": None,
        "trafficIncidents": [],
    }


@app.get("/health")
def health():
    return {"status": "ok", "is_polling": live_state["is_polling"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
