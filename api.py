"""
F1 Telemetry FastAPI Backend
Serves all data to the React frontend.
"""

import warnings
warnings.filterwarnings('ignore')

import fastf1
import pandas as pd
import numpy as np
import os
from typing import Optional, Dict, Tuple, List
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
DATA_DIR = './data'
CACHE_DIR = './cache'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

TEAM_COLORS: Dict[str, Tuple[str, str]] = {
    'Red Bull Racing': ('#3671C6', '#FFD700'),
    'Ferrari': ('#E80020', '#FFEB00'),
    'Mercedes': ('#27F4D2', '#000000'),
    'McLaren': ('#FF8000', '#47C7FC'),
    'Aston Martin': ('#229971', '#CEDC00'),
    'Alpine': ('#FF87BC', '#0093CC'),
    'Williams': ('#64C4FF', '#00A3E0'),
    'RB': ('#6692FF', '#FFFFFF'),
    'Kick Sauber': ('#52E252', '#000000'),
    'Haas F1 Team': ('#B6BABD', '#E10600'),
    'AlphaTauri': ('#6692FF', '#FFFFFF'),
    'Alfa Romeo': ('#C92D4B', '#A12239'),
}

app = FastAPI(title="F1 Telemetry API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def lighten_color(hex_color: str, factor: float = 0.45) -> str:
    """Return a lighter shade of a hex colour."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f'#{r:02x}{g:02x}{b:02x}'

def rotate(xy, *, angle):
    rot_mat = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ])
    return np.matmul(xy, rot_mat)

def _generate_track_data(year: int, gp_name: str, session_type: str = 'Q'):
    track_file = os.path.join(DATA_DIR, f'{gp_name}_{year}_track_layout.csv')
    corners_file = os.path.join(DATA_DIR, f'{gp_name}_{year}_corners.csv')

    if os.path.exists(track_file) and os.path.exists(corners_file):
        track_df = pd.read_csv(track_file)
        corners_df = pd.read_csv(corners_file).fillna({'Letter': ''})
        return track_df, corners_df

    session = fastf1.get_session(year, gp_name, session_type)
    session.load(weather=False, messages=False)
    lap = session.laps.pick_fastest()
    pos = lap.get_telemetry()
    circuit_info = session.get_circuit_info()

    track = pos.loc[:, ('X', 'Y')].to_numpy()
    track_angle = circuit_info.rotation / 180 * np.pi
    rotated_track = rotate(track, angle=track_angle)

    track_df = pd.DataFrame({
        'x': rotated_track[:, 0],
        'y': rotated_track[:, 1],
        'distance': pos['Distance'].to_numpy()
    })

    corners = circuit_info.corners
    offset_vector = np.array([[500, 0]])
    corner_x, corner_y, label_x, label_y, corner_distances = [], [], [], [], []

    for _, row in corners.iterrows():
        corner_angle = row['Angle'] / 180 * np.pi
        rotated_offset = rotate(offset_vector, angle=corner_angle)[0]
        raw_text = np.array([[row['X'] + rotated_offset[0], row['Y'] + rotated_offset[1]]])
        raw_corner = np.array([[row['X'], row['Y']]])
        final_corner = rotate(raw_corner, angle=track_angle)[0]
        final_text = rotate(raw_text, angle=track_angle)[0]
        corner_x.append(float(final_corner[0]))
        corner_y.append(float(final_corner[1]))
        label_x.append(float(final_text[0]))
        label_y.append(float(final_text[1]))
        corner_distances.append(float(row.get('Distance', 0)))

    corners_df = pd.DataFrame({
        'Number': corners['Number'].tolist(),
        'Letter': corners['Letter'].fillna('').tolist(),
        'distance': corner_distances,
        'x': corner_x, 'y': corner_y,
        'label_x': label_x, 'label_y': label_y
    })

    track_df.to_csv(track_file, index=False)
    corners_df.to_csv(corners_file, index=False)
    return track_df, corners_df

def _load_track_data(year: int, gp_name: str, session_type: str = 'Q'):
    try:
        return _generate_track_data(year, gp_name, session_type)
    except Exception as e:
        for fallback_year in [year - 1, year - 2]:
            try:
                return _generate_track_data(fallback_year, gp_name, session_type)
            except Exception:
                continue
        raise HTTPException(status_code=404, detail=f"Track data unavailable for {gp_name}: {e}")

def _format_sector(s):
    try:
        if pd.isna(s):
            return 'N/A'
        return f"{s.total_seconds():.3f}"
    except:
        return 'N/A'

def _build_session_data(year: int, gp_name: str, session_code: str, qual_session: Optional[str]):
    session = fastf1.get_session(year, gp_name, session_code)
    session.load(weather=False, messages=False)

    # Driver color map
    driver_colors_map = {}
    laps = session.laps
    drivers_all = sorted(laps['Driver'].dropna().unique())
    team_drivers: Dict[str, list] = {}
    for drv in drivers_all:
        try:
            info = session.get_driver(drv)
            team = info.get('TeamName', 'Unknown')
            team_drivers.setdefault(team, []).append(drv)
        except:
            continue
    for team, drvs in team_drivers.items():
        primary = TEAM_COLORS.get(team, ('#888888', '#AAAAAA'))[0]
        for i, drv in enumerate(drvs):
            col = primary if i == 0 else lighten_color(primary, 0.45)
            driver_colors_map[drv] = {'color': col, 'team': team, 'teamColor': primary}

    # Q split
    if qual_session and session_code == 'Q':
        try:
            q_splits = laps.split_qualifying_sessions()
            idx = {'Q1': 0, 'Q2': 1, 'Q3': 2}.get(qual_session, 2)
            if idx < len(q_splits):
                laps = q_splits[idx]
        except:
            pass

    laps = laps[laps['IsAccurate'] == True]

    drivers = sorted(laps['Driver'].dropna().unique())
    lap_times = []
    driver_tel = {}

    for drv in drivers:
        try:
            drv_laps = laps.pick_driver(drv)
            if drv_laps.empty:
                continue
            info = session.get_driver(drv)
            team = info.get('TeamName', 'Unknown')
            best_lap = drv_laps.pick_fastest()
            if best_lap is None or pd.isna(best_lap['LapTime']):
                continue
            tel = best_lap.get_telemetry()
            if tel.empty:
                continue

            t0 = tel['Time'].iloc[0]
            tel = tel.copy()
            tel['Trel'] = (tel['Time'] - t0).dt.total_seconds()
            tel['Distance'] = tel['Distance'].astype(float)

            # Downsample for performance - take every 3rd point
            tel = tel.iloc[::3].reset_index(drop=True)

            cols = ['Distance', 'Speed', 'Trel', 'Throttle', 'Brake', 'RPM', 'nGear', 'DRS']
            available = [c for c in cols if c in tel.columns]
            tel_dict = tel[available].fillna(0).to_dict(orient='list')
            driver_tel[drv] = tel_dict

            compound = best_lap.get('Compound', 'MEDIUM')
            tyre_life = best_lap.get('TyreLife', 0)
            s1 = best_lap.get('Sector1Time', pd.NaT)
            s2 = best_lap.get('Sector2Time', pd.NaT)
            s3 = best_lap.get('Sector3Time', pd.NaT)

            lt = best_lap['LapTime']
            if pd.notna(lt):
                total_sec = lt.total_seconds()
                mins = int(total_sec // 60)
                secs = total_sec % 60
                lap_str = f"{mins}:{secs:06.3f}"
            else:
                lap_str = 'N/A'
                total_sec = float('inf')

            lap_times.append({
                'driver': drv,
                'team': team,
                'lapTime': total_sec,
                'bestLapStr': lap_str,
                'compound': str(compound).upper() if pd.notna(compound) else 'MEDIUM',
                'tyreLife': int(tyre_life) if pd.notna(tyre_life) else 0,
                'S1': _format_sector(s1),
                'S2': _format_sector(s2),
                'S3': _format_sector(s3),
                'S1_sec': s1.total_seconds() if pd.notna(s1) else 9999,
                'S2_sec': s2.total_seconds() if pd.notna(s2) else 9999,
                'S3_sec': s3.total_seconds() if pd.notna(s3) else 9999,
                'color': driver_colors_map.get(drv, {}).get('color', '#888888'),
            })

            if drv not in driver_colors_map:
                driver_colors_map[drv] = {'color': '#888888', 'team': team}

        except Exception:
            continue

    lap_times.sort(key=lambda x: x['lapTime'])
    leader_time = lap_times[0]['lapTime'] if lap_times else 0

    for i, row in enumerate(lap_times):
        row['position'] = i + 1
        row['gap'] = '' if i == 0 else f"+{row['lapTime'] - leader_time:.3f}"

    # Best sectors
    if lap_times:
        best_s1 = min(r['S1_sec'] for r in lap_times)
        best_s2 = min(r['S2_sec'] for r in lap_times)
        best_s3 = min(r['S3_sec'] for r in lap_times)
        for row in lap_times:
            row['S1_best'] = row['S1_sec'] == best_s1
            row['S2_best'] = row['S2_sec'] == best_s2
            row['S3_best'] = row['S3_sec'] == best_s3

    max_duration = max((max(t['Trel']) for t in driver_tel.values() if t.get('Trel')), default=0.0)

    return lap_times, driver_tel, driver_colors_map, max_duration

# ─────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────

@app.get("/api/schedule/{year}")
def get_schedule(year: int):
    try:
        schedule = fastf1.get_event_schedule(year)
        races = schedule[schedule['EventFormat'] != 'testing'][['EventName', 'EventDate', 'EventFormat']].copy()
        races['EventDate'] = races['EventDate'].astype(str)
        return {"races": races.to_dict(orient='records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/track")
def get_track(year: int, gp: str, session_code: str = 'Q'):
    track_df, corners_df = _load_track_data(year, gp, session_code)
    # Downsample track points for faster transfer
    step = max(1, len(track_df) // 800)
    track_sampled = track_df.iloc[::step]
    return {
        "track": {
            "x": track_sampled['x'].tolist(),
            "y": track_sampled['y'].tolist(),
            "distance": track_sampled['distance'].tolist(),
        },
        "corners": corners_df.to_dict(orient='records'),
        "bounds": {
            "xMin": float(track_df['x'].min()),
            "xMax": float(track_df['x'].max()),
            "yMin": float(track_df['y'].min()),
            "yMax": float(track_df['y'].max()),
        }
    }

@app.get("/api/session")
def get_session_data(year: int, gp: str, session_code: str = 'Q', qual_session: Optional[str] = None):
    try:
        lap_times, driver_tel, driver_colors_map, max_duration = _build_session_data(
            year, gp, session_code, qual_session
        )
        return {
            "leaderboard": lap_times,
            "telemetry": driver_tel,
            "driverColors": driver_colors_map,
            "maxDuration": max_duration,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/time-delta")
def get_time_delta(year: int, gp: str, session_code: str, qual_session: Optional[str], driver1: str, driver2: str):
    try:
        lap_times, driver_tel, _, _ = _build_session_data(year, gp, session_code, qual_session)
        if driver1 not in driver_tel or driver2 not in driver_tel:
            raise HTTPException(status_code=404, detail="Driver telemetry not found")

        tel1 = driver_tel[driver1]
        tel2 = driver_tel[driver2]

        d1 = np.array(tel1['Distance'])
        t1 = np.array(tel1['Trel'])
        d2 = np.array(tel2['Distance'])
        t2 = np.array(tel2['Trel'])

        max_dist = min(d1.max(), d2.max())
        distance_grid = np.linspace(0, max_dist, 500)
        time1 = np.interp(distance_grid, d1, t1)
        time2 = np.interp(distance_grid, d2, t2)
        delta = (time1 - time2).tolist()

        speed1 = np.interp(distance_grid, d1, np.array(tel1.get('Speed', [0]*len(d1)))).tolist()
        speed2 = np.interp(distance_grid, d2, np.array(tel2.get('Speed', [0]*len(d2)))).tolist()

        return {
            "distance": distance_grid.tolist(),
            "delta": delta,
            "speed1": speed1,
            "speed2": speed2,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}
