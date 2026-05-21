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

def _format_laptime(lt) -> str:
    """Format a timedelta into m:ss.mmm string."""
    try:
        if pd.isna(lt):
            return 'N/A'
        total_sec = lt.total_seconds()
        mins = int(total_sec // 60)
        secs = total_sec % 60
        return f"{mins}:{secs:06.3f}"
    except:
        return 'N/A'

def _build_session_data(year: int, gp_name: str, session_code: str, qual_session: Optional[str]):
    session = fastf1.get_session(year, gp_name, session_code)
    session.load(weather=False, messages=False)

    is_race = (session_code == 'R')
    total_laps = session.total_laps if is_race else None
    lap_history = {} if is_race else None

    # ── Driver retired/status from session results ──────────────
    driver_status: Dict[str, str] = {}       # abbr -> 'Finished' | 'Lapped' | 'Retired'
    driver_finish_pos: Dict[str, int] = {}   # abbr -> official finishing position
    if is_race:
        try:
            results = session.results
            for _, row in results.iterrows():
                abbr = row.get('Abbreviation', '')
                status = row.get('Status', 'Finished')
                pos = row.get('Position', None)
                driver_status[abbr] = str(status)
                if abbr and pos is not None and pd.notna(pos):
                    driver_finish_pos[abbr] = int(pos)
        except Exception:
            pass

    # ── Driver color map ────────────────────────────────────────
    driver_colors_map = {}
    laps_all = session.laps
    drivers_all = sorted(laps_all['Driver'].dropna().unique())
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

    # ── Q split ─────────────────────────────────────────────────
    laps = laps_all.copy()
    if qual_session and session_code == 'Q':
        try:
            q_splits = laps.split_qualifying_sessions()
            idx = {'Q1': 0, 'Q2': 1, 'Q3': 2}.get(qual_session, 2)
            if idx < len(q_splits):
                laps = q_splits[idx]
        except:
            pass

    if not is_race:
        laps = laps[laps['IsAccurate'] == True]

    # ── Global race start time (t0) ─────────────────────────────
    # t0 = the SessionTime at which lights-out happened.
    # We derive it from: lap1_end_time - lap1_laptime for all drivers, take the min.
    t0 = pd.NaT
    if is_race and not laps_all.empty:
        first_laps = laps_all[laps_all['LapNumber'] == 1]
        if not first_laps.empty:
            computed_starts = first_laps['Time'] - first_laps['LapTime']
            t0 = computed_starts.dropna().min()
        if pd.isna(t0):
            t0 = laps_all['Time'].dropna().min()

    # ── Get circuit rotation (same as used to build track layout) ─
    track_angle = 0.0
    try:
        circuit_info = session.get_circuit_info()
        track_angle = float(circuit_info.rotation) / 180.0 * np.pi
    except Exception:
        track_angle = 0.0

    drivers = sorted(laps['Driver'].dropna().unique())
    lap_times = []
    driver_tel = {}


    for drv in drivers:
        try:
            drv_laps = laps.pick_drivers(drv)
            if drv_laps.empty:
                continue
            info = session.get_driver(drv)
            team = info.get('TeamName', 'Unknown')
            abbr = info.get('Abbreviation', str(drv))
            best_lap = drv_laps.pick_fastest()
            if best_lap is None or pd.isna(best_lap.get('LapTime')):
                continue

            if is_race:
                # ── Fetch full race telemetry ───────────────────
                tel = drv_laps.get_telemetry()

                # ── Build per-lap history ───────────────────────
                # Each entry: when this lap ENDED (trel), what position/compound/tyrelife
                history = []
                for _, lap_row in drv_laps.iterrows():
                    lap_end_time = lap_row.get('Time')
                    if pd.isna(lap_end_time) or pd.isna(t0):
                        continue
                    rel_time = (lap_end_time - t0).total_seconds()

                    # Pit-in trel: if this lap has a PitInTime, car enters pit at that moment
                    pit_in_trel = None
                    pit_out_trel = None
                    if pd.notna(lap_row.get('PitInTime')):
                        pit_in_trel = (lap_row['PitInTime'] - t0).total_seconds()
                    if pd.notna(lap_row.get('PitOutTime')):
                        pit_out_trel = (lap_row['PitOutTime'] - t0).total_seconds()

                    l_laptime = lap_row.get('LapTime')
                    l_laptime_sec = l_laptime.total_seconds() if pd.notna(l_laptime) else 0
                    s1 = lap_row.get('Sector1Time', pd.NaT)
                    s2 = lap_row.get('Sector2Time', pd.NaT)
                    s3 = lap_row.get('Sector3Time', pd.NaT)

                    history.append({
                        'lap': int(lap_row.get('LapNumber', 0)),
                        'position': int(lap_row.get('Position', 0)) if pd.notna(lap_row.get('Position')) else 0,
                        'trel': rel_time,
                        'lapTime': l_laptime_sec,
                        'lapTimeStr': _format_laptime(l_laptime),
                        'compound': str(lap_row.get('Compound', 'UNKNOWN')).upper(),
                        'tyreLife': int(lap_row.get('TyreLife', 0)) if pd.notna(lap_row.get('TyreLife')) else 0,
                        'S1': _format_sector(s1),
                        'S2': _format_sector(s2),
                        'S3': _format_sector(s3),
                        'pitInTrel': pit_in_trel,
                        'pitOutTrel': pit_out_trel,
                    })

                # Sort by lap number ascending
                history.sort(key=lambda h: h['lap'])
                lap_history[abbr] = history
            else:
                tel = best_lap.get_telemetry()

            if tel.empty:
                continue

            # ── Compute Trel ────────────────────────────────────
            if is_race and pd.notna(t0):
                drv_t0 = t0
                if 'SessionTime' in tel.columns:
                    tel = tel.copy()
                    tel['Trel'] = (tel['SessionTime'] - drv_t0).dt.total_seconds()
                else:
                    tel = tel.copy()
                    tel['Trel'] = (tel['Time'] - drv_t0).dt.total_seconds()
            else:
                drv_t0 = tel['SessionTime'].iloc[0] if 'SessionTime' in tel.columns else tel['Time'].iloc[0]
                tel = tel.copy()
                if 'SessionTime' in tel.columns:
                    tel['Trel'] = (tel['SessionTime'] - drv_t0).dt.total_seconds()
                else:
                    tel['Trel'] = (tel['Time'] - drv_t0).dt.total_seconds()

            tel['Distance'] = tel['Distance'].astype(float)

            # Compute last valid trel (for retired drivers — freeze dot beyond this)
            # A driver is truly finished when Speed drops to 0 permanently
            # Find the last index where speed > 5
            if is_race and 'Speed' in tel.columns:
                active_mask = tel['Speed'] > 5
                if active_mask.any():
                    last_active_idx = active_mask.values[::-1].argmax()
                    last_active_idx = len(tel) - 1 - last_active_idx
                    max_trel = float(tel['Trel'].iloc[last_active_idx])
                else:
                    max_trel = float(tel['Trel'].iloc[-1])
            else:
                max_trel = float(tel['Trel'].iloc[-1])

            # Downsample for performance
            step = max(1, len(tel) // 3000) if is_race else 3
            tel = tel.iloc[::step].reset_index(drop=True)

            if 'X' in tel.columns and 'Y' in tel.columns:
                # Rotate X/Y to match the track outline rotation
                coords = tel[['X', 'Y']].to_numpy()
                rotated_coords = rotate(coords, angle=track_angle)
                tel['X'] = rotated_coords[:, 0]
                tel['Y'] = rotated_coords[:, 1]

            cols = ['Distance', 'Speed', 'Trel', 'Throttle', 'Brake', 'RPM', 'nGear', 'DRS', 'X', 'Y']
            available = [c for c in cols if c in tel.columns]
            tel_dict = tel[available].fillna(0).to_dict(orient='list')
            tel_dict['maxTrel'] = max_trel
            driver_tel[abbr] = tel_dict

            # ── Leaderboard row (fastest lap data for qualifying; last lap for race) ──
            compound = best_lap.get('Compound', 'MEDIUM')
            tyre_life = best_lap.get('TyreLife', 0)
            s1 = best_lap.get('Sector1Time', pd.NaT)
            s2 = best_lap.get('Sector2Time', pd.NaT)
            s3 = best_lap.get('Sector3Time', pd.NaT)

            lt = best_lap.get('LapTime')
            if pd.notna(lt):
                total_sec = lt.total_seconds()
                lap_str = _format_laptime(lt)
            else:
                lap_str = 'N/A'
                total_sec = float('inf')

            status = driver_status.get(abbr, 'Finished')

            lap_times.append({
                'driver': abbr,
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
                'color': driver_colors_map.get(abbr, {}).get('color', '#888888'),
                'status': status,  # 'Finished' | 'Lapped' | 'Retired'
            })

            if abbr not in driver_colors_map:
                driver_colors_map[abbr] = {'color': '#888888', 'team': team}

        except Exception:
            continue

    if is_race and driver_finish_pos:
        # Sort by official finishing position from session.results
        lap_times.sort(key=lambda x: driver_finish_pos.get(x['driver'], 999))
        for i, row in enumerate(lap_times):
            finish_pos = driver_finish_pos.get(row['driver'], i + 1)
            row['position'] = finish_pos
            if i == 0:
                row['gap'] = ''
            elif row.get('status') == 'Retired':
                row['gap'] = 'DNF'
            else:
                row['gap'] = f"+{i}"  # placeholder; dynamic gap comes from lap history
    else:
        # Qualifying: sort by fastest lap time
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

    # ── Track status events (SC, VSC, Yellow, Red flag) ─────────
    track_status_events = []
    if is_race and pd.notna(t0):
        try:
            ts_df = session.track_status
            # Status codes: 1=AllClear, 2=Yellow, 3=SCDeploying, 4=SafetyCar, 5=RedFlag, 6=VSCDeployed, 7=VSCEnding
            STATUS_MAP = {
                '1': 'AllClear', '2': 'Yellow', '3': 'SCDeploying',
                '4': 'SafetyCar', '5': 'RedFlag', '6': 'VSCDeployed', '7': 'VSCEnding'
            }
            for idx, row in ts_df.iterrows():
                trel = (row['Time'] - t0).total_seconds()
                status_code = str(row.get('Status', '1'))
                # End time = start of next event
                if idx + 1 < len(ts_df):
                    end_trel = (ts_df.iloc[idx + 1]['Time'] - t0).total_seconds()
                else:
                    end_trel = max_duration
                track_status_events.append({
                    'trel': trel,
                    'endTrel': end_trel,
                    'status': STATUS_MAP.get(status_code, 'Unknown'),
                    'code': status_code,
                })
        except Exception:
            pass

    return lap_times, driver_tel, driver_colors_map, max_duration, is_race, total_laps, lap_history, track_status_events

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
        lap_times, driver_tel, driver_colors_map, max_duration, is_race, total_laps, lap_history, track_status_events = _build_session_data(
            year, gp, session_code, qual_session
        )
        return {
            "leaderboard": lap_times,
            "telemetry": driver_tel,
            "driverColors": driver_colors_map,
            "maxDuration": max_duration,
            "isRace": is_race,
            "totalLaps": total_laps,
            "lapHistory": lap_history,
            "trackStatusEvents": track_status_events,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/time-delta")
def get_time_delta(year: int, gp: str, session_code: str, qual_session: Optional[str], driver1: str, driver2: str):
    try:
        lap_times, driver_tel, _, _, _, _, _, _ = _build_session_data(year, gp, session_code, qual_session)
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
