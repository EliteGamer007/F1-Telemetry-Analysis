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

    is_live_supported = session_code in ('R', 'S', 'FP1', 'FP2', 'FP3', 'Q', 'SQ')
    is_race = session_code in ('R', 'S')

    total_laps = session.total_laps if is_race else None
    lap_history = {} if is_live_supported else None

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
    if qual_session and session_code in ('Q', 'SQ'):
        try:
            q_splits = laps.split_qualifying_sessions()
            idx = {'Q1': 0, 'Q2': 1, 'Q3': 2, 'SQ1': 0, 'SQ2': 1, 'SQ3': 2}.get(qual_session, 2)
            if idx < len(q_splits):
                laps = q_splits[idx]
        except:
            pass

    if not is_live_supported:
        laps = laps[laps['IsAccurate'] == True]

    # ── Global session start time (t0) ──────────────────────────
    t0 = pd.NaT
    if is_live_supported and not laps_all.empty:
        first_laps = laps_all[laps_all['LapNumber'] == 1]
        if not first_laps.empty:
            computed_starts = first_laps['Time'] - first_laps['LapTime']
            t0 = computed_starts.dropna().min()
        if pd.isna(t0):
            t0 = laps_all['Time'].dropna().min()

    # ── Get circuit rotation ────────────────────────────────────
    track_angle = 0.0
    try:
        circuit_info = session.get_circuit_info()
        track_angle = float(circuit_info.rotation) / 180.0 * np.pi
    except Exception:
        track_angle = 0.0

    drivers = sorted(laps['Driver'].dropna().unique())
    lap_times = []
    driver_tel = {}
    fastest_lap_tel = {}

    def _process_tel_df(tel_df, is_live=False):
        if tel_df.empty:
            return None
        
        # Compute Trel
        if is_live and pd.notna(t0):
            drv_t0 = t0
            if 'SessionTime' in tel_df.columns:
                tel_df = tel_df.copy()
                tel_df['Trel'] = (tel_df['SessionTime'] - drv_t0).dt.total_seconds()
            else:
                tel_df = tel_df.copy()
                tel_df['Trel'] = (tel_df['Time'] - drv_t0).dt.total_seconds()
        else:
            drv_t0 = tel_df['SessionTime'].iloc[0] if 'SessionTime' in tel_df.columns else tel_df['Time'].iloc[0]
            tel_df = tel_df.copy()
            if 'SessionTime' in tel_df.columns:
                tel_df['Trel'] = (tel_df['SessionTime'] - drv_t0).dt.total_seconds()
            else:
                tel_df['Trel'] = (tel_df['Time'] - drv_t0).dt.total_seconds()

        tel_df['Distance'] = tel_df['Distance'].astype(float)

        # Freeze retired drivers
        max_trel = float(tel_df['Trel'].iloc[-1])
        if is_live and 'Speed' in tel_df.columns:
            active_mask = tel_df['Speed'] > 5
            if active_mask.any():
                last_active_idx = active_mask.values[::-1].argmax()
                last_active_idx = len(tel_df) - 1 - last_active_idx
                max_trel = float(tel_df['Trel'].iloc[last_active_idx])

        # Downsample
        step = max(1, len(tel_df) // 3000) if is_live else 3
        tel_df = tel_df.iloc[::step].reset_index(drop=True)

        if 'X' in tel_df.columns and 'Y' in tel_df.columns:
            coords = tel_df[['X', 'Y']].to_numpy()
            rotated_coords = rotate(coords, angle=track_angle)
            tel_df['X'] = rotated_coords[:, 0]
            tel_df['Y'] = rotated_coords[:, 1]

        cols = ['Distance', 'Speed', 'Trel', 'Throttle', 'Brake', 'RPM', 'nGear', 'DRS', 'X', 'Y']
        available = [c for c in cols if c in tel_df.columns]
        tel_dict = tel_df[available].fillna(0).to_dict(orient='list')
        tel_dict['maxTrel'] = max_trel
        return tel_dict

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

            # Always process fastest lap telemetry for comparisons
            fast_tel_df = best_lap.get_telemetry()
            fastest_lap_tel[abbr] = _process_tel_df(fast_tel_df, is_live=False)

            if is_live_supported:
                tel = drv_laps.get_telemetry()
                tel_dict = _process_tel_df(tel, is_live=True)
                if tel_dict:
                    driver_tel[abbr] = tel_dict

                # ── Build per-lap history ───────────────────────
                history = []
                for _, lap_row in drv_laps.iterrows():
                    lap_end_time = lap_row.get('Time')
                    if pd.isna(lap_end_time) or pd.isna(t0):
                        continue
                    rel_time = (lap_end_time - t0).total_seconds()

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

                history.sort(key=lambda h: h['lap'])
                lap_history[abbr] = history
            else:
                driver_tel[abbr] = fastest_lap_tel[abbr]

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

    # ── Track status events ──────────────────────────────────────
    track_status_events = []
    if is_live_supported and pd.notna(t0):
        try:
            ts_df = session.track_status
            STATUS_MAP = {
                '1': 'AllClear', '2': 'Yellow', '3': 'SCDeploying',
                '4': 'SafetyCar', '5': 'RedFlag', '6': 'VSCDeployed', '7': 'VSCEnding'
            }
            for idx, row in ts_df.iterrows():
                trel = (row['Time'] - t0).total_seconds()
                status_code = str(row.get('Status', '1'))
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

    # ── Practice / Qualifying driver cards data ──────────────────
    practice_driver_data: Dict[str, dict] = {}
    is_practice = session_code in ('FP1', 'FP2', 'FP3')
    is_qualifying = session_code in ('Q', 'SQ')
    if is_practice or is_qualifying:
        all_laps = session.laps
        for drv in sorted(all_laps['Driver'].dropna().unique()):
            try:
                info = session.get_driver(drv)
                abbr = info.get('Abbreviation', str(drv))
                drv_laps_all = all_laps.pick_drivers(drv)
                if drv_laps_all.empty:
                    continue

                # Compute stints
                stints = []
                prev_cmp = None
                current_stint: dict = {}
                for _, lap_row in drv_laps_all.iterrows():
                    cmp = str(lap_row.get('Compound', 'UNKNOWN')).upper()
                    lap_num = int(lap_row['LapNumber']) if pd.notna(lap_row.get('LapNumber')) else 0
                    lt = lap_row.get('LapTime')
                    fresh = bool(lap_row.get('FreshTyre', False))
                    if cmp != prev_cmp:
                        if current_stint:
                            stints.append(current_stint)
                        current_stint = {
                            'compound': cmp,
                            'startLap': lap_num,
                            'endLap': lap_num,
                            'laps': 1,
                            'fresh': fresh,
                        }
                        prev_cmp = cmp
                    else:
                        current_stint['endLap'] = lap_num
                        current_stint['laps'] = current_stint['laps'] + 1
                if current_stint:
                    stints.append(current_stint)

                # Last lap info
                last_lap_row = drv_laps_all.iloc[-1]
                last_lt = last_lap_row.get('LapTime')

                # Best lap (accurate only)
                accurate_laps = drv_laps_all[drv_laps_all['IsAccurate'] == True]
                best_lap_row = None
                best_lap_time = None
                best_lap_str = 'N/A'
                best_lap_start_trel = None
                best_lap_end_trel = None
                if not accurate_laps.empty:
                    best_lap_row = accurate_laps.loc[accurate_laps['LapTime'].idxmin()]
                    best_lt = best_lap_row.get('LapTime')
                    if pd.notna(best_lt):
                        best_lap_time = best_lt.total_seconds()
                        best_lap_str = _format_laptime(best_lt)
                        # Compute trel of best lap start/end for replay
                        if pd.notna(t0) and pd.notna(best_lap_row.get('Time')):
                            end_trel = (best_lap_row['Time'] - t0).total_seconds()
                            best_lap_end_trel = end_trel
                            best_lap_start_trel = end_trel - best_lt.total_seconds()

                # Compound counts used
                compound_counts: Dict[str, int] = {}
                for s in stints:
                    c = s['compound']
                    compound_counts[c] = compound_counts.get(c, 0) + s['laps']

                # ── Qualifying lap classification ────────────────
                qual_laps = []
                if is_qualifying:
                    # Estimate personal best sector times for "push lap" classification
                    pb_s1 = pb_s2 = pb_s3 = float('inf')
                    for _, lr in drv_laps_all.iterrows():
                        s1v = lr.get('Sector1Time')
                        s2v = lr.get('Sector2Time')
                        s3v = lr.get('Sector3Time')
                        if pd.notna(s1v): pb_s1 = min(pb_s1, s1v.total_seconds())
                        if pd.notna(s2v): pb_s2 = min(pb_s2, s2v.total_seconds())
                        if pd.notna(s3v): pb_s3 = min(pb_s3, s3v.total_seconds())

                    for _, lr in drv_laps_all.iterrows():
                        lap_num = int(lr.get('LapNumber', 0)) if pd.notna(lr.get('LapNumber')) else 0
                        lr_lt = lr.get('LapTime')
                        lr_lt_sec = lr_lt.total_seconds() if pd.notna(lr_lt) else None
                        s1 = lr.get('Sector1Time')
                        s2 = lr.get('Sector2Time')
                        s3 = lr.get('Sector3Time')
                        s1_sec = s1.total_seconds() if pd.notna(s1) else None
                        s2_sec = s2.total_seconds() if pd.notna(s2) else None
                        s3_sec = s3.total_seconds() if pd.notna(s3) else None
                        pit_in = pd.notna(lr.get('PitInTime'))
                        pit_out = pd.notna(lr.get('PitOutTime'))
                        lap_time_trel = None
                        lap_start_trel = None
                        if pd.notna(t0) and pd.notna(lr.get('Time')):
                            lap_time_trel = (lr['Time'] - t0).total_seconds()
                            if pd.notna(lr_lt):
                                lap_start_trel = lap_time_trel - lr_lt.total_seconds()

                        # Classify lap type
                        if pit_out:
                            lap_type = 'out'
                        elif pit_in:
                            lap_type = 'in'
                        elif lr_lt_sec is None:
                            lap_type = 'unknown'
                        else:
                            # "hot/push" if sectors are close to personal best
                            sector_threshold = 1.05  # within 5% of PB
                            has_good_sectors = (
                                (s1_sec is not None and pb_s1 < float('inf') and s1_sec <= pb_s1 * sector_threshold) or
                                (s2_sec is not None and pb_s2 < float('inf') and s2_sec <= pb_s2 * sector_threshold) or
                                (s3_sec is not None and pb_s3 < float('inf') and s3_sec <= pb_s3 * sector_threshold)
                            )
                            is_accurate = bool(lr.get('IsAccurate', False))
                            if is_accurate and has_good_sectors:
                                lap_type = 'hot'
                            else:
                                lap_type = 'prep'

                        qual_laps.append({
                            'lap': lap_num,
                            'lapType': lap_type,
                            'lapTime': lr_lt_sec,
                            'lapTimeStr': _format_laptime(lr_lt),
                            'lapStartTrel': lap_start_trel,
                            'lapEndTrel': lap_time_trel,
                            'compound': str(lr.get('Compound', 'UNKNOWN')).upper(),
                            'tyreLife': int(lr.get('TyreLife', 0)) if pd.notna(lr.get('TyreLife')) else 0,
                            'S1': _format_sector(s1),
                            'S2': _format_sector(s2),
                            'S3': _format_sector(s3),
                            'isAccurate': bool(lr.get('IsAccurate', False)),
                        })

                practice_driver_data[abbr] = {
                    'driver': abbr,
                    'team': info.get('TeamName', 'Unknown'),
                    'number': str(info.get('DriverNumber', drv)),
                    'color': driver_colors_map.get(abbr, {}).get('color', '#888888'),
                    'teamColor': driver_colors_map.get(abbr, {}).get('teamColor', '#888888'),
                    'headshotUrl': info.get('HeadshotUrl'),
                    'totalLaps': len(drv_laps_all),
                    'stints': stints,
                    'currentCompound': str(last_lap_row.get('Compound', 'UNKNOWN')).upper(),
                    'currentTyreLife': int(last_lap_row.get('TyreLife', 0)) if pd.notna(last_lap_row.get('TyreLife')) else 0,
                    'lastLapTime': last_lt.total_seconds() if pd.notna(last_lt) else None,
                    'lastLapStr': _format_laptime(last_lt),
                    'bestLapTime': best_lap_time,
                    'bestLapStr': best_lap_str,
                    'bestLapStartTrel': best_lap_start_trel,
                    'bestLapEndTrel': best_lap_end_trel,
                    'compoundCounts': compound_counts,
                    'qualLaps': qual_laps if is_qualifying else [],
                }
            except Exception:
                continue

    # ── Traffic incidents (qualifying only) ──────────────────────
    traffic_incidents: list = []
    if is_qualifying and is_live_supported and pd.notna(t0):
        try:
            # Build dense trel→(X,Y) lookup per driver using unprocessed data
            drv_xy: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
            all_laps2 = session.laps
            for drv in sorted(all_laps2['Driver'].dropna().unique()):
                try:
                    abbr = session.get_driver(drv).get('Abbreviation', str(drv))
                    drv_laps2 = all_laps2.pick_drivers(drv)
                    if drv_laps2.empty:
                        continue
                    tel = drv_laps2.get_telemetry()
                    if tel.empty or 'X' not in tel.columns:
                        continue
                    coords = tel[['X', 'Y']].to_numpy()
                    rotated = rotate(coords, angle=track_angle)
                    sess_times = tel['SessionTime'].dt.total_seconds().to_numpy() if 'SessionTime' in tel.columns else tel['Time'].dt.total_seconds().to_numpy()
                    trels = sess_times - t0.total_seconds()
                    dists = tel['Distance'].to_numpy()
                    drv_xy[abbr] = (trels, rotated[:, 0], rotated[:, 1], dists)
                except Exception:
                    continue

            # Classify each driver's lap type at a given trel
            def get_lap_type_at(abbr: str, trel: float) -> str:
                info = practice_driver_data.get(abbr, {})
                for ql in info.get('qualLaps', []):
                    s = ql.get('lapStartTrel')
                    e = ql.get('lapEndTrel')
                    if s is not None and e is not None and s <= trel <= e:
                        return ql['lapType']
                return 'unknown'

            # Sample every N seconds and check driver proximity
            SAMPLE_INTERVAL = 1.0  # seconds (increased frequency)
            PROXIMITY_M = 200     # metres
            COOLDOWN = 15.0       # minimum gap between incidents for same pair

            max_dur = max((arr[0][-1] for arr in drv_xy.values() if len(arr[0])), default=0.0)
            track_length = max((np.nanmax(arr[3]) for arr in drv_xy.values() if len(arr[3])), default=5000.0)
            abbrs = list(drv_xy.keys())
            last_incident: Dict[Tuple[str, str], float] = {}

            def interp_pos(abbr: str, trel: float):
                trels, xs, ys, dists = drv_xy[abbr]
                x = float(np.interp(trel, trels, xs))
                y = float(np.interp(trel, trels, ys))
                d = float(np.interp(trel, trels, dists))
                return x, y, d

            t_sample = 0.0
            while t_sample < max_dur:
                for i, a1 in enumerate(abbrs):
                    for a2 in abbrs[i + 1:]:
                        pair = (min(a1, a2), max(a1, a2))
                        # Check cooldown
                        if last_incident.get(pair, -999) + COOLDOWN > t_sample:
                            continue
                        try:
                            x1, y1, d1 = interp_pos(a1, t_sample)
                            x2, y2, d2 = interp_pos(a2, t_sample)
                            euc_dist = np.hypot(x1 - x2, y1 - y2)
                            
                            dist_diff = abs(d1 - d2)
                            dist_diff = min(dist_diff, track_length - dist_diff)

                            if euc_dist < PROXIMITY_M and dist_diff < PROXIMITY_M:
                                lt1 = get_lap_type_at(a1, t_sample)
                                lt2 = get_lap_type_at(a2, t_sample)
                                # Determine which is push, which is prep (if applicable)
                                push_drv = None
                                prep_drv = None
                                if lt1 == 'hot' and lt2 in ('prep', 'out', 'in'):
                                    push_drv, prep_drv = a1, a2
                                elif lt2 == 'hot' and lt1 in ('prep', 'out', 'in'):
                                    push_drv, prep_drv = a2, a1

                                # Determine traffic color/severity
                                if push_drv and prep_drv:
                                    severity = 'yellow'  # default: possible interference
                                    color = '#facc15'
                                elif lt1 == 'hot' and lt2 == 'hot':
                                    severity = 'blue'    # two push laps passing
                                    color = '#60a5fa'
                                else:
                                    severity = 'green'   # both prep, no issue
                                    color = '#4ade80'

                                if severity != 'green':
                                    traffic_incidents.append({
                                        'trel': t_sample,
                                        'driver1': a1,
                                        'driver2': a2,
                                        'x': (x1 + x2) / 2,
                                        'y': (y1 + y2) / 2,
                                        'severity': severity,
                                        'color': color,
                                        'lapType1': lt1,
                                        'lapType2': lt2,
                                        'pushDriver': push_drv,
                                        'prepDriver': prep_drv,
                                        'distance': float(euc_dist),
                                    })
                                    last_incident[pair] = t_sample
                        except Exception:
                            pass
                t_sample += SAMPLE_INTERVAL
        except Exception:
            pass

    return lap_times, driver_tel, fastest_lap_tel, driver_colors_map, max_duration, is_live_supported, total_laps, lap_history, track_status_events, practice_driver_data, traffic_incidents

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
        lap_times, driver_tel, fastest_lap_tel, driver_colors_map, max_duration, is_live_supported, total_laps, lap_history, track_status_events, practice_driver_data, traffic_incidents = _build_session_data(
            year, gp, session_code, qual_session
        )
        return {
            "leaderboard": lap_times,
            "telemetry": driver_tel,
            "fastestLapTelemetry": fastest_lap_tel,
            "driverColors": driver_colors_map,
            "maxDuration": max_duration,
            "isRace": session_code in ('R', 'S'),
            "totalLaps": total_laps,
            "lapHistory": lap_history,
            "trackStatusEvents": track_status_events,
            "practiceDriverData": practice_driver_data if practice_driver_data else None,
            "trafficIncidents": traffic_incidents if traffic_incidents else [],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/time-delta")
def get_time_delta(year: int, gp: str, session_code: str, qual_session: Optional[str], driver1: str, driver2: str):
    try:
        lap_times, _, fastest_lap_tel, _, _, _, _, _, _, _, _ = _build_session_data(year, gp, session_code, qual_session)
        if driver1 not in fastest_lap_tel or driver2 not in fastest_lap_tel:
            raise HTTPException(status_code=404, detail="Driver telemetry not found")

        tel1 = fastest_lap_tel[driver1]
        tel2 = fastest_lap_tel[driver2]

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
