"""
F1 Telemetry Analysis Dashboard
================================
Features:
- Race Animation with all 20 drivers
- Lap Analysis with time delta comparison
- Session selection (Race/Q1/Q2/Q3)
- Tyre information with F1 symbols
- Sector-colored track heatmaps
- Realistic F1-style timing tower
"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import streamlit as st
st.set_page_config(page_title="F1 Telemetry Analysis", layout="wide", page_icon="🏎️")

import fastf1
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Tuple, List, Optional
import os

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
DATA_DIR = './data'
CACHE_DIR = './cache'
fastf1.Cache.enable_cache(CACHE_DIR)

# Team colors (primary, secondary)
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
    # 2023/2024 names
    'AlphaTauri': ('#6692FF', '#FFFFFF'),
    'Alfa Romeo': ('#C92D4B', '#A12239'),
}

# Tyre compound colors and symbols
TYRE_COMPOUNDS = {
    'SOFT': {'color': '#FF3333', 'symbol': '🔴', 'short': 'S', 'html': '<span style="color:#FF3333;font-size:18px;">●</span>'},
    'MEDIUM': {'color': '#FFD700', 'symbol': '🟡', 'short': 'M', 'html': '<span style="color:#FFD700;font-size:18px;">●</span>'},
    'HARD': {'color': '#FFFFFF', 'symbol': '⚪', 'short': 'H', 'html': '<span style="color:#FFFFFF;font-size:18px;">●</span>'},
    'INTERMEDIATE': {'color': '#43B02A', 'symbol': '🟢', 'short': 'I', 'html': '<span style="color:#43B02A;font-size:18px;">●</span>'},
    'WET': {'color': '#0067AD', 'symbol': '🔵', 'short': 'W', 'html': '<span style="color:#0067AD;font-size:18px;">●</span>'},
}

# Sector colors
SECTOR_COLORS = {
    'S1': '#FF5555',  # Red
    'S2': '#55FF55',  # Green
    'S3': '#5555FF',  # Blue
}

# ─────────────────────────────────────────────────────────────────
# DATA LOADING FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def rotate(xy, *, angle):
    rot_mat = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ])
    return np.matmul(xy, rot_mat)

def generate_track_data(year: int, gp_name: str, session_type: str = 'Q'):
    """Fetch and generate track data dynamically from FastF1 if CSVs don't exist."""
    track_file = os.path.join(DATA_DIR, f'{gp_name}_{year}_track_layout.csv')
    corners_file = os.path.join(DATA_DIR, f'{gp_name}_{year}_corners.csv')
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Load session to get track info
    session = fastf1.get_session(year, gp_name, session_type)
    session.load(weather=False, messages=False)
    
    lap = session.laps.pick_fastest()
    pos = lap.get_telemetry()
    circuit_info = session.get_circuit_info()
    
    # Track Processing
    track = pos.loc[:, ('X', 'Y')].to_numpy()
    track_angle = circuit_info.rotation / 180 * np.pi
    rotated_track = rotate(track, angle=track_angle)
    
    track_df = pd.DataFrame({
        'x': rotated_track[:, 0],
        'y': rotated_track[:, 1],
        'distance': pos['Distance'].to_numpy()
    })
    
    # Corner Processing
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
        
        corner_x.append(final_corner[0])
        corner_y.append(final_corner[1])
        label_x.append(final_text[0])
        label_y.append(final_text[1])
        
        # Find closest distance in telemetry
        dist = row.get('Distance', 0)
        corner_distances.append(dist)

    corners_df = pd.DataFrame({
        'Number': corners['Number'],
        'Letter': corners['Letter'],
        'distance': corner_distances,
        'x': corner_x,
        'y': corner_y,
        'label_x': label_x,
        'label_y': label_y
    })

    track_df.to_csv(track_file, index=False)
    corners_df.to_csv(corners_file, index=False)
    
    return track_df, corners_df

@st.cache_data
def get_grand_prix_list(year: int):
    try:
        schedule = fastf1.get_event_schedule(year)
        schedule = schedule[schedule['EventFormat'] != 'testing']
        return schedule['EventName'].tolist()
    except Exception:
        return ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Australian Grand Prix', 'Japanese Grand Prix', 'Chinese Grand Prix', 'Miami Grand Prix', 'Emilia Romagna Grand Prix', 'Monaco Grand Prix', 'Canadian Grand Prix', 'Spanish Grand Prix', 'Austrian Grand Prix', 'British Grand Prix', 'Hungarian Grand Prix', 'Belgian Grand Prix', 'Dutch Grand Prix', 'Italian Grand Prix', 'Azerbaijan Grand Prix', 'Singapore Grand Prix', 'United States Grand Prix', 'Mexico City Grand Prix', 'São Paulo Grand Prix', 'Las Vegas Grand Prix', 'Qatar Grand Prix', 'Abu Dhabi Grand Prix']

@st.cache_data
def load_track_data(year: int, gp_name: str, session_type: str = 'Q'):
    """Load track layout and corners data, generating dynamically if needed."""
    track_file = os.path.join(DATA_DIR, f'{gp_name}_{year}_track_layout.csv')
    corners_file = os.path.join(DATA_DIR, f'{gp_name}_{year}_corners.csv')
    
    if os.path.exists(track_file) and os.path.exists(corners_file):
        track_df = pd.read_csv(track_file)
        corners_df = pd.read_csv(corners_file)
        return track_df, corners_df
        
    try:
        # User requested accurate wait reasons at every point
        with st.spinner(f"Track data not found locally. Fetching {gp_name} {year} layout from FastF1... (This may take up to 30 seconds)"):
            track_df, corners_df = generate_track_data(year, gp_name, session_type)
        return track_df, corners_df
    except Exception as e:
        # If it fails (e.g., a future race that hasn't happened), fallback to previous years to get the layout
        for fallback_year in [year - 1, year - 2]:
            try:
                with st.spinner(f"Session not found for {year}. Falling back to {fallback_year} {gp_name} layout..."):
                    track_df, corners_df = generate_track_data(fallback_year, gp_name, session_type)
                return track_df, corners_df
            except Exception:
                continue
        st.error(f"Failed to generate track data for {gp_name} dynamically. Error: {e}")
        return None, None


@st.cache_data
def load_session(_year: int, _gp_name: str, _session_type: str):
    """Load FastF1 session with caching."""
    session = fastf1.get_session(_year, _gp_name, _session_type)
    session.load()
    return session

def get_driver_team_colors(_session) -> Dict[str, Tuple[str, str]]:
    """Map each driver to their team color."""
    driver_map = {}
    laps = _session.laps
    drivers = sorted(laps['Driver'].dropna().unique())
    
    team_drivers = {}
    for drv in drivers:
        try:
            drv_info = _session.get_driver(drv)
            team = drv_info.get('TeamName', 'Unknown')
            if team not in team_drivers:
                team_drivers[team] = []
            team_drivers[team].append(drv)
        except:
            continue
    
    for team, drvs in team_drivers.items():
        colors = TEAM_COLORS.get(team, ('#888888', '#AAAAAA'))
        for i, drv in enumerate(drvs):
            driver_map[drv] = (colors[i % 2], team)
    
    return driver_map

def build_telemetry_data(_session, _session_type: str, _qual_session: Optional[str] = None):
    """
    Build telemetry data for all drivers.
    Returns: leaderboard_df, driver_tel dict, driver_colors_map, max_duration
    """
    driver_colors_map = get_driver_team_colors(_session)
    driver_tel: Dict[str, pd.DataFrame] = {}
    lap_times: List[dict] = []
    
    laps = _session.laps
    
    # Filter laps by qualifying session if specified
    if _qual_session and _session_type == 'Q':
        try:
            q_splits = laps.split_qualifying_sessions()
            session_map = {'Q1': 0, 'Q2': 1, 'Q3': 2}
            idx = session_map.get(_qual_session, 2)
            if idx < len(q_splits):
                laps = q_splits[idx]
        except Exception as e:
            pass # Fallback to all laps if splitting fails
            
    laps = laps[laps['IsAccurate'] == True]
    
    drivers = sorted(laps['Driver'].dropna().unique())
    
    for drv in drivers:
        try:
            drv_laps = laps.pick_driver(drv)
            if drv_laps.empty:
                continue
            
            # Get driver info
            drv_info = _session.get_driver(drv)
            team = drv_info.get('TeamName', 'Unknown')
            
            # Get best lap for telemetry
            best_lap = drv_laps.pick_fastest()
            if best_lap is None or pd.isna(best_lap['LapTime']):
                continue
            
            # Get telemetry
            tel = best_lap.get_telemetry()
            if tel.empty:
                continue
            
            # Convert time to relative seconds
            t0 = tel['Time'].iloc[0]
            tel = tel.copy()
            tel['Trel'] = (tel['Time'] - t0).dt.total_seconds()
            tel['Distance'] = tel['Distance'].astype(float)
            
            # Get tyre compound
            compound = best_lap.get('Compound', 'MEDIUM')
            tyre_life = best_lap.get('TyreLife', 0)
            
            # Sector times
            s1 = best_lap.get('Sector1Time', pd.NaT)
            s2 = best_lap.get('Sector2Time', pd.NaT)
            s3 = best_lap.get('Sector3Time', pd.NaT)
            
            # Keep only needed columns
            cols = ['Distance', 'Speed', 'Trel', 'Throttle', 'Brake', 'RPM', 'nGear', 'DRS']
            available = [c for c in cols if c in tel.columns]
            driver_tel[drv] = tel[available].reset_index(drop=True)
            
            # Format lap time
            lt = best_lap['LapTime']
            if pd.notna(lt):
                total_sec = lt.total_seconds()
                mins = int(total_sec // 60)
                secs = total_sec % 60
                lap_str = f"{mins}:{secs:06.3f}"
            else:
                lap_str = 'N/A'
                total_sec = float('inf')
            
            # Format sector times
            def format_sector(s):
                if pd.isna(s):
                    return 'N/A'
                return f"{s.total_seconds():.3f}"
            
            lap_times.append({
                'Driver': drv,
                'Team': team,
                'LapTime': total_sec,
                'BestLapStr': lap_str,
                'Compound': str(compound).upper() if pd.notna(compound) else 'MEDIUM',
                'TyreLife': int(tyre_life) if pd.notna(tyre_life) else 0,
                'S1': format_sector(s1),
                'S2': format_sector(s2),
                'S3': format_sector(s3),
                'S1_sec': s1.total_seconds() if pd.notna(s1) else float('inf'),
                'S2_sec': s2.total_seconds() if pd.notna(s2) else float('inf'),
                'S3_sec': s3.total_seconds() if pd.notna(s3) else float('inf'),
            })
            
            # Set driver colors
            if drv not in driver_colors_map:
                colors = TEAM_COLORS.get(team, ('#888888', '#FFFFFF'))
                driver_colors_map[drv] = (colors[0], team)
                
        except Exception as e:
            continue
    
    # Sort by lap time
    leaderboard_df = pd.DataFrame(lap_times).sort_values('LapTime').reset_index(drop=True)
    leaderboard_df['Position'] = range(1, len(leaderboard_df) + 1)
    
    # Calculate gap to leader
    if not leaderboard_df.empty:
        leader_time = leaderboard_df.iloc[0]['LapTime']
        leaderboard_df['Gap'] = leaderboard_df['LapTime'].apply(
            lambda x: '' if x == leader_time else f"+{x - leader_time:.3f}"
        )
    
    # Find best sectors
    if not leaderboard_df.empty:
        best_s1 = leaderboard_df['S1_sec'].min()
        best_s2 = leaderboard_df['S2_sec'].min()
        best_s3 = leaderboard_df['S3_sec'].min()
        
        leaderboard_df['S1_best'] = leaderboard_df['S1_sec'] == best_s1
        leaderboard_df['S2_best'] = leaderboard_df['S2_sec'] == best_s2
        leaderboard_df['S3_best'] = leaderboard_df['S3_sec'] == best_s3
    
    # Max duration for animation
    max_duration = max((tel['Trel'].iloc[-1] for tel in driver_tel.values()), default=0.0)
    
    return leaderboard_df, driver_tel, driver_colors_map, max_duration

def calculate_time_delta(tel1: pd.DataFrame, tel2: pd.DataFrame, driver1: str, driver2: str):
    """
    Calculate time delta between two drivers based on distance.
    Returns: distance array, delta array (positive = driver1 ahead)
    """
    max_dist = min(tel1['Distance'].max(), tel2['Distance'].max())
    distance_grid = np.linspace(0, max_dist, 1000)
    
    time1 = np.interp(distance_grid, tel1['Distance'], tel1['Trel'])
    time2 = np.interp(distance_grid, tel2['Distance'], tel2['Trel'])
    
    delta = time1 - time2
    
    return distance_grid, delta

def get_sector_boundaries(track_df: pd.DataFrame) -> Tuple[float, float]:
    """Calculate sector boundaries based on track length."""
    total_distance = track_df['distance'].max()
    s1_end = total_distance * 0.33
    s2_end = total_distance * 0.67
    return s1_end, s2_end

# ─────────────────────────────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────────────────────────────

def render_timing_tower(leaderboard_df: pd.DataFrame, driver_colors_map: Dict):
    """Render F1-style timing tower using HTML with tyre images."""
    import base64
    
    def get_base64_img(path):
        try:
            from PIL import Image
            import io
            if os.path.exists(path):
                # Resize the heavy PNGs in memory to avoid crashing the Streamlit websocket
                with Image.open(path) as img:
                    img.thumbnail((32, 32))
                    buffer = io.BytesIO()
                    img.save(buffer, format="PNG")
                    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
            return ""
        except:
            return ""
            
    # Tyre image mapping
    tyre_img_map = {
        'SOFT': get_base64_img('images/Soft.png'),
        'MEDIUM': get_base64_img('images/Medium.png'),
        'HARD': get_base64_img('images/Hard.png'),
        'INTERMEDIATE': get_base64_img('images/Intermediate.png'),
        'WET': get_base64_img('images/Wet.png')
    }
    
    html_content = """
    <style>
    .timing-tower {
        background: linear-gradient(180deg, #0a0a0f 0%, #15151e 100%);
        border-radius: 10px;
        padding: 12px;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .timing-row {
        display: flex;
        align-items: center;
        padding: 6px 8px;
        border-radius: 4px;
        margin: 2px 0;
        background: rgba(255,255,255,0.02);
        transition: all 0.2s ease;
        border-left: 3px solid transparent;
    }
    .timing-row:hover {
        background: rgba(255,255,255,0.06);
        transform: translateX(2px);
    }
    .pos-badge {
        min-width: 26px;
        height: 26px;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 13px;
        margin-right: 8px;
    }
    .driver-abbr {
        font-weight: 700;
        font-size: 15px;
        color: #fff;
        min-width: 50px;
        letter-spacing: 0.5px;
    }
    .sector-time {
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 11px;
        padding: 3px 6px;
        border-radius: 3px;
        margin: 0 2px;
        min-width: 50px;
        text-align: center;
        background: rgba(255,255,255,0.05);
        color: #bbb;
    }
    .sector-time.best {
        background: #a855f7;
        color: #fff;
        font-weight: 600;
    }
    .laptime {
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 13px;
        color: #10b981;
        font-weight: 600;
        min-width: 85px;
        margin-left: 6px;
    }
    .gap {
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 12px;
        color: #ef4444;
        min-width: 70px;
        text-align: right;
        margin-left: auto;
    }
    .tyre-img {
        width: 22px;
        height: 22px;
        margin-left: 8px;
        border-radius: 3px;
        object-fit: cover;
    }
    </style>
    <div class="timing-tower">
    """
    
    for _, row in leaderboard_df.iterrows():
        driver = row['Driver']
        pos = row['Position']
        team_color = driver_colors_map.get(driver, ('#888888', 'Unknown'))[0]
        compound = row.get('Compound', 'MEDIUM')
        tyre_img = tyre_img_map.get(compound, tyre_img_map.get('MEDIUM', ''))
        
        # Position badge colors
        if pos == 1:
            pos_style = 'background:#ffd700;color:#000;'
        elif pos == 2:
            pos_style = 'background:#c0c0c0;color:#000;'
        elif pos == 3:
            pos_style = 'background:#cd7f32;color:#000;'
        else:
            pos_style = 'background:rgba(100,100,100,0.4);color:#fff;'
        
        # Sector classes
        s1_class = 'best' if row.get('S1_best', False) else ''
        s2_class = 'best' if row.get('S2_best', False) else ''
        s3_class = 'best' if row.get('S3_best', False) else ''
        
        html_content += f"""
        <div class="timing-row" style="border-left-color:{team_color};">
            <div class="pos-badge" style="{pos_style}">{pos}</div>
            <span class="driver-abbr">{driver}</span>
            <span class="sector-time {s1_class}">{row['S1']}</span>
            <span class="sector-time {s2_class}">{row['S2']}</span>
            <span class="sector-time {s3_class}">{row['S3']}</span>
            <span class="laptime">{row['BestLapStr']}</span>
            <span class="gap">{row['Gap']}</span>
            <img src="{tyre_img}" class="tyre-img" alt="{compound}" />
        </div>
        """
    
    html_content += "</div>"
    
    # Use components.html for better HTML rendering
    import streamlit.components.v1 as components
    components.html(html_content, height=600, scrolling=True)

# ─────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────

st.title("🏎️ F1 Telemetry Analysis")

st.markdown("### 📅 Session Selection")
col_year, col_gp, col_type, col_qual = st.columns([1, 2, 1, 2])

with col_year:
    year_sel = st.selectbox("Year", [2026, 2025, 2024], index=1)
with col_gp:
    gp_list = get_grand_prix_list(year_sel)
    gp_sel = st.selectbox("Grand Prix", gp_list, index=0)

with col_type:
    type_sel = st.radio(
        "Session Type",
        ["Qualifying", "Race"],
        index=0
    )

qual_sel = None
with col_qual:
    if type_sel == "Qualifying":
        qual_sel = st.radio(
            "Qualifying Session",
            ["Q1", "Q2", "Q3"],
            index=2,
            horizontal=True
        )
        code_sel = 'Q'
    else:
        code_sel = 'R'

if st.button("Load Session Data", type="primary"):
    st.session_state['loaded_year'] = year_sel
    st.session_state['loaded_gp'] = gp_sel
    st.session_state['loaded_type'] = type_sel
    st.session_state['loaded_qual'] = qual_sel
    st.session_state['loaded_code'] = code_sel

if 'loaded_year' not in st.session_state:
    st.info("👆 Please select a session and click 'Load Session Data' to begin.")
    st.stop()

year = st.session_state['loaded_year']
gp_name = st.session_state['loaded_gp']
session_type_display = st.session_state['loaded_type']
qual_session = st.session_state['loaded_qual']
session_code = st.session_state['loaded_code']

with st.sidebar:
    st.markdown("### 🏁 Legend")
    st.markdown("**Tyres:**")
    for compound, img_file in [("Soft", "Soft.png"), ("Medium", "Medium.png"), ("Hard", "Hard.png")]:
        st.image(f"images/{img_file}", width=40, caption=compound)
    
    st.markdown("**Sectors:**")
    st.markdown("S1 / S2 / S3")
    st.markdown("🟣 = Best sector time")

# Load track data
with st.spinner(f"Loading track layout for {gp_name}..."):
    track_df, corners_df = load_track_data(year, gp_name, session_code)

if track_df is None:
    st.error(f"Track layout data could not be generated for {gp_name}. FastF1 may not have data for this circuit yet.")
    st.stop()

# Calculate sector boundaries
s1_end, s2_end = get_sector_boundaries(track_df)

# Load session data
qual_session_display = qual_session if session_type_display == "Qualifying" else "Race"
with st.spinner(f"Loading {gp_name} {year} {qual_session_display} session data..."):
    try:
        session = load_session(year, gp_name, session_code)
        qual_filter = qual_session if session_type_display == "Qualifying" else None
        leaderboard_df, driver_tel, driver_colors_map, max_duration = build_telemetry_data(session, session_code, qual_filter)
    except Exception as e:
        st.error(f"Failed to load session data. The session may not have occurred yet or FastF1 cannot access it. Error: {e}")
        st.stop()

if leaderboard_df.empty:
    st.warning("No lap data available for this session.")
    st.stop()

drv_list = list(driver_tel.keys())
driver_colors = {drv: driver_colors_map[drv][0] for drv in drv_list}

# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs(["🏁 Race Animation", "📊 Lap Analysis", "⏱️ Time Delta"])

# ─────────────────────────────────────────────────────────────────
# TAB 1: RACE ANIMATION
# ─────────────────────────────────────────────────────────────────
with tab1:
    col_track, col_timing = st.columns([75, 25])  # 75/25 split for bigger track
    
    with col_track:
        st.markdown("### Live Animation")
        
        fps = 10
        dt = 1.0 / fps
        t_grid = np.arange(0.0, max_duration, dt)
        
        dist_to_x = lambda d: np.interp(d, track_df['distance'], track_df['x'])
        dist_to_y = lambda d: np.interp(d, track_df['distance'], track_df['y'])
        
        fig = go.Figure()
        
        # Grey track
        fig.add_trace(go.Scatter(
            x=track_df['x'], y=track_df['y'], mode='lines',
            line=dict(color='#888888', width=6), hoverinfo='none', showlegend=False
        ))
        
        # Corner labels
        if corners_df is not None:
            lbl_x = corners_df['label_x'] if 'label_x' in corners_df.columns else corners_df['x']
            lbl_y = corners_df['label_y'] if 'label_y' in corners_df.columns else corners_df['y']
            fig.add_trace(go.Scatter(
                x=lbl_x, y=lbl_y, mode='text',
                text=corners_df['Number'].astype(str) + corners_df['Letter'].fillna(''),
                textfont=dict(size=9, color='white'), hoverinfo='none', showlegend=False
            ))
        
        # Driver markers
        dynamic_traces_start = len(fig.data)
        for drv in drv_list:
            color = driver_colors[drv]
            tel = driver_tel[drv]
            d0 = float(tel['Distance'].iloc[0])
            fig.add_trace(go.Scatter(
                x=[dist_to_x(d0)], y=[dist_to_y(d0)], mode='markers+text',
                marker=dict(size=12, color=color, line=dict(width=1, color='white')),
                text=[drv], textposition='top center', textfont=dict(size=8, color='white'),
                showlegend=False, hovertemplate=f'{drv}<extra></extra>'
            ))
        
        # Build frames
        frames = []
        for ti in t_grid:
            frame_data = []
            for drv in drv_list:
                tel = driver_tel[drv]
                d = float(np.interp(ti, tel['Trel'], tel['Distance'], 
                                    left=tel['Distance'].iloc[0], right=tel['Distance'].iloc[-1]))
                frame_data.append(go.Scatter(x=[dist_to_x(d)], y=[dist_to_y(d)]))
            frames.append(go.Frame(
                data=frame_data,
                traces=list(range(dynamic_traces_start, dynamic_traces_start + len(drv_list))),
                name=f"t={ti:.2f}"
            ))
        
        fig.frames = frames
        
        pad = 100
        fig.update_layout(
            height=850,
            plot_bgcolor='rgba(15,15,25,1)',
            paper_bgcolor='rgba(15,15,25,1)',
            xaxis=dict(range=[track_df['x'].min()-pad, track_df['x'].max()+pad], visible=False, fixedrange=False),
            yaxis=dict(range=[track_df['y'].min()-pad, track_df['y'].max()+pad], visible=False, fixedrange=False, scaleanchor='x'),
            margin=dict(l=20, r=0, t=40, b=20),
            updatemenus=[dict(
                type='buttons', direction='left', x=0.02, y=1.02, xanchor='left', yanchor='bottom',
                buttons=[
                    dict(label='▶ Play', method='animate', 
                         args=[None, dict(frame=dict(duration=int(1000*dt), redraw=False), 
                                         transition=dict(duration=int(1000*dt*0.8), easing="linear"), fromcurrent=True)]),
                    dict(label='2x', method='animate', 
                         args=[None, dict(frame=dict(duration=int(1000*dt/2), redraw=False), 
                                         transition=dict(duration=int(1000*dt/2*0.8), easing="linear"), fromcurrent=True)]),
                    dict(label='4x', method='animate', 
                         args=[None, dict(frame=dict(duration=int(1000*dt/4), redraw=False), 
                                         transition=dict(duration=int(1000*dt/4*0.8), easing="linear"), fromcurrent=True)]),
                    dict(label='⏸ Pause', method='animate', 
                         args=[[None], dict(frame=dict(duration=0, redraw=False), 
                                           mode='immediate', transition=dict(duration=0))])
                ],
                bgcolor='rgba(20,20,30,0.9)', 
                font=dict(color='white', size=11),
                bordercolor='rgba(100,100,120,0.5)',
                borderwidth=1,
                pad=dict(t=5, b=5, l=8, r=8)
            )]
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col_timing:
        st.markdown("### Timing Tower")
        render_timing_tower(leaderboard_df, driver_colors_map)

# ─────────────────────────────────────────────────────────────────
# TAB 2: LAP ANALYSIS
# ─────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Compare Driver Laps")
    
    if session_type_display != "Qualifying":
        st.warning("Lap Analysis is currently only available for Qualifying sessions.")
    else:
        with st.form("lap_analysis_form"):
    
            col_select, col_telemetry = st.columns([1, 1])
    
            with col_select:
                safe_defaults = sorted(drv_list)[:2] if len(drv_list) >= 2 else sorted(drv_list)
                selected_drivers = st.multiselect(
                    "Select drivers (max 4):",
                    options=sorted(drv_list),
                    default=safe_defaults,
                    max_selections=4
                )
    
            with col_telemetry:
                telemetry_channels = st.multiselect(
                    "Telemetry channels (max 2):",
                    options=['Speed', 'Throttle', 'Brake', 'RPM', 'Gear', 'DRS'],
                    default=['Speed'],
                    max_selections=2
                )
    
            submit_lap = st.form_submit_button("Load Analysis")
        
        valid_drivers = [d for d in selected_drivers if d in drv_list]
        if not valid_drivers:
            st.info("Please select at least one driver to analyze.")
        else:
            selected_drivers = valid_drivers
            max_selected_duration = max(driver_tel[drv]['Trel'].iloc[-1] for drv in selected_drivers)
            max_distance = max(driver_tel[drv]['Distance'].max() for drv in selected_drivers)
        
            fps = 10
            dt = 1.0 / fps
            t_grid = np.arange(0.0, max_selected_duration, dt)
        
            dist_to_x = lambda d: np.interp(d, track_df['distance'], track_df['x'])
            dist_to_y = lambda d: np.interp(d, track_df['distance'], track_df['y'])
        
            n_tel = len(telemetry_channels)
            n_rows = 1 + n_tel
            row_heights = [0.4] + [0.6/max(n_tel,1)] * n_tel if n_tel > 0 else [1.0]
        
            fig_lap = make_subplots(
                rows=n_rows, cols=1,
                row_heights=row_heights,
                specs=[[{"type": "xy"}]] * n_rows,
                vertical_spacing=0.08
            )
        
            # Track (grey)
            fig_lap.add_trace(go.Scatter(
                x=track_df['x'], y=track_df['y'], mode='lines',
                line=dict(color='#888888', width=5), hoverinfo='none', showlegend=False
            ), row=1, col=1)
        
            # Corner labels
            if corners_df is not None:
                lbl_x = corners_df['label_x'] if 'label_x' in corners_df.columns else corners_df['x']
                lbl_y = corners_df['label_y'] if 'label_y' in corners_df.columns else corners_df['y']
                fig_lap.add_trace(go.Scatter(
                    x=lbl_x, y=lbl_y, mode='text',
                    text=corners_df['Number'].astype(str) + corners_df['Letter'].fillna(''),
                    textfont=dict(size=9, color='white'), hoverinfo='none', showlegend=False
                ), row=1, col=1)
        
            # Driver markers
            track_marker_start = len(fig_lap.data)
            for drv in selected_drivers:
                color = driver_colors[drv]
                tel = driver_tel[drv]
                d0 = float(tel['Distance'].iloc[0])
                fig_lap.add_trace(go.Scatter(
                    x=[dist_to_x(d0)], y=[dist_to_y(d0)], mode='markers',
                    marker=dict(size=14, color=color, line=dict(width=2, color='white')),
                    name=drv, showlegend=True
                ), row=1, col=1)
        
            # Telemetry traces
            tel_col_map = {'Speed': 'Speed', 'Throttle': 'Throttle', 'Brake': 'Brake', 
                           'RPM': 'RPM', 'Gear': 'nGear', 'DRS': 'DRS'}
            vline_indices = []
        
            for tel_idx, tel_name in enumerate(telemetry_channels):
                row_num = 2 + tel_idx
                col_name = tel_col_map.get(tel_name, 'Speed')
            
                for drv in selected_drivers:
                    tel = driver_tel[drv]
                    color = driver_colors[drv]
                    y_data = tel[col_name] if col_name in tel.columns else tel['Speed']
                
                    fig_lap.add_trace(go.Scatter(
                        x=tel['Distance'], y=y_data, mode='lines',
                        line=dict(color=color, width=2), name=f"{drv} {tel_name}", showlegend=False
                    ), row=row_num, col=1)
            
                vline_idx = len(fig_lap.data)
                vline_indices.append(vline_idx)
                fig_lap.add_trace(go.Scatter(
                    x=[0, 0], y=[0, 500], mode='lines',
                    line=dict(color='white', width=2), showlegend=False
                ), row=row_num, col=1)
            
                fig_lap.update_yaxes(title_text=tel_name, row=row_num, col=1, gridcolor='rgba(255,255,255,0.1)')
        
            # Build frames
            frames_lap = []
            for ti in t_grid:
                frame_data = []
                current_dist = 0
            
                for drv in selected_drivers:
                    tel = driver_tel[drv]
                    d = float(np.interp(ti, tel['Trel'], tel['Distance'],
                                       left=tel['Distance'].iloc[0], right=tel['Distance'].iloc[-1]))
                    current_dist = max(current_dist, d)
                    frame_data.append(go.Scatter(x=[dist_to_x(d)], y=[dist_to_y(d)]))
            
                for tel_idx, tel_name in enumerate(telemetry_channels):
                    col_name = tel_col_map.get(tel_name, 'Speed')
                    y_vals = []
                    for drv in selected_drivers:
                        tel = driver_tel[drv]
                        if col_name in tel.columns:
                            y_vals.extend(tel[col_name].dropna().tolist())
                    y_min, y_max = (min(y_vals), max(y_vals)) if y_vals else (0, 100)
                    frame_data.append(go.Scatter(x=[current_dist, current_dist], y=[y_min, y_max]))
            
                trace_indices = list(range(track_marker_start, track_marker_start + len(selected_drivers))) + vline_indices
                frames_lap.append(go.Frame(data=frame_data, traces=trace_indices, name=f"t={ti:.2f}"))
        
            fig_lap.frames = frames_lap
        
            pad = 50
            total_height = 450 + (200 * n_tel)
            fig_lap.update_layout(
                height=total_height,
                plot_bgcolor='rgba(15,15,25,1)',
                paper_bgcolor='rgba(15,15,25,1)',
                xaxis=dict(range=[track_df['x'].min()-pad, track_df['x'].max()+pad], visible=False, fixedrange=True),
                yaxis=dict(range=[track_df['y'].min()-pad, track_df['y'].max()+pad], visible=False, fixedrange=True, scaleanchor='x'),
                legend=dict(orientation='h', x=0, y=-0.02, font=dict(color='white')),
                margin=dict(l=50, r=20, t=30, b=40),
                updatemenus=[dict(
                    type='buttons', direction='left', x=0.02, y=1.02, xanchor='left', yanchor='bottom',
                    buttons=[
                        dict(label='▶ Play', method='animate', 
                             args=[None, dict(frame=dict(duration=int(1000*dt), redraw=True), 
                                             transition=dict(duration=int(1000*dt*0.8), easing="linear"), fromcurrent=True)]),
                        dict(label='2x', method='animate', 
                             args=[None, dict(frame=dict(duration=int(1000*dt/2), redraw=True), 
                                             transition=dict(duration=int(1000*dt/2*0.8), easing="linear"), fromcurrent=True)]),
                        dict(label='⏸ Pause', method='animate', 
                             args=[[None], dict(frame=dict(duration=0, redraw=False), 
                                               mode='immediate', transition=dict(duration=0))])
                    ],
                    bgcolor='rgba(20,20,30,0.9)', 
                    font=dict(color='white', size=11),
                    bordercolor='rgba(100,100,120,0.5)',
                    borderwidth=1,
                    pad=dict(t=5, b=5, l=8, r=8)
                )]
            )
        
            for i in range(n_tel):
                fig_lap.update_xaxes(range=[0, max_distance], row=2+i, col=1)
        
            st.plotly_chart(fig_lap, use_container_width=True)
        
            # Summary stats with tyre images
            st.markdown("#### Summary Statistics")
            for drv in selected_drivers:
                tel = driver_tel[drv]
                row_data = leaderboard_df[leaderboard_df['Driver'] == drv]
                if not row_data.empty:
                    row_data = row_data.iloc[0]
                    compound = row_data.get('Compound', 'MEDIUM')
                    tyre_img_map = {
                        'SOFT': 'images/Soft.png',
                        'MEDIUM': 'images/Medium.png',
                        'HARD': 'images/Hard.png',
                        'INTERMEDIATE': 'images/Intermediate.png',
                        'WET': 'images/Wet.png'
                    }
                    tyre_img = tyre_img_map.get(compound, 'images/Medium.png')
                
                    col_img, col_stats = st.columns([1, 4])
                    with col_img:
                        st.image(tyre_img, width=50)
                    with col_stats:
                        st.markdown(f"**{drv}** - {driver_colors_map[drv][1]}")
                        st.text(f"Lap: {row_data['BestLapStr']} | S1: {row_data['S1']} | S2: {row_data['S2']} | S3: {row_data['S3']}")
                        st.text(f"Max Speed: {tel['Speed'].max():.0f} km/h")
                    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────
    # TAB 3: TIME DELTA
    # ─────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Time Delta Analysis")
    st.markdown("Compare pace difference between two drivers throughout the lap.")
    
    if session_type_display != "Qualifying":
        st.warning("Time Delta Analysis is currently only available for Qualifying sessions.")
    else:
        with st.form("time_delta_form"):
    
            col1, col2 = st.columns(2)
    
            with col1:
                driver1 = st.selectbox("Driver 1 (Reference)", options=sorted(drv_list), index=0, key="delta_drv1")
    
            with col2:
                # Use the full list to avoid Streamlit form state issues with dynamic options
                driver2 = st.selectbox("Driver 2 (Comparison)", options=sorted(drv_list), index=1 if len(drv_list) > 1 else 0, key="delta_drv2")
    
            submit_delta = st.form_submit_button("Load Analysis")
        
        if driver1 in drv_list and driver2 in drv_list:
            if driver1 == driver2:
                st.info("Please select two different drivers to compare.")
                
        if driver1 in drv_list and driver2 in drv_list and driver1 != driver2:
            tel1 = driver_tel[driver1]
            tel2 = driver_tel[driver2]
        
            distance_grid, delta = calculate_time_delta(tel1, tel2, driver1, driver2)
        
            fig_delta = make_subplots(
                rows=3, cols=1,
                row_heights=[0.25, 0.45, 0.30],
                vertical_spacing=0.08,
                subplot_titles=("Track Position", f"Time Delta: {driver1} vs {driver2}", "Speed Comparison")
            )
        
            # Row 1: Track (grey)
            fig_delta.add_trace(go.Scatter(
                x=track_df['x'], y=track_df['y'], mode='lines',
                line=dict(color='#888888', width=4), hoverinfo='none', showlegend=False
            ), row=1, col=1)
        
            # Row 2: Time Delta graph with fill
            # Green fill = driver2 faster (positive delta: time1 > time2)
            # Red fill = driver1 faster (negative delta: time1 < time2)
            fig_delta.add_trace(go.Scatter(
                x=distance_grid, y=np.where(delta >= 0, delta, 0), mode='lines',
                line=dict(color='rgba(0,0,0,0)', width=0),
                fill='tozeroy', fillcolor='rgba(100,255,100,0.5)',
                name=f'{driver2} faster', showlegend=True
            ), row=2, col=1)
        
            fig_delta.add_trace(go.Scatter(
                x=distance_grid, y=np.where(delta < 0, delta, 0), mode='lines',
                line=dict(color='rgba(0,0,0,0)', width=0),
                fill='tozeroy', fillcolor='rgba(255,100,100,0.5)',
                name=f'{driver1} faster', showlegend=True
            ), row=2, col=1)
        
            fig_delta.add_trace(go.Scatter(
                x=distance_grid, y=delta, mode='lines',
                line=dict(color='white', width=2),
                name='Delta', showlegend=False
            ), row=2, col=1)
        
            fig_delta.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
            for boundary, label in [(s1_end, 'S1|S2'), (s2_end, 'S2|S3')]:
                fig_delta.add_vline(x=boundary, line_dash="dot", line_color="gray", row=2, col=1)
        
            # Row 3: Speed comparison
            color1 = driver_colors[driver1]
            color2 = driver_colors[driver2]
        
            fig_delta.add_trace(go.Scatter(
                x=tel1['Distance'], y=tel1['Speed'], mode='lines',
                line=dict(color=color1, width=2), name=driver1
            ), row=3, col=1)
        
            fig_delta.add_trace(go.Scatter(
                x=tel2['Distance'], y=tel2['Speed'], mode='lines',
                line=dict(color=color2, width=2), name=driver2
            ), row=3, col=1)
        
            fig_delta.update_layout(
                height=800,
                plot_bgcolor='rgba(15,15,25,1)',
                paper_bgcolor='rgba(15,15,25,1)',
                font=dict(color='white'),
                legend=dict(orientation='h', x=0, y=-0.05),
                margin=dict(l=60, r=20, t=60, b=40)
            )
        
            pad = 50
            fig_delta.update_xaxes(visible=False, fixedrange=True, row=1, col=1)
            fig_delta.update_yaxes(visible=False, fixedrange=True, scaleanchor='x', row=1, col=1)
        
            fig_delta.update_xaxes(title_text="Distance (m)", gridcolor='rgba(255,255,255,0.1)', row=2, col=1)
            fig_delta.update_yaxes(title_text="Delta (s)", gridcolor='rgba(255,255,255,0.1)', row=2, col=1)
        
            fig_delta.update_xaxes(title_text="Distance (m)", gridcolor='rgba(255,255,255,0.1)', row=3, col=1)
            fig_delta.update_yaxes(title_text="Speed (km/h)", gridcolor='rgba(255,255,255,0.1)', row=3, col=1)
        
            st.plotly_chart(fig_delta, use_container_width=True)
        
            # Summary metrics
            col_a, col_b, col_c = st.columns(3)
        
            final_delta = delta[-1]
            max_delta = delta.max()
            min_delta = delta.min()
        
            with col_a:
                faster_driver = driver2 if final_delta > 0 else driver1
                st.metric(
                    "Final Lap Delta",
                    f"{abs(final_delta):.3f}s",
                    delta=f"{faster_driver} faster",
                    delta_color="off"
                )
        
            with col_b:
                st.metric(
                    f"Max {driver1} Advantage",
                    f"{abs(min_delta):.3f}s" if min_delta < 0 else "0.000s"
                )
        
            with col_c:
                st.metric(
                    f"Max {driver2} Advantage",
                    f"{abs(max_delta):.3f}s" if max_delta > 0 else "0.000s"
                )
        
            # Sector breakdown
            st.markdown("#### Sector-by-Sector Breakdown")
        
            lb1 = leaderboard_df[leaderboard_df['Driver'] == driver1]
            lb2 = leaderboard_df[leaderboard_df['Driver'] == driver2]
        
            if not lb1.empty and not lb2.empty:
                lb1 = lb1.iloc[0]
                lb2 = lb2.iloc[0]
            
                sector_data = []
                for sector, s_col in [('Sector 1', 'S1_sec'), ('Sector 2', 'S2_sec'), ('Sector 3', 'S3_sec')]:
                    t1 = lb1[s_col]
                    t2 = lb2[s_col]
                    diff = t1 - t2
                    faster = driver2 if diff > 0 else driver1
                    sector_data.append({
                        'Sector': sector,
                        driver1: f"{t1:.3f}s" if t1 != float('inf') else 'N/A',
                        driver2: f"{t2:.3f}s" if t2 != float('inf') else 'N/A',
                        'Difference': f"{diff:+.3f}s" if t1 != float('inf') and t2 != float('inf') else 'N/A',
                        'Faster': faster if t1 != float('inf') and t2 != float('inf') else 'N/A'
                    })
            
                st.dataframe(pd.DataFrame(sector_data), use_container_width=True, hide_index=True)
        else:
            st.info("Please select two different drivers to compare.")

# Footer
st.markdown("---")
session_display = qual_session if session_type_display == "Qualifying" else "Race"
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 12px;'>
    F1 Telemetry Analysis Dashboard | Data provided by FastF1 | 
    Session: {gp_name} {year} {session_display}
</div>
""", unsafe_allow_html=True)
