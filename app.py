import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import fastf1
import numpy as np
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

DATA_DIR = 'data'
TRACK_FILE = os.path.join(DATA_DIR, 'Monaco_2025_track_layout.csv')
CORNERS_FILE = os.path.join(DATA_DIR, 'Monaco_2025_corners.csv')
CACHE_DIR = 'cache'

# Team color mappings (2023 season)
TEAM_COLORS = {
    'Red Bull Racing': ('#3671C6', '#15307A'),  # (primary, secondary)
    'Mercedes': ('#27F4D2', '#1CA998'),
    'Ferrari': ('#E8002D', '#B10023'),
    'McLaren': ('#FF8000', '#CC6600'),
    'Aston Martin': ('#229971', '#1A7357'),
    'Alpine': ('#FF87BC', '#D66A9A'),
    'Williams': ('#64C4FF', '#4A9BD1'),
    'AlphaTauri': ('#5E8FAA', '#4A7288'),
    'Alfa Romeo': ('#C92D4B', '#A12239'),
    'Haas F1 Team': ('#B6BABD', '#8D9195')
}

st.set_page_config(layout="wide", page_title="F1 Telemetry Analysis", page_icon="🏎️")

@st.cache_data(show_spinner=False)
def load_static_track():
    if not os.path.exists(TRACK_FILE) or not os.path.exists(CORNERS_FILE):
        return None, None
    track_df = pd.read_csv(TRACK_FILE)
    corners_df = pd.read_csv(CORNERS_FILE)
    return track_df, corners_df

@st.cache_data(show_spinner=True)
def load_session_and_laps(year: int, gp_name: str, session_name: str):
    fastf1.Cache.enable_cache(CACHE_DIR)
    session = fastf1.get_session(year, gp_name, session_name)
    session.load(weather=False, messages=False)
    return session

def get_driver_team_colors(_session) -> Dict[str, Tuple[str, str]]:
    """Map driver abbreviation to (color, team_name) using session metadata."""
    driver_colors = {}
    laps = _session.laps
    drivers = sorted(laps['Driver'].dropna().unique())
    
    # Build team roster
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
    
    # Assign colors: primary to first driver, secondary to second
    for team, drvs in team_drivers.items():
        colors = TEAM_COLORS.get(team, ('#AAAAAA', '#888888'))
        for i, drv in enumerate(drvs):
            driver_colors[drv] = (colors[i % 2], team)
    
    return driver_colors

def build_driver_telemetry_nocache(_session) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Tuple[str, str]], float]:
    laps = _session.laps
    drivers: List[str] = sorted(laps['Driver'].dropna().unique())
    driver_tel: Dict[str, pd.DataFrame] = {}
    durations: List[float] = []

    for drv in drivers:
        try:
            best = laps.pick_driver(drv).pick_fastest()
            if best is None or pd.isna(best['LapTime']):
                continue
            tel = best.get_telemetry()
            # Ensure numeric types
            tel['Distance'] = tel['Distance'].astype(float)
            tel['TimeSec'] = tel['Time'].dt.total_seconds()
            # Normalize time to lap start
            t0 = tel['TimeSec'].iloc[0]
            tel['Trel'] = tel['TimeSec'] - t0
            # Keep required columns only
            tel_keep = tel[['Distance', 'Speed', 'Trel']].copy()
            driver_tel[drv] = tel_keep.reset_index(drop=True)
            durations.append(tel_keep['Trel'].iloc[-1])
        except Exception:
            continue

    # Leaderboard by personal best lap time
    leaderboard_rows = []
    driver_colors_map = get_driver_team_colors(_session)
    for drv in driver_tel.keys():
        best = laps.pick_driver(drv).pick_fastest()
        if best is not None and not pd.isna(best['LapTime']):
            lap_seconds = float(best['LapTime'].total_seconds())
            minutes = int(lap_seconds // 60)
            seconds = lap_seconds % 60
            lap_str = f"{minutes}:{seconds:06.3f}"
            leaderboard_rows.append({
                'Driver': drv,
                'BestLapSeconds': lap_seconds,
                'BestLapStr': lap_str
            })
    leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values('BestLapSeconds', ascending=True).reset_index(drop=True)

    max_duration = float(max(durations)) if durations else 0.0
    return leaderboard_df, driver_tel, driver_colors_map, max_duration

st.title("F1 Telemetry Analysis — All Drivers")

track_df, corners_df = load_static_track()

if track_df is None:
    st.error("Track or corners CSV not found in data/. Please add Monaco_2025 files.")
else:
    # Load session data once
    with st.spinner("Loading session and best laps…"):
        session = load_session_and_laps(2025, 'Monaco', 'Q')
        leaderboard_df, driver_tel, driver_colors_map, max_duration = build_driver_telemetry_nocache(session)

    drv_list = list(driver_tel.keys())
    driver_colors: Dict[str, str] = {drv: driver_colors_map[drv][0] for drv in drv_list}  # extract primary color

    # Tabs: Race Animation | Lap Analysis
    tab1, tab2 = st.tabs(["Race Animation", "Lap Analysis"])

    with tab1:
        st.markdown("### Live Race Animation — All Drivers")
        
        # Build animation over a time grid (seconds)
        fps = 25
        dt = 1.0 / fps
        t_grid = np.arange(0.0, max_duration, dt)

        # Helper: distance(t) per driver via interpolation, then map to x,y using static track
        dist_to_x = lambda d: np.interp(d, track_df['distance'], track_df['x'])
        dist_to_y = lambda d: np.interp(d, track_df['distance'], track_df['y'])

        # Figure with track + drivers + dynamic leaderboard overlay
        fig = make_subplots(rows=1, cols=1, specs=[[{"type": "xy"}]])

        # Static track
        fig.add_trace(go.Scatter(
            x=track_df['x'], y=track_df['y'], mode='lines',
            line=dict(color='grey', width=5), hoverinfo='none', name='Track'
        ), row=1, col=1)

        # Static corners labels
        if corners_df is not None:
            lbl_x = corners_df['label_x'] if 'label_x' in corners_df.columns else corners_df['x']
            lbl_y = corners_df['label_y'] if 'label_y' in corners_df.columns else corners_df['y']
            fig.add_trace(go.Scatter(
                x=lbl_x, y=lbl_y, mode='text',
                text=corners_df['Number'].astype(str) + corners_df['Letter'].fillna(''),
                textfont=dict(size=9, color='white'), hoverinfo='none', showlegend=False
            ), row=1, col=1)

        # Add a scatter for each driver (dynamic traces start from first frame)
        dynamic_traces_start = len(fig.data)
        for drv in drv_list:
            color = driver_colors[drv]
            tel = driver_tel[drv]
            d0 = float(tel['Distance'].iloc[0])
            fig.add_trace(go.Scatter(
                x=[dist_to_x(d0)], y=[dist_to_y(d0)], mode='markers',
                marker=dict(size=10, color=color, line=dict(width=1, color='white')),
                name=drv, showlegend=False
            ), row=1, col=1)

        frame_leaderboards = []

        frames = []
        for ti in t_grid:
            frame_data = []
            current_distances = {}
            for drv in drv_list:
                tel = driver_tel[drv]
                d = float(np.interp(ti, tel['Trel'], tel['Distance'], left=tel['Distance'].iloc[0], right=tel['Distance'].iloc[-1]))
                current_distances[drv] = d
                frame_data.append(go.Scatter(x=[dist_to_x(d)], y=[dist_to_y(d)]))
            
            # Store leaderboard data for this frame
            ranked = sorted(current_distances.items(), key=lambda kv: -kv[1])
            frame_leaderboards.append(ranked)
            
            frames.append(go.Frame(
                data=frame_data,
                traces=list(range(dynamic_traces_start, dynamic_traces_start + len(drv_list))),
                name=f"t={ti:.2f}"
            ))

        fig.frames = frames

        # Layout and animation controls
        pad = 100
        fig.update_layout(
            height=720,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(range=[track_df['x'].min()-pad, track_df['x'].max()+pad], visible=False, fixedrange=True),
            yaxis=dict(range=[track_df['y'].min()-pad, track_df['y'].max()+pad], visible=False, fixedrange=True, scaleanchor="x"),
            updatemenus=[dict(
                type='buttons', direction='right', x=0.0, y=1.10,
                buttons=[
                    dict(label='▶ 1x', method='animate', args=[None, dict(frame=dict(duration=int(1000*dt), redraw=False), transition=dict(duration=0), fromcurrent=True)]),
                    dict(label='⏩ 2x', method='animate', args=[None, dict(frame=dict(duration=int(1000*dt/2), redraw=False), transition=dict(duration=0), fromcurrent=True)]),
                    dict(label='⏭ 4x', method='animate', args=[None, dict(frame=dict(duration=int(1000*dt/4)+1, redraw=False), transition=dict(duration=0), fromcurrent=True)]),
                    dict(label='⏸ Pause', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate', transition=dict(duration=0))])
                ]
            )]
        )

        st.plotly_chart(fig, use_container_width=True)
        
        # Leaderboard below the plot
        st.markdown("###Live Leaderboard")
        show_all = st.checkbox("Show all drivers", value=False, key="expand_lb")
        
        # Use initial frame leaderboard (frame 0)
        if frame_leaderboards:
            display_count = len(frame_leaderboards[0]) if show_all else min(10, len(frame_leaderboards[0]))
            lb_rows = []
            for rank, (drv, dist) in enumerate(frame_leaderboards[0][:display_count], 1):
                color = driver_colors[drv]
                team = driver_colors_map[drv][1]
                lb_rows.append({
                    'Pos': rank,
                    'Driver': drv,
                    'Team': team,
                    'Distance (m)': f"{dist:.0f}"
                })
            lb_df = pd.DataFrame(lb_rows)
            
            # Style with team colors
            def color_row(row):
                drv = row['Driver']
                color = driver_colors[drv]
                return [f'background-color: {color}20' for _ in row]
            
            st.dataframe(lb_df, use_container_width=True, hide_index=True)

    # ===== TAB 2: Lap Analysis =====
    with tab2:
        st.markdown("### Lap Analysis — Compare Drivers")
        
        # Driver selection
        available_drivers = sorted(driver_tel.keys())
        selected = st.multiselect(
            "Select drivers to compare:",
            options=available_drivers,
            default=available_drivers[:2] if len(available_drivers) >= 2 else available_drivers
        )
        
        if len(selected) == 0:
            st.info("Please select at least one driver.")
        else:
            # Build animated track visualization with selected drivers
            st.markdown("#### Live Lap Visualization")
            
            # Calculate max duration for selected drivers
            selected_durations = [driver_tel[drv]['Trel'].iloc[-1] for drv in selected]
            max_selected_duration = float(max(selected_durations)) if selected_durations else 0.0
            
            # Animation parameters
            fps = 25
            dt = 1.0 / fps
            t_grid_selected = np.arange(0.0, max_selected_duration, dt)
            
            # Helper functions
            dist_to_x = lambda d: np.interp(d, track_df['distance'], track_df['x'])
            dist_to_y = lambda d: np.interp(d, track_df['distance'], track_df['y'])
            
            # Build animated figure
            fig_lap = make_subplots(rows=1, cols=1, specs=[[{"type": "xy"}]])
            
            # Static track
            fig_lap.add_trace(go.Scatter(
                x=track_df['x'], y=track_df['y'], mode='lines',
                line=dict(color='grey', width=5), hoverinfo='none', showlegend=False
            ), row=1, col=1)
            
            # Corners
            if corners_df is not None:
                lbl_x = corners_df['label_x'] if 'label_x' in corners_df.columns else corners_df['x']
                lbl_y = corners_df['label_y'] if 'label_y' in corners_df.columns else corners_df['y']
                fig_lap.add_trace(go.Scatter(
                    x=lbl_x, y=lbl_y, mode='text',
                    text=corners_df['Number'].astype(str) + corners_df['Letter'].fillna(''),
                    textfont=dict(size=9, color='white'), hoverinfo='none', showlegend=False
                ), row=1, col=1)
            
            # Add driver markers
            dynamic_start = len(fig_lap.data)
            for drv in selected:
                color = driver_colors[drv]
                tel = driver_tel[drv]
                d0 = float(tel['Distance'].iloc[0])
                fig_lap.add_trace(go.Scatter(
                    x=[dist_to_x(d0)], y=[dist_to_y(d0)], mode='markers',
                    marker=dict(size=12, color=color, line=dict(width=2, color='white')),
                    name=drv, showlegend=True
                ), row=1, col=1)
            
            # Build frames
            frames_lap = []
            for ti in t_grid_selected:
                frame_data = []
                for drv in selected:
                    tel = driver_tel[drv]
                    d = float(np.interp(ti, tel['Trel'], tel['Distance'], left=tel['Distance'].iloc[0], right=tel['Distance'].iloc[-1]))
                    frame_data.append(go.Scatter(x=[dist_to_x(d)], y=[dist_to_y(d)]))
                
                frames_lap.append(go.Frame(
                    data=frame_data,
                    traces=list(range(dynamic_start, dynamic_start + len(selected))),
                    name=f"t={ti:.2f}"
                ))
            
            fig_lap.frames = frames_lap
            
            # Layout
            pad = 50
            fig_lap.update_layout(
                height=650,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(range=[track_df['x'].min()-pad, track_df['x'].max()+pad], visible=False, fixedrange=True),
                yaxis=dict(range=[track_df['y'].min()-pad, track_df['y'].max()+pad], visible=False, fixedrange=True, scaleanchor='x'),
                legend=dict(orientation='h', x=0, y=-0.05),
                updatemenus=[dict(
                    type='buttons', direction='right', x=0.0, y=1.08,
                    buttons=[
                        dict(label='▶ 1x', method='animate', args=[None, dict(frame=dict(duration=int(1000*dt), redraw=False), transition=dict(duration=0), fromcurrent=True)]),
                        dict(label='⏩ 2x', method='animate', args=[None, dict(frame=dict(duration=int(1000*dt/2), redraw=False), transition=dict(duration=0), fromcurrent=True)]),
                        dict(label='⏭ 4x', method='animate', args=[None, dict(frame=dict(duration=int(1000*dt/4)+1, redraw=False), transition=dict(duration=0), fromcurrent=True)]),
                        dict(label='⏸ Pause', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate', transition=dict(duration=0))])
                    ]
                )]
            )
            
            st.plotly_chart(fig_lap, use_container_width=True)
            
            # Two columns: corner analysis + speed trace
            col1, col2 = st.columns([0.5, 0.5])
            
            with col1:
                st.markdown("#### Corner-by-Corner Speed")
                if corners_df is not None and not corners_df.empty:
                    corner_analysis = []
                    for _, corner in corners_df.iterrows():
                        corner_dist = corner['distance'] if 'distance' in corner else 0
                        corner_label = str(corner['Number']) + str(corner.get('Letter', ''))
                        
                        for drv in selected:
                            tel = driver_tel[drv]
                            speed_at_corner = float(np.interp(corner_dist, tel['Distance'], tel['Speed']))
                            corner_analysis.append({
                                'Corner': corner_label,
                                'Driver': drv,
                                'Speed (km/h)': speed_at_corner
                            })
                    
                    corner_df = pd.DataFrame(corner_analysis)
                    corner_pivot = corner_df.pivot(index='Corner', columns='Driver', values='Speed (km/h)')
                    
                    fig_corner = go.Figure()
                    for drv in selected:
                        if drv in corner_pivot.columns:
                            color = driver_colors[drv]
                            fig_corner.add_trace(go.Bar(
                                x=corner_pivot.index,
                                y=corner_pivot[drv],
                                name=drv,
                                marker_color=color
                            ))
                    
                    fig_corner.update_layout(
                        xaxis_title="Corner",
                        yaxis_title="Speed (km/h)",
                        height=350,
                        barmode='group',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        showlegend=False,
                        margin=dict(l=40, r=20, t=20, b=40)
                    )
                    st.plotly_chart(fig_corner, use_container_width=True)
                else:
                    st.warning("Corner data not available.")
            
            with col2:
                st.markdown("#### Speed Trace")
                fig_speed = go.Figure()
                for drv in selected:
                    tel = driver_tel[drv]
                    color = driver_colors[drv]
                    fig_speed.add_trace(go.Scatter(
                        x=tel['Distance'], y=tel['Speed'], mode='lines',
                        line=dict(color=color, width=2),
                        name=drv
                    ))
                
                # Add corner markers
                if corners_df is not None:
                    for _, corner in corners_df.iterrows():
                        corner_dist = corner['distance'] if 'distance' in corner else 0
                        fig_speed.add_vline(x=corner_dist, line_dash='dot', line_color='yellow', opacity=0.3)
                
                fig_speed.update_layout(
                    xaxis_title="Distance (m)",
                    yaxis_title="Speed (km/h)",
                    height=350,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=False,
                    margin=dict(l=40, r=20, t=20, b=40)
                )
                st.plotly_chart(fig_speed, use_container_width=True)
            
            # Summary stats table
            st.markdown("#### Summary Statistics")
            stats_rows = []
            for drv in selected:
                tel = driver_tel[drv]
                best_lap = leaderboard_df[leaderboard_df['Driver'] == drv].iloc[0] if drv in leaderboard_df['Driver'].values else None
                stats_rows.append({
                    'Driver': drv,
                    'Team': driver_colors_map[drv][1],
                    'Best Lap': best_lap['BestLapStr'] if best_lap is not None else 'N/A',
                    'Max Speed (km/h)': f"{tel['Speed'].max():.1f}",
                    'Avg Speed (km/h)': f"{tel['Speed'].mean():.1f}"
                })
            stats_df = pd.DataFrame(stats_rows)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)