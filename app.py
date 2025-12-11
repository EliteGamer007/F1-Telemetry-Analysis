import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import fastf1
import numpy as np

DATA_DIR = 'data'
TRACK_FILE = os.path.join(DATA_DIR, 'Baku_2023_track_layout.csv')
CORNERS_FILE = os.path.join(DATA_DIR, 'Baku_2023_corners.csv')
CACHE_DIR = 'cache'

st.set_page_config(layout="centered", page_title="F1 Telemetry Analysis", page_icon="🏎️")

@st.cache_data
def load_data():
    if not os.path.exists(TRACK_FILE) or not os.path.exists(CORNERS_FILE):
        return None, None, None
    
    track_df = pd.read_csv(TRACK_FILE)
    corners_df = pd.read_csv(CORNERS_FILE)
    
    fastf1.Cache.enable_cache(CACHE_DIR)
    session = fastf1.get_session(2023, 'Baku', 'Q')
    session.load(weather=False, messages=False)
    lap = session.laps.pick_fastest()
    tel = lap.get_telemetry()
    
    tel['Distance'] = tel['Distance'].astype(float)
    tel['TimeSec'] = tel['Time'].dt.total_seconds()
    
    track_df['speed'] = np.interp(track_df['distance'], tel['Distance'], tel['Speed'])
    track_df['time'] = np.interp(track_df['distance'], tel['Distance'], tel['TimeSec'])
    
    return track_df, corners_df

st.title("F1 Telemetry Analysis")

track_df, corners_df = load_data()

if track_df is not None:
    step = 5 
    sim_df = track_df.iloc[::step].reset_index(drop=True)
    
    avg_time_step = sim_df['time'].diff().mean()
    if pd.isna(avg_time_step):
        avg_time_step = 0.25
        
    duration_1x = avg_time_step * 1000
    
    fig = make_subplots(
        rows=2, cols=1, 
        row_heights=[0.7, 0.3],
        vertical_spacing=0.1,
        specs=[[{"type": "xy"}], [{"type": "xy"}]]
    )

    # Static Track
    fig.add_trace(go.Scatter(
        x=track_df['x'], y=track_df['y'], mode='lines',
        line=dict(color='grey', width=5), hoverinfo='none'
    ), row=1, col=1)

    # Static Corners
    if corners_df is not None:
        lbl_x = corners_df['label_x'] if 'label_x' in corners_df.columns else corners_df['x']
        lbl_y = corners_df['label_y'] if 'label_y' in corners_df.columns else corners_df['y']
        fig.add_trace(go.Scatter(
            x=lbl_x, y=lbl_y, mode='text',
            text=corners_df['Number'].astype(str) + corners_df['Letter'].fillna(''),
            textfont=dict(size=9, color='white'), hoverinfo='none'
        ), row=1, col=1)

    # Static Speed Graph
    fig.add_trace(go.Scatter(
        x=track_df['distance'], y=track_df['speed'], mode='lines',
        line=dict(color='dimgrey', width=2), hoverinfo='none'
    ), row=2, col=1)

    # Car Dot (Index 3)
    fig.add_trace(go.Scatter(
        x=[track_df['x'].iloc[0]], y=[track_df['y'].iloc[0]], mode='markers',
        marker=dict(size=12, color='red', line=dict(width=1, color='white')), name='Car'
    ), row=1, col=1)

    # Speed Dot (Index 4)
    fig.add_trace(go.Scatter(
        x=[track_df['distance'].iloc[0]], y=[track_df['speed'].iloc[0]], mode='markers',
        marker=dict(size=10, color='red'), showlegend=False
    ), row=2, col=1)

    #  Digital Speedometer (Index 5)
    # Position: Bottom Left (min_x, min_y)
    speed_x_pos = track_df['x'].min()
    speed_y_pos = track_df['y'].min()
    
    fig.add_trace(go.Scatter(
        x=[speed_x_pos], y=[speed_y_pos], mode='text',
        text=[f"{track_df['speed'].iloc[0]:.0f} km/h"],
        textfont=dict(size=20, color='cyan', family="monospace"),
        textposition="top right",
        showlegend=False
    ), row=1, col=1)

    frames = []
    for i in range(len(sim_df)):
        row = sim_df.iloc[i]
        frames.append(go.Frame(
            data=[
                go.Scatter(x=[row['x']], y=[row['y']]), # Update Car
                go.Scatter(x=[row['distance']], y=[row['speed']]), # Update Graph Dot
                go.Scatter(text=[f"{row['speed']:.0f} km/h"]) # Update Speedometer Text
            ],
            traces=[3, 4, 5], # Indices of dynamic traces
            name=str(i)
        ))

    fig.frames = frames

    fig.update_layout(
        height=700, width=800,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        xaxis=dict(range=[track_df['x'].min()-100, track_df['x'].max()+100], visible=False, fixedrange=True),
        yaxis=dict(range=[track_df['y'].min()-100, track_df['y'].max()+100], visible=False, fixedrange=True, scaleanchor="x"),
        xaxis2=dict(title="Distance (m)", showgrid=False),
        yaxis2=dict(title="Speed", showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.0, y=1.15,
                buttons=[
                    dict(label="▶ 1x", method="animate", args=[None, dict(frame=dict(duration=duration_1x, redraw=False), transition=dict(duration=duration_1x, easing="linear"), fromcurrent=True)]),
                    dict(label="⏩ 2x", method="animate", args=[None, dict(frame=dict(duration=duration_1x/2, redraw=False), transition=dict(duration=duration_1x/2, easing="linear"), fromcurrent=True)]),
                    dict(label="⏭ 4x", method="animate", args=[None, dict(frame=dict(duration=duration_1x/4, redraw=False), transition=dict(duration=duration_1x/4, easing="linear"), fromcurrent=True)]),
                    dict(label="⏸ Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))])
                ]
            )
        ]
    )

    st.plotly_chart(fig, use_container_width=True)