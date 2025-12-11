import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import fastf1

TRACK_FILE = 'Abu Dhabi_2023_track_layout.csv'
CORNERS_FILE = 'Abu Dhabi_2023_corners.csv'
CACHE_DIR = 'cache'

st.set_page_config(layout="wide", page_title="F1 Telemetry Viz")

@st.cache_data
def load_data():
    if not os.path.exists(TRACK_FILE):
        return None, None, None
    
    track_df = pd.read_csv(TRACK_FILE)
    corners_df = pd.read_csv(CORNERS_FILE) if os.path.exists(CORNERS_FILE) else None
    
    fastf1.Cache.enable_cache(CACHE_DIR)
    session = fastf1.get_session(2023, 'Abu Dhabi', 'Q')
    session.load(weather=False, messages=False)
    lap = session.laps.pick_fastest()
    tel = lap.get_telemetry()
    tel['Distance'] = tel['Distance'].astype(float)
    
    return track_df, corners_df, tel

st.title(" F1 Telemetry Analysis: Abu Dhabi")

track_df, corners_df, telemetry_df = load_data()

if track_df is not None:
    max_dist = int(track_df['distance'].max())
    curr_dist = st.slider("Lap Distance (m)", 0, max_dist, 0)

    car_pos = track_df.iloc[(track_df['distance'] - curr_dist).abs().argsort()[:1]]
    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Track Map")
        fig_map = go.Figure()
        
        fig_map.add_trace(go.Scatter(
            x=track_df['x'], y=track_df['y'], mode='lines',
            line=dict(color='grey', width=4), hoverinfo='none'
        ))

        if corners_df is not None:
            fig_map.add_trace(go.Scatter(
                x=corners_df['x'], y=corners_df['y'], mode='text',
                text=corners_df['Number'].astype(str) + corners_df['Letter'].fillna(''),
                textfont=dict(size=10, color='white'), hoverinfo='none'
            ))

        fig_map.add_trace(go.Scatter(
            x=car_pos['x'], y=car_pos['y'], mode='markers',
            marker=dict(size=15, color='red', symbol='circle'), name='Car'
        ))

        fig_map.update_layout(
            width=800, height=600, plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
            margin=dict(l=0, r=0, t=0, b=0), showlegend=False
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with col2:
        st.subheader("Telemetry Trace")
        
        fig_tel = px.line(telemetry_df, x='Distance', y='Speed', color_discrete_sequence=['red'])
        
        fig_tel.add_vline(x=curr_dist, line_width=2, line_dash="dash", line_color="blue")
        
        fig_tel.update_layout(height=300, xaxis_title="Distance (m)", yaxis_title="Speed (km/h)")
        st.plotly_chart(fig_tel, use_container_width=True)
        
        curr_speed = telemetry_df.iloc[(telemetry_df['Distance'] - curr_dist).abs().argsort()[:1]]['Speed'].values[0]
        st.metric("Current Speed", f"{curr_speed:.0f} km/h")