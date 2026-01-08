import fastf1
import numpy as np
import pandas as pd
import os

YEAR = 2025
GP = 'Monaco'
SESSION = 'Q'
CACHE_DIR = 'cache'
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)
TRACK_FILE =   os.path.join(DATA_DIR,f"{GP}_{YEAR}_track_layout.csv")
CORNERS_FILE = os.path.join(DATA_DIR,f"{GP}_{YEAR}_corners.csv")

def rotate(xy, *, angle):
    rot_mat = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ])
    return np.matmul(xy, rot_mat)

def get_track_data():
    if os.path.exists(TRACK_FILE) and os.path.exists(CORNERS_FILE):
        return pd.read_csv(TRACK_FILE), pd.read_csv(CORNERS_FILE), f"{GP} {YEAR}"

    fastf1.Cache.enable_cache(CACHE_DIR)
    session = fastf1.get_session(YEAR, GP, SESSION)
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

    track_df.to_csv(TRACK_FILE, index=False)
    corners_df.to_csv(CORNERS_FILE, index=False)
    
    return track_df, corners_df, session.event['Location']

if __name__ == "__main__":
    try:
        track, corners, name = get_track_data()
        print(f" Data generated: {TRACK_FILE}, {CORNERS_FILE}")
    except Exception as e:
        print(f" Error: {e}")