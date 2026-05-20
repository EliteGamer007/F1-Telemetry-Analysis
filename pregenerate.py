import fastf1
import os
import numpy as np
import pandas as pd

DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

def rotate(xy, *, angle):
    rot_mat = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ])
    return np.matmul(xy, rot_mat)

def pregenerate_all():
    fastf1.Cache.enable_cache('cache')
    schedule = fastf1.get_event_schedule(2026)
    races = schedule[schedule['EventFormat'] != 'testing']['EventName'].tolist()
    
    for gp_name in races:
        track_file = os.path.join(DATA_DIR, f'{gp_name}_{2026}_track_layout.csv')
        if os.path.exists(track_file):
            continue
            
        try:
            print(f"Generating 2026 track data for {gp_name}...")
            session = fastf1.get_session(2026, gp_name, 'Q')
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
                corner_x.append(final_corner[0])
                corner_y.append(final_corner[1])
                label_x.append(final_text[0])
                label_y.append(final_text[1])
                corner_distances.append(row.get('Distance', 0))
                
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
            corners_df.to_csv(os.path.join(DATA_DIR, f'{gp_name}_{2026}_corners.csv'), index=False)
        except Exception as e:
            print(f"Skipped {gp_name}: {e}")

if __name__ == "__main__":
    pregenerate_all()
