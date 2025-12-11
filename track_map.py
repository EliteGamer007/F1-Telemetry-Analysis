import fastf1
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

YEAR = 2023
GP = 'Abu Dhabi'
SESSION = 'Q'
CACHE_DIR = 'cache'

def get_track_layout(year, gp, session_type):
    fastf1.Cache.enable_cache(CACHE_DIR)
    
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    
    fastest_lap = session.laps.pick_fastest()
    telemetry = fastest_lap.get_telemetry()
    
    circuit_info = session.get_circuit_info()
    track_angle = circuit_info.rotation / 180 * np.pi

    x = telemetry['X'].values
    y = telemetry['Y'].values

    x_rotated = x * np.cos(track_angle) - y * np.sin(track_angle)
    y_rotated = x * np.sin(track_angle) + y * np.cos(track_angle)

    track_df = pd.DataFrame({
        'x': x_rotated,
        'y': y_rotated,
        'z': telemetry['Z'].values,
        'distance': telemetry['Distance'].values
    })
    
    return track_df, session.event['Location']

if __name__ == "__main__":
    try:
        track_data, track_name = get_track_layout(YEAR, GP, SESSION)
        
        filename = f"{GP}_{YEAR}_track_layout.csv"
        track_data.to_csv(filename, index=False)
        print(f"Track layout saved to {filename}")

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(track_data['x'], track_data['y'], color='black', linewidth=2)
        ax.scatter(track_data['x'].iloc[0], track_data['y'].iloc[0], color='red', s=100)
        ax.set_title(f"{track_name} {YEAR}")
        ax.axis('equal')
        plt.show()

    except Exception as e:
        print(f"Error: {e}")