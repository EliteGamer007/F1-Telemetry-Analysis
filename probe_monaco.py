import warnings; warnings.filterwarnings('ignore')
import logging; logging.getLogger('fastf1').setLevel(logging.ERROR)
import fastf1, pandas as pd, time, os

os.makedirs('./cache_live', exist_ok=True)
fastf1.Cache.enable_cache('./cache_live')

print('Trying Monaco 2026 Race...')
t0 = time.time()
try:
    s = fastf1.get_session(2026, 'Monaco', 'R')
    s.load(laps=True, telemetry=False, weather=False, messages=False)
    laps = s.laps
    elapsed = time.time()-t0
    if laps is not None and not laps.empty:
        max_lap = int(laps['LapNumber'].max())
        drivers = laps['Driver'].nunique()
        print(f'SUCCESS in {elapsed:.1f}s: {drivers} drivers, {len(laps)} lap rows, max lap completed={max_lap}')
        latest = laps.sort_values('LapNumber').groupby('Driver').last().reset_index()
        latest = latest.sort_values('Position')
        print('Top drivers:')
        for _, r in latest.head(10).iterrows():
            lt = r.get('LapTime')
            lt_s = lt.total_seconds() if pd.notna(lt) else 0
            mins = int(lt_s // 60)
            secs = lt_s % 60
            pos = int(r.Position) if pd.notna(r.Position) else '?'
            lap = int(r.LapNumber) if pd.notna(r.LapNumber) else '?'
            print(f'  P{pos} {r.Driver} L{lap} {mins}:{secs:06.3f}')
    else:
        print(f'EMPTY in {elapsed:.1f}s - session not available yet')
except Exception as e:
    print(f'ERROR after {time.time()-t0:.1f}s: {e}')
