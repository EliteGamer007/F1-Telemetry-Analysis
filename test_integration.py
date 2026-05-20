import requests

print('Loading Monaco Q3 session...')
r = requests.get(
    'http://localhost:8000/api/session',
    params={'year': 2025, 'gp': 'Monaco Grand Prix', 'session_code': 'Q', 'qual_session': 'Q3'},
    timeout=120
)
print('Session status:', r.status_code)
if r.status_code == 200:
    d = r.json()
    lb = d['leaderboard']
    print(f'Drivers on leaderboard: {len(lb)}')
    for row in lb[:5]:
        print(f"  P{row['position']} {row['driver']} | {row['bestLapStr']} | {row['gap'] or 'LEADER'} | {row['compound']} | S1={row['S1']} S2={row['S2']} S3={row['S3']}")
    tel_keys = list(d['telemetry'].keys())
    print(f'\nTelemetry for {len(tel_keys)} drivers')
    first = d['telemetry'][tel_keys[0]]
    print(f'Fields: {list(first.keys())}')
    print(f'Points per driver: {len(first["Distance"])}')
    print(f'MaxDuration: {d["maxDuration"]:.2f}s')

    # Test time delta
    if len(tel_keys) >= 2:
        print('\nTesting time delta...')
        r2 = requests.get(
            'http://localhost:8000/api/time-delta',
            params={
                'year': 2025, 'gp': 'Monaco Grand Prix',
                'session_code': 'Q', 'qual_session': 'Q3',
                'driver1': tel_keys[0], 'driver2': tel_keys[1]
            },
            timeout=30
        )
        print('Delta status:', r2.status_code)
        if r2.status_code == 200:
            dd = r2.json()
            print(f'Delta points: {len(dd["delta"])}')
            print(f'Max delta: {max(dd["delta"]):.3f}s  Min: {min(dd["delta"]):.3f}s')
else:
    print('Error:', r.text[:500])
