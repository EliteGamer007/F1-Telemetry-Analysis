import urllib.request, json

req = urllib.request.urlopen('http://localhost:8000/api/session?year=2024&gp=Abu%20Dhabi&session_code=FP2', timeout=120)
d = json.loads(req.read())
pd_data = d.get('practiceDriverData') or {}
print('Practice drivers returned:', len(pd_data))
for abbr, info in list(pd_data.items())[:4]:
    print(' ', abbr,
          'best=' + info['bestLapStr'],
          'compound=' + info['currentCompound'],
          'tyreLife=' + str(info['currentTyreLife']),
          'last=' + info['lastLapStr'],
          'stints=' + str(len(info['stints'])),
          'totalLaps=' + str(info['totalLaps']))
    for s in info['stints']:
        flag = 'NEW' if s['fresh'] else 'used'
        print('    Stint:', s['compound'], 'L' + str(s['startLap']) + '-' + str(s['endLap']), str(s['laps']) + ' laps', flag)
print()
print('Keys in response:', list(d.keys()))
