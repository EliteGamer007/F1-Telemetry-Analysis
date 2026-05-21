import urllib.request, json

req = urllib.request.urlopen('http://localhost:8000/api/session?year=2024&gp=Abu%20Dhabi&session_code=R')
d = json.loads(req.read())
print('maxDuration:', round(d['maxDuration']))
print('totalLaps:', d['totalLaps'])
lb = d['leaderboard']
print('Leaderboard sample:')
for r in lb[:5]:
    pos = r['position']
    drv = r['driver']
    bls = r['bestLapStr']
    cmp = r['compound']
    sts = r['status']
    print('  P' + str(pos) + ' ' + drv + ' | ' + bls + ' | ' + cmp + ' | status=' + sts)

if d.get('lapHistory'):
    ver = d['lapHistory'].get('VER', [])
    if ver and len(ver) > 29:
        h29 = ver[28]
        h30 = ver[29]
        print('VER lap 29: compound=' + h29['compound'] + ' tyreLife=' + str(h29['tyreLife']) + ' pitInTrel=' + str(h29['pitInTrel']))
        print('VER lap 30: compound=' + h30['compound'] + ' tyreLife=' + str(h30['tyreLife']) + ' pitOutTrel=' + str(h30['pitOutTrel']))
    retired = [k for k, v in d['lapHistory'].items() if len(v) < 55]
    print('Drivers with < 55 laps (possibly retired):', retired)
