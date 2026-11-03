"""Small public smoke test, not a capacity/load test. No provider downloads."""
import argparse
import concurrent.futures
import json
import statistics
import time
from pathlib import Path
import requests


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--api',required=True); p.add_argument('--frontend',required=True)
    p.add_argument('--output',default='data/serving-benchmark.json')
    args=p.parse_args(); base=args.api.rstrip('/')
    def get(path):
        start=time.perf_counter()
        r=requests.get(base+path,timeout=65,headers={'Origin':args.frontend})
        r.raise_for_status()
        assert r.headers.get('access-control-allow-origin')==args.frontend
        return r,round(time.perf_counter()-start,3)
    health,_=get('/health'); assert health.json()['mode']=='prepared-only'
    catalog,_=get('/api/strategy/weekends')
    reports=[]
    for event in catalog.json()['weekends']:
        path=f"/api/strategy/prerace/{event['year']}/{event['round']}"
        r,duration=get(path)
        assert r.json()['drivers'] and r.json()['schema_version']==1
        reports.append(dict(event=event['event'],seconds=duration,compressed_bytes=int(r.headers.get('content-length',0))))
    path='/api/strategy/prerace/2026/4'
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        durations=[v[1] for v in pool.map(get,[path]*5)]
    replay=[]
    for event in catalog.json()['replays']:
        prefix=f"/api/replay/{event['year']}/{event['round']}"
        manifest,elapsed=get(prefix+'/manifest'); total=elapsed; drivers=set()
        for c in manifest.json()['chunks']:
            r,duration=get(prefix+'/chunks/'+c['id']); total+=duration
            assert r.headers.get('content-encoding')=='gzip'
            drivers.update(r.json()['telemetry'])
        assert drivers=={r['driver'] for r in manifest.json()['session']['leaderboard']}
        replay.append(dict(event=event['event'],drivers=len(drivers),sequential_download_seconds=round(total,3)))
    assert requests.get(base+'/api/strategy/prerace/2026/999',timeout=30).status_code==404
    assert requests.post(base+'/api/strategy/scenarios',json={},timeout=30).status_code==409
    assert requests.get(args.frontend,timeout=30).status_code==200
    result=dict(reports=reports,five_concurrent_reads_seconds=durations,median_concurrent=statistics.median(durations),replay=replay,
                note='Warm public smoke measurements from this machine. Cold wake and maximum capacity not measured.')
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
