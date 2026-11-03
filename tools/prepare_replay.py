"""Package one cached race into bounded public replay chunks."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from race_strategy.weekend import atomic_json

def main():
    p=argparse.ArgumentParser(); p.add_argument('--year',type=int,required=True); p.add_argument('--round',type=int,required=True)
    a=p.parse_args()
    import fastf1
    import api
    fastf1.Cache.offline_mode(True)
    event=fastf1.get_event(a.year,a.round)
    track=api.get_track(a.year,event.EventName,'R')
    data=api.get_session_data(a.year,event.EventName,'R',None)
    tel=data.pop('telemetry'); fastest=data.pop('fastestLapTelemetry'); chunks=[]
    folder=Path('artifacts/replays'); folder.mkdir(parents=True,exist_ok=True)
    codes=list(tel)
    for i in range(0,len(codes),3):
        selected=codes[i:i+3]
        raw=json.dumps(dict(telemetry={c:tel[c] for c in selected},fastestLapTelemetry={c:fastest[c] for c in selected if c in fastest}),allow_nan=False,separators=(',',':')).encode()
        key=hashlib.sha256(raw).hexdigest()[:16]; name=f'{a.year}-{a.round}-{key}.json.gz'; packed=gzip.compress(raw,mtime=0)
        if len(packed)>2*1024*1024: raise ValueError('Chunk too large')
        (folder/name).write_bytes(packed); chunks.append(dict(id=key,file=name,bytes=len(packed)))
    name=f'{a.year}-{a.round}-manifest.json'
    atomic_json(folder/name,dict(track=track,session=data,chunks=chunks))
    catalog=json.loads(Path('artifacts/catalog.json').read_text(encoding='utf-8'))
    catalog['replays']=[r for r in catalog.get('replays',[]) if (r['year'],r['round'])!=(a.year,a.round)]
    catalog['replays'].append(dict(year=a.year,round=a.round,event=str(event.EventName),date=str(event.EventDate.date()),manifest=name))
    atomic_json('artifacts/catalog.json',catalog)
    print(f'Packaged {len(codes)} drivers, {len(chunks)} chunks, {sum(c["bytes"] for c in chunks)} compressed bytes')

if __name__=='__main__': main()
