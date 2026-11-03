"""Public runtime: stdlib + FastAPI. No FastF1, pandas, NumPy or model training."""
from collections import OrderedDict
from hashlib import sha256
from pathlib import Path
import json
import os
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response

DEFAULT_ROOT=Path(__file__).resolve().parents[1]/"artifacts"


def create_app(root=None,read_limit=None):
    root=Path(root or os.getenv("F1_ARTIFACT_DIR",DEFAULT_ROOT))
    app=FastAPI(title="F1 Telemetry & Strategy",version="3.0.0")
    app.add_middleware(CORSMiddleware,allow_origins=os.getenv("CORS_ORIGINS","http://localhost:3000").split(","),allow_methods=["GET","POST"],allow_headers=["Content-Type","If-None-Match"],expose_headers=["ETag","Retry-After"])
    app.add_middleware(GZipMiddleware,minimum_size=1000)
    limit=read_limit or int(os.getenv("READ_LIMIT_PER_MINUTE","120"))
    buckets=OrderedDict(); global_bucket=[0,0]

    @app.middleware("http")
    async def bounded_reads(request,call_next):
        if request.url.path.startswith("/api/") and request.method!="OPTIONS":
            now=int(time.monotonic()//60)
            # Do not trust arbitrary forwarded-for headers. Render/uvicorn supplies the peer.
            key=request.client.host if request.client else "unknown"
            minute,count=buckets.pop(key,(now,0))
            count=count+1 if minute==now else 1
            buckets[key]=(now,count)
            while len(buckets)>1024: buckets.popitem(last=False)
            if global_bucket[0]!=now: global_bucket[:]=[now,0]
            global_bucket[1]+=1
            if count>limit or global_bucket[1]>1200:
                return JSONResponse({"detail":"Read limit reached. Please retry in a minute."},status_code=429,headers={"Retry-After":"60"})
        response=await call_next(request)
        response.headers["X-Content-Type-Options"]="nosniff"
        return response

    def read(path):
        try:
            # Paths come only from our validated catalog, never a user-controlled path.
            resolved=path.resolve()
            if not resolved.is_relative_to(root.resolve()): raise ValueError("Invalid artifact path")
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError,ValueError):
            raise HTTPException(503,"Prepared artifact unavailable or invalid")

    def catalog(): return read(root/"catalog.json")
    def find(year,round_number,collection="weekends"):
        item=next((r for r in catalog().get(collection,[]) if r["year"]==year and r["round"]==round_number),None)
        if item is None: raise HTTPException(404,"This weekend has not been prepared. Select one from the catalog.")
        return item

    @app.get("/health")
    async def health():
        return dict(status="ok",mode="prepared-only",custom_compute=False,version="3.0.0")

    @app.get("/api/strategy/weekends")
    async def weekends():
        data=catalog()
        return dict(weekends=[{k:v for k,v in r.items() if k!="file"} for r in data["weekends"]],replays=data.get("replays",[]),custom_compute=False)

    @app.get("/api/strategy/prerace/{year}/{round_number}")
    async def prerace(year:int,round_number:int,request:Request):
        entry=find(year,round_number)
        data=read(root/"reports"/entry["file"])
        if data.get("schema_version")!=1 or not data.get("drivers") or not data.get("manifest",{}).get("source_hash"):
            raise HTTPException(503,"Prepared report failed schema validation")
        raw=json.dumps(data,separators=(",",":"),allow_nan=False).encode()
        etag='"'+sha256(raw).hexdigest()+'"'
        headers={"ETag":etag,"Cache-Control":"public, max-age=300"}
        if request.headers.get("if-none-match")==etag: return Response(status_code=304,headers=headers)
        return Response(raw,media_type="application/json",headers=headers)

    @app.post("/api/strategy/scenarios")
    async def scenarios():
        raise HTTPException(409,"Custom compute is disabled on the free pilot. Select one of the nine prepared scenarios in the report.")

    @app.get("/api/strategy/jobs/{job_id}")
    async def jobs(job_id:str):
        raise HTTPException(404,"No runtime jobs: this deployment uses prepared scenarios only")

    @app.get("/api/replay/{year}/{round_number}/manifest")
    async def replay_manifest(year:int,round_number:int):
        entry=find(year,round_number,"replays")
        return read(root/"replays"/entry["manifest"])

    @app.get("/api/replay/{year}/{round_number}/chunks/{chunk_id}")
    async def replay_chunk(year:int,round_number:int,chunk_id:str):
        manifest=await replay_manifest(year,round_number)
        chunk=next((c for c in manifest["chunks"] if c["id"]==chunk_id),None)
        if chunk is None: raise HTTPException(404,"Unknown prepared chunk")
        path=(root/"replays"/chunk["file"]).resolve()
        if not path.is_relative_to((root/"replays").resolve()) or not path.is_file():
            raise HTTPException(503,"Replay chunk unavailable")
        return Response(path.read_bytes(),media_type="application/json",headers={"Content-Encoding":"gzip","Cache-Control":"public, max-age=86400"})

    @app.get("/api/schedule/{year}")
    async def schedule(year:int):
        return dict(races=[dict(EventName=r["event"],EventDate=r["date"],EventFormat="prepared",RoundNumber=r["round"],sessions=["Race"])
                           for r in catalog().get("replays",[]) if r["year"]==year])

    def by_name(year,gp):
        entry=next((r for r in catalog().get("replays",[]) if r["year"]==year and r["event"]==gp),None)
        if entry is None: raise HTTPException(404,"Replay not prepared for this event")
        return entry

    @app.get("/api/track")
    async def track(year:int,gp:str,session_code:str="R"):
        entry=by_name(year,gp)
        return (await replay_manifest(year,entry["round"]))["track"]

    @app.get("/api/session")
    async def session(year:int,gp:str,session_code:str="R"):
        by_name(year,gp)
        raise HTTPException(409,"Use the chunked replay manifest; full raw session ingestion is disabled")

    @app.get("/api/time-delta")
    async def delta():
        raise HTTPException(409,"Compare prepared fastest-lap telemetry in your browser; raw ingestion is disabled")

    return app


app=create_app()
