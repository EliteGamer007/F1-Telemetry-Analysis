"""Content-addressed immutable publication. Catalog is replaced only after validation."""
import argparse
import gzip
import json
from pathlib import Path
from race_strategy.report import validate_report
from race_strategy.weekend import atomic_json, digest


def publish_reports(source_dir, public_dir):
    source_dir,public_dir=Path(source_dir),Path(public_dir)
    validated=[validate_report(json.loads(p.read_text(encoding="utf-8"))) for p in sorted(source_dir.glob("*.json"))]
    if not validated:
        raise ValueError("No complete reports to publish")
    entries=[]
    for report in validated:
        name=f"{report['year']}-{report['round']}-{digest(report)[:12]}.json"
        # Models stay inside reports to support exact, reproducible evaluation.
        atomic_json(public_dir/"reports"/name,report)
        size=len(gzip.compress((public_dir/"reports"/name).read_bytes()))
        if size>1024*1024:
            raise ValueError("Report exceeds compressed payload budget")
        entries.append({k:report[k] for k in ("year","round","event","location","date","support")} |
                       dict(file=name,era=report["manifest"]["era"],compressed_bytes=size))
    existing=public_dir/"catalog.json"
    replays=json.loads(existing.read_text(encoding="utf-8")).get("replays",[]) if existing.exists() else []
    atomic_json(existing,dict(schema_version=1,weekends=sorted(entries,key=lambda e:(e["year"],e["round"]),reverse=True),replays=replays))
    return existing


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("source",type=Path); p.add_argument("destination",type=Path)
    args=p.parse_args(); print(publish_reports(args.source,args.destination))
