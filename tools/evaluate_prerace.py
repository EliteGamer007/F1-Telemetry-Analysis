import argparse
import json
from pathlib import Path
from race_strategy.evaluate import evaluate_reports
from race_strategy.weekend import load_session, normalize_session, atomic_json


def main():
    p=argparse.ArgumentParser(description="Evaluate already frozen reports using target races; never changes model files")
    p.add_argument("reports",nargs="+",type=Path)
    p.add_argument("--output",type=Path,default=Path("data/evaluation.json"))
    args=p.parse_args(); reports=[]; laps=[]
    for path in args.reports:
        r=json.loads(path.read_text(encoding="utf-8")); reports.append(r)
        try:
            session=load_session(r["year"],r["round"],"R",cache_only=True)
            laps.extend(row|dict(year=r["year"],round=r["round"]) for row in normalize_session(session,"R"))
        except ValueError:
            print(f"No cached held-out race for {r['event']}; excluded, not fabricated")
    result=evaluate_reports(reports,laps)
    atomic_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in {"records","slices"}},indent=2))


if __name__=="__main__": main()
