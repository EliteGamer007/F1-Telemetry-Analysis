"""Build a frozen forecast from prepared inputs; never load target race labels."""
import argparse
import json
from pathlib import Path
from race_strategy.pace import fit_prior, fit_weekend
from race_strategy.planner import EventPolicy
from race_strategy.report import build_report
from race_strategy.weekend import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder",type=Path)
    p.add_argument("--laps",type=int,required=True,help="Scheduled GP distance, not inferred from race results")
    p.add_argument("--pit-loss",type=float,required=True,help="Explicit owner scenario assumption, seconds")
    p.add_argument("--output",type=Path,default=Path("data/reports"))
    args=p.parse_args()
    bundle=json.loads((args.folder/"weekend.json").read_text(encoding="utf-8"))
    training=json.loads((args.folder/"training.json").read_text(encoding="utf-8"))
    model=fit_weekend(bundle,fit_prior(training,bundle))
    report=build_report(bundle,model,EventPolicy(laps=args.laps,pit_loss=args.pit_loss))
    filename=args.output/f"{bundle['year']}-{bundle['round']}.json"
    atomic_json(filename,report)
    print(f"{filename}: {len(report['drivers'])} drivers, {report['support']}; "
          f"{sum(d['practice_samples'] for d in report['drivers'].values())} long-run laps")


if __name__=="__main__": main()
