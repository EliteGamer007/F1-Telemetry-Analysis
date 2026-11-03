"""python -m tools.prepare_weekend --year 2026 --round 12 --training 1,3,6"""
import argparse
from pathlib import Path
from race_strategy.weekend import prepare_weekend, prepare_training_table, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--training", default="")
    parser.add_argument("--output", default="data/prepared")
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()
    bundle = prepare_weekend(args.year, args.round, cache_only=args.cache_only)
    folder = Path(args.output) / f"{args.year}-{args.round}"
    atomic_json(folder / "weekend.json", bundle)
    keys = [tuple(map(int, r.split(":"))) if ":" in r else (args.year, int(r)) for r in args.training.split(",") if r]
    training = prepare_training_table(keys, bundle, cache_only=args.cache_only)
    atomic_json(folder / "training.json", training)
    print(f"Prepared {bundle['event']}: {len(bundle['laps'])} laps, {len(training)} historical events")


if __name__ == "__main__":
    main()
