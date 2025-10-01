# src/main.py
import argparse
from src.cli.commands import cmd_train_detect, cmd_train_classify, cmd_track, cmd_api

def build_parser():
    p = argparse.ArgumentParser(description="VehicleColorAI CLI")
    p.add_argument("--config", default="src/config.json")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("train-detect")
    d.add_argument("--variant", choices=["n","s"], default="n")
    d.add_argument("--part1-weights"); d.add_argument("--part2-weights")
    d.add_argument("--force-part1", action="store_true")
    d.add_argument("--force-part2", action="store_true")
    d.set_defaults(func=cmd_train_detect)

    c = sub.add_parser("train-classify")
    c.add_argument("--variant", choices=["n","s"], default="n")
    c.add_argument("--weights")
    c.set_defaults(func=cmd_train_classify)

    t = sub.add_parser("track")
    t.add_argument("--video", required=True)
    t.add_argument("--det-weights", required=True)
    t.add_argument("--cls-weights", required=True)
    t.add_argument("--output", default="")
    t.set_defaults(func=cmd_track)

    a = sub.add_parser("api")
    a.add_argument("--host", default="0.0.0.0"); a.add_argument("--port", type=int, default=8000)
    a.add_argument("--reload", action="store_true")
    a.set_defaults(func=cmd_api)
    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)

if __name__ == "__main__":
    main()
