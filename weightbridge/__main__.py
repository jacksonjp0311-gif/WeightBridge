"""CLI: python -m weightbridge <command>"""
import argparse
import json
from pathlib import Path

from .core import apply_patch, inspect_checkpoint
from .demo import compare_demo, run_demo


def main():
    parser = argparse.ArgumentParser(description="Evidence-gated neural parameter patch research sandbox")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("demo", help="Create and score isolated synthetic defect-removal experiment")
    p.add_argument("--dir", default="demo-run")
    p = sub.add_parser("inspect", help="Summarize checkpoint tensors without exposing whole arrays")
    p.add_argument("model")
    p = sub.add_parser("apply", help="Validate and apply a proposed sparse weight patch to a NEW candidate file")
    p.add_argument("--base", required=True)
    p.add_argument("--patch", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("compare-demo", help="Evaluate same synthetic holdout on base and candidate")
    p.add_argument("--base", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.cmd == "demo":
        result = run_demo(args.dir)
    elif args.cmd == "inspect":
        result = inspect_checkpoint(args.model)
    elif args.cmd == "apply":
        patch = json.loads(Path(args.patch).read_text(encoding="utf-8"))
        policy = json.loads(Path(args.policy).read_text(encoding="utf-8"))
        result = apply_patch(args.base, patch, policy, args.out)
    else:
        result = compare_demo(args.base, args.candidate, args.out)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
