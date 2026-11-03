"""CLI: python -m weightbridge <command>"""
import argparse
import json
from pathlib import Path

from .core import apply_patch, inspect_checkpoint, json_dump_atomic
from .demo import compare_demo, run_demo
from .patches import apply_safetensors_patch
from .inference import HFTextRunner, InferenceConfig
from .diagnostics.breakpoint_cli import propose_breakpoints


def main():
    parser = argparse.ArgumentParser(description="Evidence-gated neural parameter patch research sandbox")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("demo", help="Create and score isolated synthetic defect-removal experiment")
    p.add_argument("--dir", default="demo-run")
    p = sub.add_parser("inspect", help="Summarize checkpoint tensors without exposing whole arrays")
    p.add_argument("model")
    p.add_argument("--max-tensors", type=int, default=128)
    p = sub.add_parser("apply", help="Validate and apply a proposed sparse weight patch to a NEW candidate file")
    p.add_argument("--base", required=True)
    p.add_argument("--patch", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("breakpoint-menu", help="Propose bounded scalar edits from labeled toy practice data")
    p.add_argument("--base", required=True)
    p.add_argument("--practice", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--best-patch", action="store_true", help="Write the best proposal as an apply-ready patch")
    p = sub.add_parser("apply-hf", help="Patch a sharded or unsharded safetensors checkpoint into a new candidate")
    p.add_argument("--base", required=True)
    p.add_argument("--patch", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("compare-demo", help="Evaluate same synthetic holdout on base and candidate")
    p.add_argument("--base", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("generate", help="Run budgeted local Hugging Face inference")
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", default=None)
    p.add_argument("--prompt-file", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-input-tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--presence-penalty", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout-seconds", type=float, default=600)
    p.add_argument("--cpu-threads", type=int, default=0)
    p.add_argument("--disable-thinking", action="store_true")
    args = parser.parse_args()
    if args.cmd == "demo":
        result = run_demo(args.dir)
    elif args.cmd == "inspect":
        result = inspect_checkpoint(args.model, max_tensors=args.max_tensors)
    elif args.cmd == "apply":
        patch = json.loads(Path(args.patch).read_text(encoding="utf-8"))
        policy = json.loads(Path(args.policy).read_text(encoding="utf-8"))
        result = apply_patch(args.base, patch, policy, args.out)
    elif args.cmd == "breakpoint-menu":
        policy = json.loads(Path(args.policy).read_text(encoding="utf-8"))
        result = propose_breakpoints(args.base, args.practice, policy, best_patch=args.best_patch)
        json_dump_atomic(args.out, result)
    elif args.cmd == "apply-hf":
        patch = json.loads(Path(args.patch).read_text(encoding="utf-8"))
        policy = json.loads(Path(args.policy).read_text(encoding="utf-8"))
        result = apply_safetensors_patch(args.base, patch, policy, args.out)
    elif args.cmd == "generate":
        if bool(args.prompt) == bool(args.prompt_file):
            parser.error("provide exactly one of --prompt or --prompt-file")
        prompt = args.prompt if args.prompt is not None else Path(args.prompt_file).read_text(encoding="utf-8")
        config = InferenceConfig(max_new_tokens=args.max_new_tokens,
                                 max_input_tokens=args.max_input_tokens,
                                 temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                                 presence_penalty=args.presence_penalty, seed=args.seed,
                                 thinking_enabled=not args.disable_thinking,
                                 timeout_seconds=args.timeout_seconds, cpu_threads=args.cpu_threads)
        result = HFTextRunner.from_pretrained(args.model, device=args.device, dtype=args.dtype).generate(
            prompt, config).to_dict()
        if args.out:
            target = Path(args.out)
            with target.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(result, stream, indent=2, sort_keys=True, ensure_ascii=False)
                stream.write("\n")
    else:
        result = compare_demo(args.base, args.candidate, args.out)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
