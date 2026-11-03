"""Host commands for a practice search and independent verification."""
from .controller import (
    init_research, research_status, rollback_research, score_final, search_cycle, verify_cycle,
)


def add_phase_iii_parser(subparsers):
    parser = subparsers.add_parser("phase-iii", help="Run a frozen continual adaptation protocol")
    stages = parser.add_subparsers(dest="phase_stage", required=True)
    init = stages.add_parser("init", help="Pin a base and protocol in a new research workspace")
    init.add_argument("--base", required=True)
    init.add_argument("--protocol", required=True)
    search = stages.add_parser("search", help="Select a finalist from practice data only")
    search.add_argument("--cycle", required=True)
    search.add_argument("--intent", required=True)
    search.add_argument("--practice", required=True)
    verify = stages.add_parser("verify", help="Host only. Independently evaluate a frozen finalist")
    verify.add_argument("--cycle", required=True)
    verify.add_argument("--holdout", required=True)
    verify.add_argument("--retain", action="append", default=[], help="Prior accepted holdout in ancestry order")
    final = stages.add_parser("final", help="Host only. Score the sealed final suite once")
    final.add_argument("--suite", required=True)
    status = stages.add_parser("status", help="Inspect research lineage metadata")
    rollback = stages.add_parser("rollback", help="Host only. Select an accepted research ancestor")
    rollback.add_argument("--ancestor", required=True)
    rollback.add_argument("--reason", required=True)
    for stage in (init, search, verify, final, status, rollback):
        stage.add_argument("--root", required=True)


def run_phase_iii_command(args):
    if args.phase_stage == "init":
        return init_research(args.root, args.base, args.protocol)
    if args.phase_stage == "search":
        return search_cycle(args.root, args.cycle, args.intent, args.practice)
    if args.phase_stage == "verify":
        return verify_cycle(args.root, args.cycle, args.holdout, args.retain)
    if args.phase_stage == "final":
        return score_final(args.root, args.suite)
    if args.phase_stage == "rollback":
        return rollback_research(args.root, args.ancestor, reason=args.reason)
    return research_status(args.root)
