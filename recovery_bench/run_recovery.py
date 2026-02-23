#!/usr/bin/env python3
"""
Run recovery agent on existing initial traces.

Use this to evaluate different models' recovery capabilities on the same
set of failed initial traces.

Usage:
    python -m recovery_bench.run_recovery --traces jobs/initial-xxx --model anthropic/claude-opus-4-5-20251101

Examples:
    # Recover with opus on haiku's failed traces
    python -m recovery_bench.run_recovery \
        --traces jobs/initial-claude-haiku-4-5-20251001-20260202 \
        --model anthropic/claude-opus-4-5-20251101

    # Use LettaCode recovery agent instead of terminus
    python -m recovery_bench.run_recovery \
        --traces jobs/initial-xxx \
        --model anthropic/claude-opus-4-5-20251101 \
        --agent recovery_bench.replay_letta_code:ReplayLettaCode
"""

import argparse
import sys
from pathlib import Path

from .utils import (
    get_unsolved_tasks,
    reorganize_directories,
    run_recovery,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run recovery agent on existing initial traces"
    )
    parser.add_argument(
        "--traces",
        type=str,
        required=True,
        help="Path to initial traces folder (e.g., jobs/initial-haiku-xxx)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to use for replay (e.g., anthropic/claude-opus-4-5-20251101)",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="recovery_bench.recovery_terminus:RecoveryTerminus",
        help="Agent import path (default: RecoveryTerminus for terminus-2 traces)",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default=None,
        help="Job name for output folder (auto-generated if not specified)",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        action="append",
        dest="task_ids",
        help="Specific task ID(s) to replay (default: all unsolved)",
    )
    parser.add_argument(
        "--n-concurrent",
        type=int,
        default=8,
        help="Number of concurrent processes",
    )

    args = parser.parse_args()

    traces_path = Path(args.traces)
    if not traces_path.exists():
        print(f"Error: Traces folder {args.traces} does not exist")
        return 1

    # Reorganize directories with hash prefixes if needed
    reorganize_directories(args.traces)

    # Get task IDs to replay
    if args.task_ids:
        task_ids = args.task_ids
    else:
        task_ids = get_unsolved_tasks(args.traces)
        if not task_ids:
            print("No unsolved tasks found in traces folder")
            return 0

    print(f"Running recovery on {len(task_ids)} task(s) with {args.model}")

    # Generate job name if not specified
    if args.job_name:
        job_name = args.job_name
    else:
        model_short = args.model.split("/")[-1]
        traces_short = traces_path.name
        job_name = f"recovery-{model_short}-on-{traces_short}"

    return run_recovery(
        traces_folder=args.traces,
        model=args.model,
        task_ids=task_ids,
        job_name=job_name,
        agent=args.agent,
        n_concurrent=args.n_concurrent,
    )


if __name__ == "__main__":
    sys.exit(main())
