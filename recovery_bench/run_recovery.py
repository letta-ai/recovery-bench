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
import logging
import sys

logger = logging.getLogger(__name__)
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
    parser.add_argument(
        "--agent-kwargs",
        type=str,
        default=None,
        help="JSON string of kwargs to pass to agent (e.g., '{\"model_kwargs\": {\"reasoning_effort\": \"high\"}}')",
    )

    args = parser.parse_args()

    traces_path = Path(args.traces)
    if not traces_path.exists():
        logger.error(f"Traces folder {args.traces} does not exist")
        return 1

    # Reorganize directories with hash prefixes if needed
    reorganize_directories(args.traces)

    # Get task IDs to replay
    if args.task_ids:
        task_ids = args.task_ids
    else:
        task_ids = get_unsolved_tasks(args.traces)
        if not task_ids:
            logger.warning("No unsolved tasks found in traces folder")
            return 0

    logger.info(f"Running recovery on {len(task_ids)} task(s) with {args.model}")

    # Generate job name if not specified
    if args.job_name:
        job_name = args.job_name
    else:
        model_short = args.model.split("/")[-1]
        traces_short = traces_path.name
        job_name = f"recovery-{model_short}-on-{traces_short}"

    # Parse agent kwargs
    agent_kwargs = None
    if args.agent_kwargs:
        import json
        try:
            agent_kwargs = json.loads(args.agent_kwargs)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON for --agent-kwargs: {e}")
            return 1

    return run_recovery(
        traces_folder=args.traces,
        model=args.model,
        task_ids=task_ids,
        job_name=job_name,
        agent=args.agent,
        n_concurrent=args.n_concurrent,
        agent_kwargs=agent_kwargs,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    sys.exit(main())
