#!/usr/bin/env python3
"""
Run recovery agent on existing initial traces.

Use this to evaluate different models' recovery capabilities on the same
set of failed initial traces.

Usage:
    # Using a model config file
    python -m recovery_bench.run_recovery \
        --traces jobs/initial-claude-haiku-4-5-20251001-20260219_234042 \
        --model-config configs/models/opus-4.6-high.json

    # Using a model name directly
    python -m recovery_bench.run_recovery \
        --traces jobs/initial-claude-haiku-4-5-20251001-20260219_234042 \
        --model anthropic/claude-opus-4-6

    # Use LettaCode recovery agent instead of terminus
    python -m recovery_bench.run_recovery \
        --traces jobs/initial-xxx \
        --model-config configs/models/opus-4.6-high.json \
        --agent recovery_bench.recovery_letta_code:RecoveryLettaCode
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from .utils import (
    aggregate_usage,
    get_unsolved_tasks,
    reorganize_directories,
    run_recovery,
)

logger = logging.getLogger(__name__)


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
        default=None,
        help="Model to use for recovery (e.g., anthropic/claude-opus-4-5-20251101)",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Path to model config JSON file (e.g., configs/models/o3-mini-high.json)",
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

    # Load model config if provided
    model_kwargs = {}
    if args.model_config:
        config_path = Path(args.model_config)
        if not config_path.exists():
            logger.error(f"Model config file not found: {args.model_config}")
            return 1
        try:
            with open(config_path) as f:
                config = json.load(f)
            model_kwargs = config.get("model_kwargs", {})
            logger.info(f"Loaded model config from {args.model_config}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in model config: {e}")
            return 1

    # --model flag takes priority, then config file
    model = args.model or config.get("model") if args.model_config else args.model

    if not model:
        logger.error("Model must be specified via --model or --model-config")
        return 1

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

    logger.info(f"Running recovery on {len(task_ids)} task(s) with {model}")

    # Generate job name if not specified
    if args.job_name:
        job_name = args.job_name
    else:
        model_short = model.split("/")[-1]
        traces_short = traces_path.name
        job_name = f"recovery-{model_short}-on-{traces_short}"

    result = run_recovery(
        traces_folder=args.traces,
        model=model,
        task_ids=task_ids,
        job_name=job_name,
        agent=args.agent,
        n_concurrent=args.n_concurrent,
        model_kwargs=model_kwargs,
    )

    # Aggregate and log usage across all tasks
    job_dir = Path("jobs") / job_name
    if job_dir.exists():
        usage = aggregate_usage(str(job_dir))
        if usage["tasks_with_usage"] > 0:
            logger.info(
                f"Total usage: {usage['total_tokens']} tokens "
                f"({usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion), "
                f"cost: ${usage['cost_usd']:.4f} "
                f"across {usage['tasks_with_usage']} task(s)"
            )

    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    sys.exit(main())
