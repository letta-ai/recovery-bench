#!/usr/bin/env python3
"""
Generate recovery-bench traces.

This script automates the entire trace generation pipeline:
1. Generate initial traces with an agent (default: terminus-2)
2. Reorganize traces with hash prefixes for recovery lookup
3. Iteratively run recovery agent on unsolved tasks

It also supports running recovery on existing traces (--resume-initial).

Usage:
    # Full pipeline: initial traces + recovery
    python -m recovery_bench.generate_traces \
        --initial-model anthropic/claude-haiku-4-5-20251001 \
        --recovery-model anthropic/claude-opus-4-5-20251101 \
        --task-id cancel-async-tasks

    # Recovery only on existing traces
    python -m recovery_bench.generate_traces \
        --recovery-model configs/models/opus-4.6-high.json \
        --resume-initial jobs/initial-haiku-xxx

    # Custom agents (LettaCode initial, RecoveryLettaCode recovery)
    python -m recovery_bench.generate_traces --initial-model openai/gpt-4o-mini \
        --initial-agent recovery_bench.letta_code_agent:LettaCode \
        --recovery-agent recovery_bench.recovery_letta_code:RecoveryLettaCode \
        --task-id constraints-scheduling
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from typing import List

from .utils import (
    cleanup_docker,
    get_agent_name,
    reorganize_directories,
    resolve_model,
    run_command,
    run_recovery_pipeline,
    shorten_model_name,
)

logger = logging.getLogger(__name__)


def generate_initial_traces(
    model_name: str,
    run_id: str,
    dataset_version: str = "2.0",
    n_concurrent: int = 6,
    task_ids: List[str] | None = None,
    agent_import_path: str | None = None,
    model_kwargs: dict | None = None,
) -> str:
    """Generate initial traces using harbor run."""
    logger.info(f"Generating initial traces for {model_name}...")

    cmd = [
        "harbor",
        "run",
        "--dataset",
        f"terminal-bench@{dataset_version}",
    ]

    # Use custom agent or default to terminus-2
    if agent_import_path and ":" in agent_import_path:
        # Import path format (e.g., module.path:ClassName)
        cmd.extend(["--agent-import-path", agent_import_path])
    elif agent_import_path:
        # Built-in agent name (e.g., terminus-2)
        cmd.extend(["--agent", agent_import_path])
    else:
        cmd.extend(["--agent", "terminus-2"])

    cmd.extend([
        "--model",
        model_name,
        "--job-name",
        run_id,
        "--n-concurrent",
        str(n_concurrent),
    ])

    if task_ids:
        for task_id in task_ids:
            cmd.extend(["--task-name", task_id])

    if model_kwargs:
        cmd.extend(["--agent-kwarg", f"model_kwargs={json.dumps(model_kwargs)}"])

    run_command(cmd)
    return f"jobs/{run_id}"


def main():
    parser = argparse.ArgumentParser(description="Generate traces pipeline")
    parser.add_argument(
        "--initial-model",
        type=str,
        default=None,
        help="Model name or JSON config for initial traces (e.g., anthropic/claude-haiku-4-5 or configs/models/haiku.json). Required unless --resume-initial is used.",
    )
    parser.add_argument(
        "--recovery-model",
        type=str,
        default=None,
        help="Model name or JSON config for recovery (e.g., anthropic/claude-opus-4-5 or configs/models/opus-4.6-high.json). Required for recovery; skips recovery if omitted.",
    )
    parser.add_argument(
        "--n-concurrent", type=int, default=8, help="Number of concurrent processes"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1,
        help="Maximum number of recovery iterations",
    )
    parser.add_argument(
        "--run-initial",
        action="store_true",
        default=False,
        help="Just run initial traces, skip recovery",
    )
    parser.add_argument(
        "--cleanup-container",
        action="store_true",
        default=False,
        help="Cleanup container before running",
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default="2.0",
        help="Dataset version to use for initial trace generation (e.g., 2.0)",
    )
    parser.add_argument(
        "--resume-initial",
        type=str,
        default=None,
        help="Path to existing initial traces (skips initial generation, runs recovery only)",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        action="append",
        dest="task_ids",
        help="Specific task ID(s) to run (can be specified multiple times)",
    )
    parser.add_argument(
        "--initial-agent",
        type=str,
        default=None,
        help="Agent for initial runs: built-in name (e.g., terminus-2) or import path (e.g., recovery_bench.letta_code_agent:LettaCode). Defaults to terminus-2.",
    )
    parser.add_argument(
        "--recovery-agent",
        type=str,
        default="recovery_bench.recovery_terminus:RecoveryTerminus",
        help="Agent import path for recovery (e.g., recovery_bench.recovery_letta_code:RecoveryLettaCode)",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default=None,
        help="Custom job name for recovery output (auto-generated if not specified)",
    )

    args = parser.parse_args()

    # Validate: need --initial-model unless resuming from existing traces
    if not args.resume_initial and not args.initial_model:
        parser.error("--initial-model is required unless --resume-initial is used")

    # Validate: need --recovery-model for recovery when resuming
    if args.resume_initial and not args.recovery_model:
        parser.error("--recovery-model is required when using --resume-initial")

    # Resolve initial model config
    if args.initial_model:
        initial_model, initial_model_kwargs = resolve_model(args.initial_model)
    else:
        initial_model, initial_model_kwargs = None, {}

    # Resolve recovery model (only if specified)
    if args.recovery_model:
        recovery_model, recovery_model_kwargs = resolve_model(args.recovery_model)
    else:
        recovery_model, recovery_model_kwargs = None, {}

    # Generate timestamp for unique run IDs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.cleanup_container:
        cleanup_docker()

    # Step 1: Generate initial traces or resume from existing
    if args.resume_initial:
        initial_traces_dir = args.resume_initial
        logger.info(
            f"Resuming from existing initial trajectories at {initial_traces_dir}"
        )
    else:
        model_short = shorten_model_name(initial_model)
        initial_run_id = f"initial-{model_short}-{timestamp}"
        initial_traces_dir = generate_initial_traces(
            initial_model,
            initial_run_id,
            args.dataset_version,
            args.n_concurrent,
            args.task_ids,
            args.initial_agent,
            initial_model_kwargs,
        )

    if args.run_initial or not recovery_model:
        logger.info(f"Initial traces complete: {initial_traces_dir}")
        return 0

    # Step 2: Reorganize initial traces with hash prefixes
    reorganize_directories(initial_traces_dir)

    # Keep track of all trace directories
    all_trace_dirs = [initial_traces_dir]

    # Step 3: Iteratively run recovery agent on unsolved tasks
    current_traces_dir = initial_traces_dir
    for iteration in range(1, args.max_iterations + 1):
        logger.info(f"--- Starting iteration {iteration} ---")

        # Build job name
        if args.job_name and args.max_iterations == 1:
            recovery_job_name = args.job_name
        else:
            recovery_agent_name = get_agent_name(args.recovery_agent)
            recovery_model_short = shorten_model_name(recovery_model)
            recovery_job_name = (
                args.job_name or f"{recovery_agent_name}-{recovery_model_short}-{timestamp}"
            )
            if args.max_iterations > 1:
                recovery_job_name = f"{recovery_job_name}-iter{iteration}"

        recovery_traces_dir, rc = run_recovery_pipeline(
            traces_folder=current_traces_dir,
            model=recovery_model,
            job_name=recovery_job_name,
            agent=args.recovery_agent,
            n_concurrent=args.n_concurrent,
            task_ids=args.task_ids if args.resume_initial else None,
            model_kwargs=recovery_model_kwargs,
            reorganize=False,  # Already reorganized above (or by previous iteration)
        )

        if not recovery_traces_dir:
            logger.info(f"No unsolved tasks found in iteration {iteration}, stopping.")
            break

        # Hash reorganize the new traces for next iteration
        reorganize_directories(recovery_traces_dir)

        all_trace_dirs.append(recovery_traces_dir)
        current_traces_dir = recovery_traces_dir

    logger.info("--- Pipeline completed successfully! ---")
    logger.info(f"Initial traces: {initial_traces_dir}")
    logger.info(f"All trace directories: {all_trace_dirs}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    sys.exit(main())
