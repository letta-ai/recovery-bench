#!/usr/bin/env python3
"""
Generate recovery-bench traces.

This script automates the entire trace generation pipeline:
1. Generate initial traces with an agent (default: terminus-2)
2. Reorganize traces with hash prefixes for recovery lookup
3. Iteratively run recovery agent on unsolved tasks

Usage: python -m recovery_bench.generate_traces <model_name>

Examples:
    # Default agents (terminus-2 initial, RecoveryTerminus recovery)
    python -m recovery_bench.generate_traces openai/gpt-4o-mini --task-id cancel-async-tasks

    # Custom agents (LettaCode initial, RecoveryLettaCode recovery)
    python -m recovery_bench.generate_traces openai/gpt-4o-mini \\
        --initial-agent recovery_bench.letta_code_agent:LettaCode \\
        --recovery-agent recovery_bench.recovery_letta_code:RecoveryLettaCode \\
        --task-id constraints-scheduling
"""

import argparse
from datetime import datetime
from typing import List

from .utils import (
    cleanup_docker,
    get_unsolved_tasks,
    reorganize_directories,
    run_command,
    run_recovery,
)


def generate_initial_traces(
    model_name: str,
    run_id: str,
    dataset_version: str = "2.0",
    n_concurrent: int = 6,
    task_ids: List[str] | None = None,
    agent_import_path: str | None = None,
) -> str:
    """Generate initial traces using harbor run."""
    print(f"Generating initial traces for {model_name}...")

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

    run_command(cmd)
    return f"jobs/{run_id}"


def run_recovery_for_unsolved(
    traces_folder: str,
    model: str,
    job_name: str,
    n_concurrent: int = 4,
    agent: str = "recovery_bench.recovery_terminus:RecoveryTerminus",
) -> str:
    """Run recovery agent for unsolved tasks."""
    print(f"Running recovery for unsolved tasks in {traces_folder}...")

    unsolved_task_ids = get_unsolved_tasks(traces_folder)

    if not unsolved_task_ids:
        print("No unsolved tasks found, skipping recovery.")
        return traces_folder

    print(f"Found {len(unsolved_task_ids)} unsolved tasks")

    run_recovery(
        traces_folder=traces_folder,
        model=model,
        task_ids=unsolved_task_ids,
        job_name=job_name,
        agent=agent,
        n_concurrent=n_concurrent,
    )

    return f"jobs/{job_name}"


def main():
    parser = argparse.ArgumentParser(description="Generate traces pipeline")
    parser.add_argument("model_name", help="Model name to use for trace generation")
    parser.add_argument(
        "--n-concurrent", type=int, default=4, help="Number of concurrent processes"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum number of recovery iterations",
    )
    parser.add_argument(
        "--run-initial",
        action="store_true",
        default=False,
        help="Just run initial traces",
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
        help="Path to an existing initial trajectories directory to resume from (skips initial generation)",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        action="append",
        dest="task_ids",
        help="Specific task ID(s) to run (can be specified multiple times)",
    )
    parser.add_argument(
        "--recovery-model",
        type=str,
        default=None,
        help="Model to use for recovery (defaults to same as initial model)",
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

    args = parser.parse_args()

    # Generate timestamp for unique run IDs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model_name.split("/")[-1]

    if args.cleanup_container:
        cleanup_docker()

    # Step 1: Generate initial traces or resume from existing
    if args.resume_initial:
        initial_traces_dir = args.resume_initial
        print(
            f"Resuming from existing initial trajectories at {initial_traces_dir}"
        )
    else:
        initial_run_id = f"initial-{model_short}-{timestamp}"
        initial_traces_dir = generate_initial_traces(
            args.model_name,
            initial_run_id,
            args.dataset_version,
            args.n_concurrent,
            args.task_ids,
            args.initial_agent,
        )

    if args.run_initial:
        print(f"Just running initial traces for {args.model_name}")
        return

    # Step 2: Hash reorganize initial traces
    print(f"Reorganizing traces in {initial_traces_dir}...")
    reorganize_directories(initial_traces_dir)

    # Keep track of all trace directories
    all_trace_dirs = [initial_traces_dir]

    # Step 3: Iteratively run recovery agent on unsolved tasks
    current_traces_dir = initial_traces_dir
    for iteration in range(1, args.max_iterations + 1):
        print(f"\n--- Starting iteration {iteration} ---")

        # Get unsolved tasks from current directory
        unsolved_task_ids = get_unsolved_tasks(
            current_traces_dir
        )

        if not unsolved_task_ids:
            print(f"No unsolved tasks found in iteration {iteration}, stopping.")
            break

        print(
            f"Found {len(unsolved_task_ids)} unsolved tasks for iteration {iteration}"
        )

        # Run recovery
        recovery_model = args.recovery_model or args.model_name
        recovery_model_short = recovery_model.split("/")[-1]
        recovery_job_name = f"recovery-{recovery_model_short}-{timestamp}-iter{iteration}"
        recovery_traces_dir = run_recovery_for_unsolved(
            traces_folder=current_traces_dir,
            model=recovery_model,
            job_name=recovery_job_name,
            n_concurrent=args.n_concurrent,
            agent=args.recovery_agent,
        )

        # Hash reorganize the new traces
        print(f"Reorganizing traces in {recovery_traces_dir}...")
        reorganize_directories(recovery_traces_dir)

        # Add to list of all trace directories
        all_trace_dirs.append(recovery_traces_dir)

        # Update current directory for next iteration
        current_traces_dir = recovery_traces_dir

    print(f"\n--- Pipeline completed successfully! ---")
    print(f"Initial traces: {initial_traces_dir}")
    print(f"All trace directories: {all_trace_dirs}")


if __name__ == "__main__":
    main()
