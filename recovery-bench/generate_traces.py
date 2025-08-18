#!/usr/bin/env python3
"""
Generate recovery-bench traces.

This script automates the entire trace generation pipeline:
1. Generate initial traces
2. Get task IDs that failed but without 10 (max_episode) traces
3. Run replay agent on those tasks
4. Filter again to get unsolved tasks
5. Collect all traces into a single folder

Usage: python -m self_correction.generate_traces <model_name>
Example: python -m self_correction.generate_traces anthropic/claude-3-5-haiku-20241022
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import List

from .utils import (
    cleanup_docker,
    get_unsolved_tasks,
    collect_traces,
    reorganize_directories,
    run_command,
    run_replay_agent_tb,
)


def generate_initial_traces(
    model_name: str, run_id: str, dataset_version: str = "0.2.15", n_concurrent: int = 6
) -> str:
    """Generate initial traces using tb run."""
    print(f"Generating initial traces for {model_name}...")

    cmd = [
        "tb",
        "run",
        "--dataset-name",
        "terminal-bench-core",
        "--dataset-version",
        dataset_version,
        "--agent",
        "terminus",
        "--model-name",
        model_name,
        "--run-id",
        run_id,
        "--n-concurrent",
        str(n_concurrent),
        "--cleanup",
        "--local-registry-path",
        "./registry.json",
    ]

    run_command(cmd)
    return f"runs/{run_id}"


def run_replay_agent_for_unsolved(
    trajectory_folder: str,
    model_name: str,
    run_id: str,
    min_episodes: int = 10,
    n_concurrent: int = 4,
) -> str:
    """Run replay agent for unsolved tasks."""
    print(f"Running replay agent for unsolved tasks in {trajectory_folder}...")

    unsolved_task_ids = get_unsolved_tasks(
        trajectory_folder, min_episodes_desired=min_episodes
    )

    if not unsolved_task_ids:
        print("No unsolved tasks found, skipping replay agent.")
        return trajectory_folder

    print(f"Found {len(unsolved_task_ids)} unsolved tasks")

    run_replay_agent_tb(
        trajectory_folder=trajectory_folder,
        model_name=model_name,
        task_ids=unsolved_task_ids,
        run_id=run_id,
        agent_import_path="recovery-bench.replay_agent:ReplayAgent",
        n_concurrent=n_concurrent,
    )

    return f"runs/{run_id}"


def collect_all_traces(logs_dirs: List[str], output_dir: str, min_episodes: int = 10):
    """Collect traces from multiple directories."""
    print(f"Collecting traces from {len(logs_dirs)} directories into {output_dir}...")

    logs_dirs_paths = [Path(d) for d in logs_dirs]
    output_dir_path = Path(output_dir)

    collect_traces(logs_dirs_paths, output_dir_path, min_episodes)


def main():
    parser = argparse.ArgumentParser(description="Generate traces pipeline")
    parser.add_argument("model_name", help="Model name to use for trace generation")
    parser.add_argument(
        "--task-folder",
        default="./terminal-bench/tasks",
        help="Path to task folder",
    )
    parser.add_argument(
        "--n-concurrent", type=int, default=4, help="Number of concurrent processes"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum number of replay iterations",
    )
    parser.add_argument(
        "--min-episodes",
        type=int,
        default=10,
        help="Minimum episodes for filtering and collection",
    )  # If you just want to run initial traces, set this to True
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
        default="0.2.15",
        help="Dataset version to use for initial trace generation",
    )

    args = parser.parse_args()

    # Generate timestamp for unique run IDs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model_name.split("/")[-1]

    if args.cleanup_container:
        cleanup_docker()

    # Step 1: Generate initial traces
    initial_run_id = f"initial-{model_short}-{timestamp}"
    initial_traces_dir = generate_initial_traces(
        args.model_name, initial_run_id, args.dataset_version, args.n_concurrent
    )

    if args.run_initial:
        print(f"Just running initial traces for {args.model_name}")
        return

    # Step 2: Hash reorganize initial traces
    print(f"Reorganizing traces in {initial_traces_dir}...")
    reorganize_directories(initial_traces_dir, args.task_folder)

    # Keep track of all trace directories
    all_trace_dirs = [initial_traces_dir]

    # Step 3: Iteratively run replay agent on unsolved tasks
    current_traces_dir = initial_traces_dir
    for iteration in range(1, args.max_iterations + 1):
        print(f"\n--- Starting iteration {iteration} ---")

        # Get unsolved tasks from current directory
        unsolved_task_ids = get_unsolved_tasks(
            current_traces_dir, min_episodes_desired=args.min_episodes
        )

        if not unsolved_task_ids:
            print(f"No unsolved tasks found in iteration {iteration}, stopping.")
            break

        print(
            f"Found {len(unsolved_task_ids)} unsolved tasks for iteration {iteration}"
        )

        # Run replay agent
        replay_run_id = f"replay-{model_short}-{timestamp}-iter{iteration}"
        replay_traces_dir = run_replay_agent_for_unsolved(
            current_traces_dir,
            args.model_name,
            replay_run_id,
            args.min_episodes,
            args.n_concurrent,
        )

        # Hash reorganize the new traces
        print(f"Reorganizing traces in {replay_traces_dir}...")
        reorganize_directories(replay_traces_dir, args.task_folder)

        # Add to list of all trace directories
        all_trace_dirs.append(replay_traces_dir)

        # Update current directory for next iteration
        current_traces_dir = replay_traces_dir

    # Step 4: Collect all traces into single folder
    collected_dir = f"runs/{model_short}-collected-{timestamp}"
    collect_all_traces(all_trace_dirs, collected_dir, args.min_episodes)

    print(f"\n--- Pipeline completed successfully! ---")
    print(f"Initial traces: {initial_traces_dir}")
    print(f"All trace directories: {all_trace_dirs}")
    print(f"Final collected traces: {collected_dir}")

    return collected_dir


if __name__ == "__main__":
    main()
