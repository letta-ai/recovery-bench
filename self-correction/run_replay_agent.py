#!/usr/bin/env python3
"""
Run replay/recovery agent.

This script runs the replay agent on collected traces for self-correction.

Usage: python -m self_correction.run_replay_agent --trajectory-folder FOLDER --model-name MODEL
Example: python -m self_correction.run_replay_agent --trajectory-folder runs/collected --model-name anthropic/claude-sonnet-4-20250514
"""

import argparse
import sys
from pathlib import Path

from .utils import (
    get_unsolved_tasks,
    reorganize_directories,
    reverse_reorganize_directories,
    run_replay_agent_tb,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run replay agent for multiple task IDs"
    )
    parser.add_argument(
        "--trajectory-folder",
        type=str,
        required=True,
        help="Path to the trajectory folder",
    )
    parser.add_argument(
        "--model-name", type=str, required=True, help="Name of the model to use"
    )

    # General options
    parser.add_argument("--run-id", type=str, default=None, help="Run ID (optional)")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="terminal-bench-core",
        help="Name of the dataset",
    )
    parser.add_argument(
        "--dataset-version", type=str, default="0.2.15", help="Version of the dataset"
    )
    parser.add_argument(
        "--agent-import-path",
        type=str,
        default="self-correction.replay_agent:ReplayAgent",
        help="Import path for the agent",
    )
    parser.add_argument(
        "--n-concurrent", type=int, default=1, help="Number of concurrent processes"
    )
    parser.add_argument(
        "--task-folder",
        type=str,
        default="./terminal-bench/tasks",
        help="Path to the task folder for reorganization",
    )
    parser.add_argument(
        "--cleanup-container",
        action="store_true",
        default=False,
        help="Cleanup container before running",
    )
    parser.add_argument(
        "--task-ids",
        nargs="+",
        type=str,
        default=None,
        help="One or more task IDs to run the replay agent, usually used for debugging",
    )
    args, unknown_args = parser.parse_known_args()

    trajectory_path = Path(args.trajectory_folder)
    if not trajectory_path.exists():
        print(f"Error: Trajectory folder {args.trajectory_folder} does not exist")
        return 1

    # Get unsolved task IDs from trajectory folder
    print(f"Getting unsolved task IDs from {args.trajectory_folder}...")
    task_ids = (
        get_unsolved_tasks(args.trajectory_folder)
        if not args.task_ids
        else args.task_ids
    )

    # Use task_folder if provided, otherwise use None to rely on environment variable
    task_folder = args.task_folder if args.task_folder else None
    reverse_reorganize_directories(args.trajectory_folder)
    reorganize_directories(args.trajectory_folder, task_folder)

    if not task_ids:
        print("No unsolved task IDs found")
        return 1

    print(f"Found {len(task_ids)} unsolved task IDs")

    print(f"Running replay agent for {len(task_ids)} task(s)")

    return run_replay_agent_tb(
        trajectory_folder=args.trajectory_folder,
        model_name=args.model_name,
        task_ids=task_ids,
        run_id=args.run_id,
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
        agent_import_path=args.agent_import_path,
        n_concurrent=args.n_concurrent,
        additional_args=unknown_args,
        cleanup_container=args.cleanup_container,
    )


if __name__ == "__main__":
    sys.exit(main())
