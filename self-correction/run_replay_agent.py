#!/usr/bin/env python3

import argparse
import subprocess
import sys
import os
from pathlib import Path
from typing import List

from .register_unsolved_tasks import get_unsolved_tasks
from .hash_reorganize import reorganize_directories, reverse_reorganize_directories
from .utils import cleanup_docker

def run_replay_agent(
    trajectory_folder: str,
    model_name: str,
    task_ids: List[str],
    run_id: str | None = None,
    dataset_name: str = "terminal-bench-core",
    dataset_version: str = "head",
    agent_import_path: str = "self-correction.replay_agent:ReplayAgent",
    n_concurrent: int = 1,
    global_timeout_multiplier: float = 2.0,
    additional_args: List[str] | None = None
):
    """
    Run the replay agent for multiple task IDs using tb run command.
    
    Args:
        trajectory_folder: Path to the trajectory folder
        model_name: Name of the model to use
        task_ids: List of task IDs to run
        run_id: Optional run ID
        dataset_name: Name of the dataset
        dataset_version: Version of the dataset
        agent_import_path: Import path for the agent
        n_concurrent: Number of concurrent processes
        additional_args: Additional arguments to pass to tb run
    """
    
    # Clean up Docker before running tb
    cleanup_docker()
    
    # Set environment variable
    env = os.environ.copy()
    env["TRAJECTORY_FOLDER"] = trajectory_folder
    
    # Build base command
    cmd = [
        "tb", "run",
        "--dataset-name", dataset_name,
        "--dataset-version", dataset_version,
        "--agent-import-path", agent_import_path,
        "--model-name", model_name,
        "--n-concurrent", str(n_concurrent),
        "--global-timeout-multiplier", str(global_timeout_multiplier), # because we are replaying, we need to give it more time
    ]
    
    # Add run ID if provided
    if run_id:
        cmd.extend(["--run-id", run_id])
    
    # Add task IDs
    for task_id in task_ids:
        cmd.extend(["--task-id", task_id])
    
    # Add additional arguments if provided
    if additional_args:
        cmd.extend(additional_args)

    cmd.extend(["--cleanup"])
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Environment TRAJECTORY_FOLDER: {trajectory_folder}")
    print(f"Task IDs: {task_ids}")
    
    try:
        # Run the command
        result = subprocess.run(cmd, env=env, check=True)
        print(f"Command completed successfully with return code: {result.returncode}")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code: {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"Error running command: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(description="Run replay agent for multiple task IDs")
    parser.add_argument("--trajectory-folder", type=str, required=True,
                        help="Path to the trajectory folder")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Name of the model to use")
    
    # General options
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run ID (optional)")
    parser.add_argument("--dataset-name", type=str, default="terminal-bench-core",
                        help="Name of the dataset")
    parser.add_argument("--dataset-version", type=str, default="head",
                        help="Version of the dataset")
    parser.add_argument("--agent-import-path", type=str, 
                        default="self-correction.replay_agent:ReplayAgent",
                        help="Import path for the agent")
    parser.add_argument("--n-concurrent", type=int, default=1,
                        help="Number of concurrent processes")
    parser.add_argument("--task-folder", type=str, default=None,
                        help="Path to the task folder for reorganization (defaults to trajectory folder)")
    
    args, unknown_args = parser.parse_known_args()
    
    trajectory_path = Path(args.trajectory_folder)
    if not trajectory_path.exists():
        print(f"Error: Trajectory folder {args.trajectory_folder} does not exist")
        return 1
    
    # Get unsolved task IDs from trajectory folder
    print(f"Getting unsolved task IDs from {args.trajectory_folder}...")
    task_ids = get_unsolved_tasks(args.trajectory_folder)

    # Use task_folder if provided, otherwise use None to rely on environment variable
    task_folder = args.task_folder if args.task_folder else None
    reverse_reorganize_directories(args.trajectory_folder)
    reorganize_directories(args.trajectory_folder, task_folder)
    
    if not task_ids:
        print("No unsolved task IDs found")
        return 1
    
    print(f"Found {len(task_ids)} unsolved task IDs")
    
    print(f"Running replay agent for {len(task_ids)} task(s)")
    
    return run_replay_agent(
        trajectory_folder=args.trajectory_folder,
        model_name=args.model_name,
        task_ids=task_ids,
        run_id=args.run_id,
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
        agent_import_path=args.agent_import_path,
        n_concurrent=args.n_concurrent,
        additional_args=unknown_args
    )

if __name__ == "__main__":
    sys.exit(main()) 