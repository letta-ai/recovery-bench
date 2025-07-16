#!/usr/bin/env python3
"""
Generate traces pipeline script.

This script automates the entire trace generation pipeline:
1. Generate initial traces
2. Get task IDs that failed but without 10 (max_episode) traces
3. Run replay agent on those tasks
4. Filter again to get unsolved tasks
5. Collect all traces into a single folder
6. Named after model_time

Usage: python generate_traces.py <model_name>
Example: python generate_traces.py anthropic/claude-3-5-haiku-20241022
"""

import argparse
import subprocess
import sys
import os
import json
from pathlib import Path
from datetime import datetime
import time
from typing import List, Optional

from .register_unsolved_tasks import get_unsolved_tasks
from .collect_traces import collect_traces
from .run_replay_agent import run_replay_agent
from .hash_reorganize import reorganize_directories


def run_command(cmd: List[str], env: dict = None, cwd: str = None):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    if env is None:
        env = os.environ.copy()
    
    result = subprocess.run(cmd, env=env, cwd=cwd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        sys.exit(1)
    return result


def generate_initial_traces(model_name: str, run_id: str) -> str:
    """Generate initial traces using tb run."""
    print(f"Generating initial traces for {model_name}...")
    
    cmd = [
        "tb", "run",
        "--dataset-name", "terminal-bench-core",
        "--dataset-version", "head",
        "--agent", "terminus",
        "--model-name", model_name,
        "--run-id", run_id,
        "--n-concurrent", "6",
        "--cleanup"
    ]
    
    run_command(cmd)
    return f"runs/{run_id}"


def run_replay_agent_for_unsolved(
    trajectory_folder: str,
    model_name: str,
    run_id: str,
    max_episodes: int = 10,
    n_concurrent: int = 4
) -> str:
    """Run replay agent for unsolved tasks."""
    print(f"Running replay agent for unsolved tasks in {trajectory_folder}...")
    
    # Get unsolved task IDs
    unsolved_task_ids = get_unsolved_tasks(trajectory_folder, max_episodes_desired=max_episodes)
    
    if not unsolved_task_ids:
        print("No unsolved tasks found, skipping replay agent.")
        return trajectory_folder
    
    print(f"Found {len(unsolved_task_ids)} unsolved tasks")
    
    # Use the run_replay_agent function instead of direct tb command
    run_replay_agent(
        trajectory_folder=trajectory_folder,
        model_name=model_name,
        task_ids=unsolved_task_ids,
        run_id=run_id,
        agent_import_path="self-correction.replay_agent:ReplayAgent",
        n_concurrent=n_concurrent
    )
    
    return f"runs/{run_id}"


def collect_all_traces(logs_dirs: List[str], output_dir: str, min_episodes: int = 10):
    """Collect traces from multiple directories."""
    print(f"Collecting traces from {len(logs_dirs)} directories into {output_dir}...")
    
    # Convert to Path objects as required by the function
    logs_dirs_paths = [Path(d) for d in logs_dirs]
    output_dir_path = Path(output_dir)
    
    # Use the function directly instead of subprocess
    collect_traces(logs_dirs_paths, output_dir_path, min_episodes)


def main():
    parser = argparse.ArgumentParser(description="Generate traces pipeline")
    parser.add_argument("model_name", help="Model name to use for trace generation")
    parser.add_argument("--task-folder", default="/home/kevinlin/terminal-bench/tasks", 
                       help="Path to task folder")
    parser.add_argument("--max-episodes", type=int, default=10, 
                       help="Maximum episodes for filtering")
    parser.add_argument("--n-concurrent", type=int, default=4, 
                       help="Number of concurrent processes")
    parser.add_argument("--max-iterations", type=int, default=3, 
                       help="Maximum number of replay iterations")
    parser.add_argument("--min-episodes", type=int, default=10, 
                       help="Minimum episodes for collection")
    
    args = parser.parse_args()
    
    # Generate timestamp for unique run IDs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model_name.split("/")[-1]
    
    # Step 1: Generate initial traces
    initial_run_id = f"initial-{model_short}-{timestamp}"
    initial_traces_dir = generate_initial_traces(args.model_name, initial_run_id)
    
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
        unsolved_task_ids = get_unsolved_tasks(current_traces_dir, max_episodes_desired=args.max_episodes)
        
        if not unsolved_task_ids:
            print(f"No unsolved tasks found in iteration {iteration}, stopping.")
            break
        
        print(f"Found {len(unsolved_task_ids)} unsolved tasks for iteration {iteration}")
        
        # Run replay agent
        replay_run_id = f"replay-{model_short}-{timestamp}-iter{iteration}"
        replay_traces_dir = run_replay_agent_for_unsolved(
            current_traces_dir,
            args.model_name,
            replay_run_id,
            args.max_episodes,
            args.n_concurrent
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
