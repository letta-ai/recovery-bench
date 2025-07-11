#!/usr/bin/env python3

import json
import os
import shutil
from pathlib import Path
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

def find_task_info(logs_dir: Path) -> Dict[str, Tuple[Path, int, bool]]:
    """
    Find all tasks in a logs directory and return their info.
    
    Returns:
        Dict mapping task_id -> (task_dir_path, episode_count, is_resolved)
    """
    task_info = {}
    
    for task_id in os.listdir(logs_dir):
        task_path = logs_dir / task_id
        if not task_path.is_dir():
            continue
            
        # Find the directory that contains "1-of-1" in its name
        matching_dirs = [d for d in task_path.iterdir() if d.is_dir() and "1-of-1" in d.name]
        
        if not matching_dirs:
            continue
            
        # Use the first matching directory
        target_dir = matching_dirs[0]
        
        # Read the results.json for the task
        results_path = target_dir / "results.json"
        if not results_path.exists():
            continue
            
        with open(results_path, "r") as f:
            results = json.load(f)
        
        is_resolved = results.get("is_resolved", False)
        actual_task_id = results.get("task_id", task_id)
        
        # Count episodes in agent-logs directory
        agent_logs_dir = target_dir / "agent-logs"
        episode_count = 0
        if agent_logs_dir.exists() and agent_logs_dir.is_dir():
            episode_count = len(list(agent_logs_dir.iterdir()))
        
        task_info[actual_task_id] = (target_dir, episode_count, is_resolved)
    
    return task_info

def collect_traces(logs_dirs: List[Path], output_dir: Path, min_episodes: int = 10):
    """
    Collect traces from multiple logs directories, keeping only unresolved tasks
    and selecting the version with the most episodes for each task.
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect task info from all directories
    all_task_info = defaultdict(list)
    
    for logs_dir in logs_dirs:
        print(f"Scanning {logs_dir}...")
        task_info = find_task_info(logs_dir)
        
        for task_id, (task_dir, episode_count, is_resolved) in task_info.items():
            all_task_info[task_id].append((task_dir, episode_count, is_resolved, logs_dir))
    
    # Select the best version of each task (most episodes, unresolved only)
    selected_tasks = {}
    total_tasks = 0
    
    for task_id, versions in all_task_info.items():
        # Filter to only unresolved tasks
        unresolved_versions = [(task_dir, episode_count, is_resolved, source_dir) 
                              for task_dir, episode_count, is_resolved, source_dir in versions 
                              if not is_resolved]
        
        if not unresolved_versions:
            continue
        
        # Select the version with the most episodes
        best_version = max(unresolved_versions, key=lambda x: x[1])
        
        # Check if the best version meets the minimum episodes requirement
        if best_version[1] < min_episodes:
            continue
        
        selected_tasks[task_id] = best_version
        total_tasks += 1
    
    print(f"Found {total_tasks} unresolved tasks to collect (with >= {min_episodes} episodes)")
    
    # Copy selected tasks to output directory
    for task_id, (source_task_dir, episode_count, _, source_logs_dir) in selected_tasks.items():
        # Create the task directory structure in output
        task_output_dir = output_dir / task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy the entire task directory structure
        target_name = source_task_dir.name
        dest_path = task_output_dir / target_name
        
        if dest_path.exists():
            shutil.rmtree(dest_path)
        
        shutil.copytree(source_task_dir, dest_path)
        
        print(f"{task_id}: {episode_count} episodes (from {source_logs_dir})")
    
    print(f"\nTotal tasks collected: {total_tasks}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect traces from multiple logs directories")
    parser.add_argument("--logs-dirs", nargs="+", required=True, 
                       help="List of logs directories to merge")
    parser.add_argument("--output-dir", required=True,
                       help="Output directory for collected traces")
    parser.add_argument("--min-episodes", type=int, default=10,
                       help="Minimum number of episodes to collect")

    
    args = parser.parse_args()
    
    logs_dirs = [Path(d) for d in args.logs_dirs]
    output_dir = Path(args.output_dir)
    
    # Validate input directories
    for logs_dir in logs_dirs:
        if not logs_dir.exists():
            print(f"Error: Directory {logs_dir} does not exist")
            exit(1)
        if not logs_dir.is_dir():
            print(f"Error: {logs_dir} is not a directory")
            exit(1)
    
    collect_traces(logs_dirs, output_dir, args.min_episodes) 