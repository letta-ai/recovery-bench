# given a directory of logs, we want to register unsolved tasks in the registry.json

import json
import os
from pathlib import Path
import argparse

from .utils import create_task_hash

def get_unsolved_tasks(logs_dir: Path, max_episodes_desired: int = None, print_output: bool = False) -> list[str]:
    # max_eposides are used internally for registering tasks that are not solved but do not need additional episodes

    unsolved_ids = []

    for unsolved_id in os.listdir(logs_dir):
        # find the directory that contains "1-of-1" in its name
        if not (logs_dir / unsolved_id).is_dir():
            continue
        unsolved_dir = logs_dir / unsolved_id
        matching_dirs = [d for d in unsolved_dir.iterdir() if d.is_dir() and "1-of-1" in d.name]
        
        if not matching_dirs:
            continue
            
        # use the first matching directory
        target_dir = matching_dirs[0]
        
        # read the results.json for the unsolved id
        with open(target_dir / "results.json", "r") as f:
            results = json.load(f)
        if results["is_resolved"]:
            continue
        if not (target_dir / "agent-logs").is_dir():
            print(f"Skipping {target_dir} because it does not have an agent-logs directory")
            continue
        if max_episodes_desired is not None and len(list((target_dir / "agent-logs").iterdir())) > max_episodes_desired:
            print(f"Skipping {target_dir} because it has more than {max_episodes_desired} episodes")
            continue
        unsolved_ids.append(results["task_id"])    

    print(f"Found {len(unsolved_ids)} unsolved tasks")
    
    # Output the task IDs to console
    if print_output:
        print("Task IDs:")
        for task_id in unsolved_ids:
            print(f"  {task_id}")

    return unsolved_ids
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", type=str, required=True)
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    get_unsolved_tasks(logs_dir, args.max_episodes, print_output=True)