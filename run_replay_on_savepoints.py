#!/usr/bin/env python3
"""
Run ReplayAgent on save_points episodes.
Bypasses reorganization to preserve hash-prefixed directory structure.
"""

import subprocess
import os
import time
from pathlib import Path

def get_all_tasks_from_savepoints():
    """Extract all task names from save_points directory.
    
    Directory format: {hash}-{task-name}
    Returns: list of task names (without hash prefix)
    """
    save_points_path = Path('save_points')
    task_names = []
    
    if not save_points_path.exists():
        print("Error: save_points directory does not exist")
        return []
    
    for task_dir in save_points_path.iterdir():
        if task_dir.is_dir() and not task_dir.name.startswith('.'):
            # Extract task name from hash-prefixed directory
            # Format: "hash-task-name" -> "task-name"
            if "-" in task_dir.name:
                # Split on first dash to separate hash from task name
                parts = task_dir.name.split("-", 1)
                if len(parts) == 2:
                    hash_part = parts[0]
                    task_name = parts[1]
                    # Verify it looks like a hash (8 hex chars) + task name
                    if len(hash_part) == 8 and all(c in "0123456789abcdef" for c in hash_part.lower()):
                        task_names.append(task_name)
                    else:
                        # If it doesn't match the pattern, use the whole name
                        task_names.append(task_dir.name)
                else:
                    task_names.append(task_dir.name)
            else:
                task_names.append(task_dir.name)
    
    return sorted(list(set(task_names)))

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ReplayAgent on save_points")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5",
        help="Model to use (default: gpt-5)"
    )
    parser.add_argument(
        "--n-concurrent",
        type=int,
        default=1,
        help="Number of concurrent tasks (default: 1)"
    )
    args = parser.parse_args()
    
    # Set environment - ReplayAgent reads from TRAJECTORY_FOLDER
    env = os.environ.copy()
    env["TRAJECTORY_FOLDER"] = str(Path('save_points').absolute())
    
    # Generate run ID
    run_id = f"replay-{args.model}-savepoints-{int(time.time())}"
    
    # Automatically get all tasks from save_points
    task_ids = get_all_tasks_from_savepoints()
    
    if not task_ids:
        print("❌ No tasks found in save_points directory")
        return 1
    
    print(f"📋 Found {len(task_ids)} tasks in save_points")
    
    # Use standard ReplayAgent
    agent_path = "recovery-bench.replay_agent:ReplayAgent"
    print("🔷 Using standard ReplayAgent (Docker/Tmux)")
    
    # Build command using tb run directly with correct --model argument
    # This bypasses run_replay_agent.py reorganization
    cmd = [
        "tb", "run",
        "--dataset-name", "terminal-bench-core",
        "--dataset-version", "0.2.15",
        "--agent-import-path", agent_path,
        "--model", args.model,
        "--n-concurrent", str(args.n_concurrent),
        "--global-timeout-multiplier", "1.0",
        "--local-registry-path", "./registry.json",
        "--run-id", run_id,
        "--cleanup"
    ]
    
    # Add task IDs
    for task in task_ids:
        cmd.extend(["--task-id", task])
    
    print(f"\n🚀 Running ReplayAgent on save_points episodes")
    print(f"📊 Total tasks: {len(task_ids)}")
    if len(task_ids) <= 10:
        print(f"📋 Tasks: {task_ids}")
    else:
        print(f"📋 First 10 tasks: {task_ids[:10]}...")
        print(f"📋 Last 10 tasks: ...{task_ids[-10:]}")
    print(f"📁 Trajectory Folder: {env['TRAJECTORY_FOLDER']}")
    print(f"🔧 Command: {' '.join(cmd[:10])}... (with {len(task_ids)} task IDs)")
    print()
    
    try:
        result = subprocess.run(cmd, env=env, check=True)
        print(f"\n✅ ReplayAgent completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ReplayAgent failed with return code: {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"\n❌ Error running ReplayAgent: {e}")
        return 1

if __name__ == "__main__":
    exit(main())

