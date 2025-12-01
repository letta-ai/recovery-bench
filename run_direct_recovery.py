#!/usr/bin/env python3
"""
Direct recovery agent runner that bypasses reorganization and runs directly on save_points.
"""

import subprocess
import os
import json
import time
from pathlib import Path

def get_all_recovery_tasks():
    """Get ALL tasks that have recovery traces in save_points, regardless of registry."""
    recovery_tasks = []
    save_points_path = Path('save_points')
    
    for task_dir in save_points_path.iterdir():
        if task_dir.is_dir():
            original_task_name = task_dir.name
            # Check for recovery_trace.json (don't filter by registry)
            found_recovery_trace = False
            for sub_dir in task_dir.iterdir():
                if sub_dir.is_dir() and "1-of-1" in sub_dir.name:
                    agent_logs_path = sub_dir / "agent-logs"
                    if agent_logs_path.is_dir():
                        for episode_dir in agent_logs_path.iterdir():
                            if episode_dir.is_dir() and episode_dir.name.startswith("episode-"):
                                if (episode_dir / "recovery_trace.json").exists():
                                    recovery_tasks.append(original_task_name)
                                    found_recovery_trace = True
                                    break
                        if found_recovery_trace:
                            break
            # Don't break - continue to find all tasks
    
    return sorted(list(set(recovery_tasks)))

def main():
    # Get ALL recovery tasks from save_points
    tasks = get_all_recovery_tasks()
    print(f"Found {len(tasks)} recovery tasks: {tasks}")
    
    if not tasks:
        print("No recovery tasks found!")
        return 1
    
    # Set environment
    env = os.environ.copy()
    env["TRAJECTORY_FOLDER"] = str(Path('save_points').absolute())
    
    # Generate run ID
    run_id = f"recovery-direct-{len(tasks)}tasks-{int(time.time())}"
    
    # Build command using tb run directly (bypassing reorganization)
    cmd = [
        "tb", "run",
        "--dataset-name", "terminal-bench-core",
        "--dataset-version", "0.2.15",
        "--agent-import-path", "recovery_agent:RecoveryAgent",
        "--model", "gpt-5",
        "--n-concurrent", "1",
        "--global-timeout-multiplier", "1.0",
        "--local-registry-path", "./registry.json",
        "--run-id", run_id,
        "--cleanup"
    ]
    
    # Add task IDs
    for task in tasks:
        cmd.extend(["--task-id", task])
    
    print(f"\nRunning recovery agent directly on {len(tasks)} tasks: {tasks}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Environment TRAJECTORY_FOLDER: {env['TRAJECTORY_FOLDER']}")
    
    try:
        result = subprocess.run(cmd, env=env, check=True)
        print(f"Recovery agent completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Recovery agent failed with return code: {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"Error running recovery agent: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
