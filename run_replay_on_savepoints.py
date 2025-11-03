#!/usr/bin/env python3
"""
Run ReplayAgent on save_points episodes.
Bypasses reorganization to preserve hash-prefixed directory structure.
"""

import subprocess
import os
import time
from pathlib import Path

def main():
    # Set environment - ReplayAgent reads from TRAJECTORY_FOLDER
    env = os.environ.copy()
    env["TRAJECTORY_FOLDER"] = str(Path('save_points').absolute())
    
    # Generate run ID
    run_id = f"replay-gpt-5-savepoints-{int(time.time())}"
    
    # Tasks we saved (task names from registry, not hash-prefixed)
    task_ids = ["chess-best-move", "parallelize-graph"]
    
    # Build command using tb run directly with correct --model argument
    # This bypasses run_replay_agent.py reorganization
    cmd = [
        "tb", "run",
        "--dataset-name", "terminal-bench-core",
        "--dataset-version", "0.2.15",
        "--agent-import-path", "recovery-bench.replay_agent:ReplayAgent",
        "--model", "gpt-5",  # Correct argument name (not --model-name)
        "--n-concurrent", "1",
        "--global-timeout-multiplier", "1.0",
        "--local-registry-path", "./registry.json",
        "--run-id", run_id,
        "--cleanup"
    ]
    
    # Add task IDs
    for task in task_ids:
        cmd.extend(["--task-id", task])
    
    print(f"\n🚀 Running ReplayAgent on save_points episodes")
    print(f"Tasks: {task_ids}")
    print(f"Trajectory Folder: {env['TRAJECTORY_FOLDER']}")
    print(f"Command: {' '.join(cmd)}")
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

