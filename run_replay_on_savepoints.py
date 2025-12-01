#!/usr/bin/env python3
"""
Run ReplayAgent on save_points episodes.
Bypasses reorganization to preserve hash-prefixed directory structure.
"""

import subprocess
import os
import time
from pathlib import Path

def ensure_package_structure():
    """Ensure recovery_bench is a proper Python package."""
    init_file = Path(__file__).parent / "recovery_bench" / "__init__.py"
    if not init_file.exists():
        print(f"Creating {init_file}")
        init_file.touch()
        return True
    return False

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
    
    if ensure_package_structure():
        print("Created __init__.py for recovery_bench package")
    
    script_dir = Path(__file__).parent.absolute()
    env = os.environ.copy()
    env["TRAJECTORY_FOLDER"] = str(script_dir / 'save_points')
    
    # set rootless Docker environment variables  
    import getpass
    user_id = os.getuid()
    env["XDG_RUNTIME_DIR"] = f"/run/user/{user_id}"
    env["DOCKER_HOST"] = f"unix:///run/user/{user_id}/docker.sock"
    
    # add recovery-bench to PYTHONPATH
    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        env["PYTHONPATH"] = f"{script_dir}:{pythonpath}"
    else:
        env["PYTHONPATH"] = str(script_dir)
    
    # generate run ID
    run_id = f"replay-{args.model}-savepoints-{int(time.time())}"
    
    # get all tasks from save_points
    task_ids = get_all_tasks_from_savepoints()
    
    if not task_ids:
        print("No tasks found in save_points directory")
        return 1
    
    print(f"📋 Found {len(task_ids)} tasks in save_points")
    
    agent_path = "recovery_bench.replay_agent:ReplayAgent"
    print("🔷 Using standard ReplayAgent (Docker/Tmux)")
    
    # Build command using tb run directly
    tb_cmd = str(script_dir / "venv" / "bin" / "tb")
    cmd = [
        tb_cmd, "run",
        "--dataset-path", "./terminal-bench/tasks",
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
    print(f"🔧 PYTHONPATH: {env['PYTHONPATH']}")
    print(f"🔧 Command: {' '.join(cmd[:10])}... (with {len(task_ids)} task IDs)")
    print()
    
    # Change to script directory before running
    os.chdir(script_dir)
    
    try:
        result = subprocess.run(cmd, env=env, check=True)
        print(f"\n ReplayAgent completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\nReplayAgent failed with return code: {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"\nError running ReplayAgent: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
