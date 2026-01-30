import hashlib
import subprocess
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Import harbor dependencies
try:
    from harbor.models.task.task import Task
except ImportError:
    Task = None


def create_task_hash(task_description: str) -> str:
    """Create 8-character hash from task description."""
    return hashlib.sha256(task_description.encode("utf-8")).hexdigest()[:8]


def cleanup_docker():
    """Clean up Docker containers and system resources."""
    print("Cleaning up Docker containers and system resources...")

    # Remove all containers (running and stopped)
    try:
        result = subprocess.run(
            "docker rm $(docker ps -aq) -f", shell=True, capture_output=True, text=True
        )
        if result.returncode == 0:
            print("Successfully removed Docker containers")
        else:
            print(f"Docker rm command output: {result.stderr}")
    except Exception as e:
        print(f"Error removing Docker containers: {e}")

    # Clean up Docker system
    try:
        result = subprocess.run(
            ["docker", "system", "prune", "-f"], capture_output=True, text=True
        )
        if result.returncode == 0:
            print("Successfully cleaned up Docker system")
        else:
            print(f"Docker system prune failed: {result.stderr}")
    except Exception as e:
        print(f"Error cleaning up Docker system: {e}")


def get_unsolved_tasks(
    logs_dir: str, min_episodes_desired: int = None, print_output: bool = False
) -> List[str]:
    """Get list of unsolved task IDs from a logs directory (ATIF format)."""
    logs_dir = Path(logs_dir)
    unsolved_ids = []

    for task_id in os.listdir(logs_dir):
        task_dir = logs_dir / task_id
        if not task_dir.is_dir():
            continue

        # Look for result.json (ATIF format)
        results_file = task_dir / "result.json"
        if not results_file.exists():
            continue
            
        try:
            with open(results_file, "r") as f:
                results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            continue

        # Check if resolved - ATIF uses verifier_result.rewards.reward
        # reward > 0 means resolved
        verifier_result = results.get("verifier_result", {})
        rewards = verifier_result.get("rewards", {})
        reward = rewards.get("reward", 0.0)
        if reward > 0:
            if print_output:
                print(f"Skipping {task_id}: resolved (reward={reward})")
            continue

        # Count episodes from agent_result metadata or trajectory
        episode_count = results.get("agent_result", {}).get("metadata", {}).get("n_episodes", 0)
        
        # Fallback: count from trajectory.json
        if episode_count == 0:
            trajectory_file = task_dir / "agent" / "trajectory.json"
            if not trajectory_file.exists():
                trajectory_file = task_dir / "trajectory.json"
            if trajectory_file.exists():
                try:
                    with open(trajectory_file, "r") as f:
                        trajectory = json.load(f)
                    steps = trajectory.get("steps", trajectory)
                    episode_count = sum(
                        1 for step in steps 
                        if step.get("source") == "agent" or step.get("role") == "assistant"
                    )
                except (json.JSONDecodeError, FileNotFoundError):
                    pass

        if episode_count == 0:
            if print_output:
                print(f"Skipping {task_id}: no episodes found")
            continue

        if (
            min_episodes_desired is not None
            and episode_count < min_episodes_desired
        ):
            if print_output:
                print(f"Skipping {task_id}: {episode_count} episodes < {min_episodes_desired}")
            continue
        
        # Extract task name from results
        task_name = results.get("task_name", task_id)
        unsolved_ids.append(task_name)

    if print_output:
        print(f"Found {len(unsolved_ids)} unsolved tasks")
        for task_id in unsolved_ids:
            print(f"  {task_id}")

    return unsolved_ids


def extract_task_description(
    task_name: str, task_folder: str | None = None
) -> str | None:
    """Extract task description from terminal-bench task.yaml file."""
    if Task is None:
        print("Warning: terminal_bench not available for task description extraction")
        return None

    try:
        folder = task_folder if task_folder is not None else os.getenv("TASK_FOLDER")
        if folder is None:
            print(f"Warning: No task folder specified for {task_name}")
            return None

        task_path = Path(folder) / task_name / "task.yaml"
        if not task_path.exists():
            print(f"Warning: Task file not found for {task_name}")
            return None

        task = Task.from_yaml(task_path)
        return task.instruction

    except Exception as e:
        print(f"Error reading task {task_name}: {e}")
        return None


def find_task_directories(base_path: str) -> Dict[str, str]:
    """Find task directories with their names."""
    task_dirs = {}
    base_path = Path(base_path)

    if not base_path.exists():
        print(f"Error: Path {base_path} does not exist")
        return task_dirs

    for item in base_path.iterdir():
        if (
            item.is_dir()
            and not item.name.startswith(".")
            and not item.name.endswith((".json", ".log", ".lock"))
        ):
            task_subdir = None
            for subitem in item.iterdir():
                if subitem.is_dir() and subitem.name.startswith(item.name + "."):
                    task_subdir = subitem
                    break

            if task_subdir:
                task_dirs[str(item)] = item.name
            else:
                task_dirs[str(item)] = item.name

    return task_dirs


def find_hash_directories(base_path: str) -> Dict[str, list]:
    """Find directories with hash prefixes."""
    hash_dirs = {}
    base_path = Path(base_path)

    if not base_path.exists():
        print(f"Error: Path {base_path} does not exist")
        return hash_dirs

    for item in base_path.iterdir():
        if (
            item.is_dir()
            and not item.name.startswith(".")
            and not item.name.endswith((".json", ".log", ".lock"))
        ):
            if len(item.name) > 9 and item.name[8] == "-":
                hash_part = item.name[:8]
                if all(c in "0123456789abcdef" for c in hash_part.lower()):
                    hash_dirs[hash_part] = [str(item)]

    return hash_dirs


def is_hash_prefixed_directory(task_dir: str, task_name: str) -> bool:
    """Check if directory is already hash-prefixed by inspecting subdirectory structure."""
    if len(task_name) > 9 and task_name[8] == "-":
        hash_part = task_name[:8]
        if all(c in "0123456789abcdef" for c in hash_part.lower()):
            name_after_dash = task_name[9:]
            task_path = Path(task_dir)
            for subitem in task_path.iterdir():
                if subitem.is_dir() and subitem.name.startswith(name_after_dash + "."):
                    return True
    return False


def extract_instruction_from_trajectory(task_dir: Path) -> str | None:
    """Extract instruction from trajectory.json (ATIF format)."""
    # Try agent/ subdirectory first (Harbor output structure)
    trajectory_file = task_dir / "agent" / "trajectory.json"
    if not trajectory_file.exists():
        # Fall back to direct path
        trajectory_file = task_dir / "trajectory.json"
    if not trajectory_file.exists():
        return None
    
    try:
        with open(trajectory_file, "r") as f:
            trajectory = json.load(f)
        
        # ATIF v1.5 format: steps array with source field
        if "steps" in trajectory:
            for step in trajectory["steps"]:
                if step.get("source") == "user":
                    return step.get("message", "")
        
        # Fallback: old format with role field
        for step in trajectory:
            if step.get("role") == "user":
                return step.get("content", "")
        return None
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def reorganize_directories(base_path: str) -> None:
    """Reorganize directories by adding task hash prefix based on trajectory instruction."""
    print(f"Reorganizing {base_path}")

    base_path = Path(base_path)
    if not base_path.exists():
        print("Path does not exist")
        return

    task_to_hash = {}

    for task_dir in base_path.iterdir():
        if not task_dir.is_dir():
            continue
        
        task_name = task_dir.name
        
        if is_hash_prefixed_directory(str(task_dir), task_name):
            print(f"  {task_name} -> SKIPPED (already has hash prefix)")
            continue

        instruction = extract_instruction_from_trajectory(task_dir)
        if instruction:
            task_hash = create_task_hash(instruction)
            task_to_hash[task_dir] = task_hash
            print(f"  {task_name} -> {task_hash}")
        else:
            print(f"  {task_name} -> SKIPPED (no trajectory.json or instruction)")

    for task_dir, task_hash in task_to_hash.items():
        task_name = task_dir.name
        new_task_dir = task_dir.parent / f"{task_hash}-{task_name}"

        try:
            shutil.move(str(task_dir), str(new_task_dir))
            print(f"    Renamed to {task_hash}-{task_name}")
        except Exception as e:
            print(f"    Error renaming {task_dir}: {e}")

    print(f"Processed {len(task_to_hash)} tasks")


def reverse_reorganize_directories(base_path: str) -> None:
    """Reverse reorganization by removing hash prefixes from directory names."""
    print(f"Reversing reorganization in {base_path}")

    hash_dirs = find_hash_directories(base_path)
    if not hash_dirs:
        print("No hash-prefixed directories found")
        return

    moved_count = 0

    for hash_prefix, task_dirs in hash_dirs.items():
        for task_dir in task_dirs:
            task_dir_path = Path(task_dir)
            task_name = task_dir_path.name

            original_name = task_name[9:]  # Remove "12345678-"
            target_dir = task_dir_path.parent / original_name

            try:
                shutil.move(str(task_dir), str(target_dir))
                print(f"    Renamed {task_name} back to {original_name}")
                moved_count += 1
            except Exception as e:
                print(f"    Error renaming {task_dir}: {e}")

    print(f"Processed {moved_count} tasks")


def find_task_info(logs_dir: Path) -> Dict[str, Tuple[Path, int, bool]]:
    """Find all tasks in a logs directory and return their info (ATIF format).

    Returns:
        Dict mapping task_id -> (task_dir_path, episode_count, is_resolved)
    """
    task_info = {}

    for task_id in os.listdir(logs_dir):
        task_dir = logs_dir / task_id
        if not task_dir.is_dir():
            continue

        results_file = task_dir / "results.json"
        if not results_file.exists():
            continue

        try:
            with open(results_file, "r") as f:
                results = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            continue

        is_resolved = results.get("is_resolved", False)
        actual_task_id = results.get("task_id", task_id)

        # Count episodes from trajectory.json
        episode_count = 0
        trajectory_file = task_dir / "trajectory.json"
        if trajectory_file.exists():
            try:
                with open(trajectory_file, "r") as f:
                    trajectory = json.load(f)
                episode_count = sum(
                    1 for step in trajectory 
                    if step.get("role") == "assistant"
                )
            except (json.JSONDecodeError, FileNotFoundError):
                pass

        task_info[actual_task_id] = (task_dir, episode_count, is_resolved)

    return task_info


def collect_traces(logs_dirs: List[Path], output_dir: Path, min_episodes: int = 10):
    """Collect traces from multiple logs directories, keeping only unresolved tasks
    and selecting the version with the most episodes for each task.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_task_info = defaultdict(list)

    for logs_dir in logs_dirs:
        print(f"Scanning {logs_dir}...")
        task_info = find_task_info(logs_dir)

        for task_id, (task_dir, episode_count, is_resolved) in task_info.items():
            all_task_info[task_id].append(
                (task_dir, episode_count, is_resolved, logs_dir)
            )

    selected_tasks = {}
    total_tasks = 0

    for task_id, versions in all_task_info.items():
        unresolved_versions = [
            (task_dir, episode_count, is_resolved, source_dir)
            for task_dir, episode_count, is_resolved, source_dir in versions
            if not is_resolved
        ]

        if not unresolved_versions:
            continue

        best_version = max(unresolved_versions, key=lambda x: x[1])

        if best_version[1] < min_episodes:
            continue

        selected_tasks[task_id] = best_version
        total_tasks += 1

    print(
        f"Found {total_tasks} unresolved tasks to collect (with >= {min_episodes} episodes)"
    )

    for task_id, (
        source_task_dir,
        episode_count,
        _,
        source_logs_dir,
    ) in selected_tasks.items():
        task_output_dir = output_dir / task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)

        target_name = source_task_dir.name
        dest_path = task_output_dir / target_name

        if dest_path.exists():
            shutil.rmtree(dest_path)

        shutil.copytree(source_task_dir, dest_path)

        print(f"{task_id}: {episode_count} episodes (from {source_logs_dir})")

    print(f"\nTotal tasks collected: {total_tasks}")
    print(f"Output directory: {output_dir}")


def run_command(cmd: List[str], env: dict = None, cwd: str = None):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    if env is None:
        env = os.environ.copy()

    result = subprocess.run(cmd, env=env, cwd=cwd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        return result
    return result


def run_replay_agent_tb(
    trajectory_folder: str,
    model_name: str,
    task_ids: List[str],
    run_id: str | None = None,
    dataset_name: str = "terminal-bench",
    dataset_version: str = "2.0",
    agent_import_path: str = "recovery_bench.replay_agent:ReplayAgent",
    n_concurrent: int = 1,
    global_timeout_multiplier: float = 2.0,
    additional_args: List[str] | None = None,
    cleanup_container: bool = False,
):
    """Run the replay agent for multiple task IDs using harbor run command."""

    if cleanup_container:
        cleanup_docker()

    env = os.environ.copy()
    env["TRAJECTORY_FOLDER"] = trajectory_folder

    cmd = [
        "harbor",
        "run",
        "--dataset",
        f"{dataset_name}@{dataset_version}",
        "--agent-import-path",
        agent_import_path,
        "--model",
        model_name,
        "--n-concurrent",
        str(n_concurrent),
    ]

    if run_id:
        cmd.extend(["--job-name", run_id])

    for task_id in task_ids:
        cmd.extend(["--task-name", task_id])

    if additional_args:
        cmd.extend(additional_args)

    print(f"Running command: {' '.join(cmd)}")
    print(f"Environment TRAJECTORY_FOLDER: {trajectory_folder}")
    print(f"Task IDs: {task_ids}")

    try:
        result = subprocess.run(cmd, env=env, check=True)
        print(f"Command completed successfully with return code: {result.returncode}")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code: {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"Error running command: {e}")
        return 1
