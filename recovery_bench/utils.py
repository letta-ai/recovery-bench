import hashlib
import subprocess
import json
import os
import shutil
from pathlib import Path
from typing import List


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
    """Get list of unsolved task IDs from a logs directory.
    
    Only supports ATIF format (terminus-2 style trajectory.json).
    For LettaCode, use the separate letta-specific utilities.
    """
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
        
        # Fallback: count from trajectory.json (ATIF format)
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


def is_hash_prefixed_directory(task_name: str) -> bool:
    """Check if directory name starts with an 8-char hex hash prefix."""
    if len(task_name) > 9 and task_name[8] == "-":
        hash_part = task_name[:8]
        if all(c in "0123456789abcdef" for c in hash_part.lower()):
            return True
    return False


def extract_instruction_from_trajectory(task_dir: Path) -> str | None:
    """Extract instruction from task directory.
    
    Supports:
    - ATIF trajectory.json (terminus-2)
    - LettaCode run_script.sh
    """
    agent_dir = task_dir / "agent"
    
    # Try LettaCode run_script.sh
    if agent_dir.exists():
        run_script = agent_dir / "run_script.sh"
        if run_script.exists():
            instruction = _extract_instruction_from_letta_script(run_script)
            if instruction:
                return instruction
    
    # Try ATIF trajectory.json
    trajectory_file = agent_dir / "trajectory.json" if agent_dir.exists() else None
    if not trajectory_file or not trajectory_file.exists():
        trajectory_file = task_dir / "trajectory.json"
    if trajectory_file and trajectory_file.exists():
        return _extract_instruction_from_atif(trajectory_file)
    
    return None


def _extract_instruction_from_letta_script(run_script: Path) -> str | None:
    """Extract instruction from LettaCode run_script.sh."""
    import re
    try:
        content = run_script.read_text()
        # Match -p followed by quoted string
        match = re.search(r"-p\s+'((?:[^'\\]|\\.)*)'", content)
        if match:
            return match.group(1).replace("'\"'\"'", "'")
        match = re.search(r'-p\s+"((?:[^"\\]|\\.)*)"', content)
        if match:
            return match.group(1).replace('\\"', '"').replace("\\n", "\n")
    except Exception:
        pass
    return None


def _extract_instruction_from_atif(trajectory_file: Path) -> str | None:
    """Extract instruction from ATIF trajectory.json."""
    try:
        with open(trajectory_file, "r") as f:
            trajectory = json.load(f)
        
        full_message = None
        
        # ATIF v1.5 format: steps array with source field
        if "steps" in trajectory:
            for step in trajectory["steps"]:
                if step.get("source") == "user":
                    full_message = step.get("message", "")
                    break
        
        # Fallback: old format with role field
        if not full_message:
            for step in trajectory:
                if step.get("role") == "user":
                    full_message = step.get("content", "")
                    break
        
        if not full_message:
            return None
        
        # Strip terminus-2 system prompt - task description comes after "Task Description:\n"
        task_marker = "Task Description:\n"
        if task_marker in full_message:
            task_part = full_message.split(task_marker, 1)[1]
            # Also strip the "Current terminal state:" suffix if present
            terminal_marker = "\n\nCurrent terminal state:"
            if terminal_marker in task_part:
                task_part = task_part.split(terminal_marker, 1)[0]
            return task_part
        
        return full_message
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
        
        if is_hash_prefixed_directory(task_name):
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


def run_recovery(
    traces_folder: str,
    model: str,
    task_ids: List[str],
    job_name: str | None = None,
    agent: str = "recovery_bench.replay_terminus:ReplayTerminus",
    n_concurrent: int = 4,
):
    """Run recovery agent on initial traces using harbor."""
    env = os.environ.copy()
    env["TRAJECTORY_FOLDER"] = traces_folder

    cmd = [
        "harbor",
        "run",
        "--dataset", "terminal-bench@2.0",
        "--agent-import-path", agent,
        "--model", model,
        "--n-concurrent", str(n_concurrent),
    ]

    if job_name:
        cmd.extend(["--job-name", job_name])

    for task_id in task_ids:
        cmd.extend(["--task-name", task_id])

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, env=env, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        return e.returncode
    except Exception as e:
        print(f"Error: {e}")
        return 1
