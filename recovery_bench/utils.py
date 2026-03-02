import hashlib
import logging
import subprocess
import json
import os
import shutil
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def resolve_model(model_str: str) -> tuple[str, dict]:
    """Resolve a model string to (model_name, model_kwargs).

    Accepts either a model name (e.g. 'anthropic/claude-opus-4-5') or a path
    to a JSON config file (e.g. 'configs/models/opus-4.6-high.json').
    """
    path = Path(model_str)
    if path.suffix == ".json" and path.exists():
        with open(path) as f:
            config = json.load(f)
        model_name = config.get("model")
        if not model_name:
            raise ValueError(f"Model config {model_str} missing 'model' key")
        return model_name, config.get("model_kwargs", {})
    return model_str, {}


def shorten_model_name(model: str) -> str:
    """Extract short model name from a full model string (e.g. 'anthropic/claude-opus-4-5' -> 'claude-opus-4-5')."""
    return model.split("/")[-1]


def create_task_hash(task_description: str) -> str:
    """Create 8-character hash from task description."""
    return hashlib.sha256(task_description.encode("utf-8")).hexdigest()[:8]


def cleanup_docker():
    """Clean up Docker containers and system resources."""
    logger.info("Cleaning up Docker containers and system resources...")

    # Remove all containers (running and stopped)
    try:
        result = subprocess.run(
            "docker rm $(docker ps -aq) -f", shell=True, capture_output=True, text=True
        )
        if result.returncode == 0:
            logger.info("Successfully removed Docker containers")
        else:
            logger.warning(f"Docker rm command output: {result.stderr}")
    except Exception as e:
        logger.error(f"Error removing Docker containers: {e}")

    # Clean up Docker system
    try:
        result = subprocess.run(
            ["docker", "system", "prune", "-f"], capture_output=True, text=True
        )
        if result.returncode == 0:
            logger.info("Successfully cleaned up Docker system")
        else:
            logger.warning(f"Docker system prune failed: {result.stderr}")
    except Exception as e:
        logger.error(f"Error cleaning up Docker system: {e}")


def get_unsolved_tasks(logs_dir: str, print_output: bool = False) -> List[str]:
    """Get list of unsolved task IDs from a logs directory."""
    logs_dir = Path(logs_dir)
    unsolved_ids = []

    for task_id in os.listdir(logs_dir):
        task_dir = logs_dir / task_id
        if not task_dir.is_dir():
            continue

        results_file = task_dir / "result.json"
        if not results_file.exists():
            continue
            
        try:
            with open(results_file, "r") as f:
                results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            continue

        # reward > 0 means resolved
        verifier_result = results.get("verifier_result") or {}
        rewards = verifier_result.get("rewards") or {}
        reward = rewards.get("reward", 0.0)
        if reward > 0:
            if print_output:
                logger.debug(f"Skipping {task_id}: resolved (reward={reward})")
            continue
        
        task_name = results.get("task_name", task_id)
        unsolved_ids.append(task_name)

    if print_output:
        logger.info(f"Found {len(unsolved_ids)} unsolved tasks")
        for task_id in unsolved_ids:
            logger.debug(f"  {task_id}")

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
    logger.info(f"Reorganizing {base_path}")

    base_path = Path(base_path)
    if not base_path.exists():
        logger.warning("Path does not exist")
        return

    task_to_hash = {}

    for task_dir in base_path.iterdir():
        if not task_dir.is_dir():
            continue
        
        task_name = task_dir.name
        
        if is_hash_prefixed_directory(task_name):
            logger.debug(f"  {task_name} -> SKIPPED (already has hash prefix)")
            continue

        instruction = extract_instruction_from_trajectory(task_dir)
        if instruction:
            task_hash = create_task_hash(instruction)
            task_to_hash[task_dir] = task_hash
            logger.debug(f"  {task_name} -> {task_hash}")
        else:
            logger.debug(f"  {task_name} -> SKIPPED (no trajectory.json or instruction)")

    for task_dir, task_hash in task_to_hash.items():
        task_name = task_dir.name
        new_task_dir = task_dir.parent / f"{task_hash}-{task_name}"

        try:
            shutil.move(str(task_dir), str(new_task_dir))
            logger.debug(f"    Renamed to {task_hash}-{task_name}")
        except Exception as e:
            logger.error(f"    Error renaming {task_dir}: {e}")

    logger.info(f"Processed {len(task_to_hash)} tasks")


def get_agent_name(agent_import_path: str) -> str:
    """Get agent name from an import path like 'module.path:ClassName'.

    Dynamically imports the class and calls its name() static method.
    Falls back to converting the class name to kebab-case.
    """
    if ":" not in agent_import_path:
        # Built-in agent name (e.g., "terminus-2"), use as-is
        return agent_import_path

    module_path, class_name = agent_import_path.rsplit(":", 1)
    try:
        import importlib

        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls.name()
    except Exception:
        # Fallback: convert CamelCase to kebab-case
        import re

        return re.sub(r"(?<!^)(?=[A-Z])", "-", class_name).lower()


def run_command(cmd: List[str], env: dict = None, cwd: str = None):
    """Run a command and return the result."""
    logger.info(f"Running: {' '.join(cmd)}")
    if env is None:
        env = os.environ.copy()

    result = subprocess.run(cmd, env=env, cwd=cwd, capture_output=False, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed with return code {result.returncode}")
        return result
    return result


def aggregate_usage(job_dir: str) -> dict:
    """Aggregate usage stats from per-task usage.json files across a job directory."""
    job_path = Path(job_dir)
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}
    task_count = 0

    for task_dir in sorted(job_path.iterdir()):
        if not task_dir.is_dir():
            continue
        usage_file = task_dir / "usage.json"
        if not usage_file.exists():
            continue
        try:
            with open(usage_file) as f:
                usage = json.load(f)
            totals["prompt_tokens"] += usage.get("prompt_tokens", 0)
            totals["completion_tokens"] += usage.get("completion_tokens", 0)
            totals["total_tokens"] += usage.get("total_tokens", 0)
            totals["cost_usd"] += usage.get("cost_usd", 0.0)
            task_count += 1
        except (json.JSONDecodeError, KeyError):
            continue

    totals["cost_usd"] = round(totals["cost_usd"], 6)
    totals["tasks_with_usage"] = task_count

    # Save to job directory
    usage_file = job_path / "usage.json"
    try:
        with open(usage_file, "w") as f:
            json.dump(totals, f, indent=2)
        logger.info(f"Usage saved to {usage_file}")
    except Exception as e:
        logger.error(f"Failed to save usage: {e}")

    return totals
