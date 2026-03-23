import hashlib
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import List

from harbor.models.agent.context import AgentContext
from litellm import ModelResponse, Usage, completion_cost
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    PromptTokensDetailsWrapper,
)

logger = logging.getLogger(__name__)


def resolve_config(model_str: str) -> tuple[str, dict]:
    """Resolve a model string to (model_name, agent_kwargs).

    Accepts either a plain model name (e.g. 'anthropic/claude-opus-4-5') or a
    path to a JSON config file (e.g. 'configs/letta-code/sonnet-46-max.json').

    When a JSON config is provided, the ``model`` key is extracted and
    everything else (``model_kwargs``, ``letta_code_model``, etc.) is returned
    as ``agent_kwargs`` — an opaque dict passed through to the agent via
    ``--agent-kwarg`` flags.

    Returns:
        model_name: LiteLLM model identifier (e.g. 'anthropic/claude-sonnet-4-6').
        agent_kwargs: All remaining config entries to forward to the agent.
    """
    path = Path(model_str)
    if path.suffix == ".json" and path.exists():
        with open(path) as f:
            config = json.load(f)
        model_name = config.pop("model", None)
        if not model_name:
            raise ValueError(f"Config {model_str} missing 'model' key")
        return model_name, config
    return model_str, {}


def shorten_model_name(model: str) -> str:
    """Extract short model name from a full model string (e.g. 'anthropic/claude-opus-4-5' -> 'claude-opus-4-5')."""
    return model.split("/")[-1]


def create_task_hash(task_description: str) -> str:
    """Create 8-character hash from task description."""
    return hashlib.sha256(task_description.encode("utf-8")).hexdigest()[:8]



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
    """Extract instruction from an ATIF trajectory.json in a task directory."""
    agent_dir = task_dir / "agent"
    trajectory_file = agent_dir / "trajectory.json" if agent_dir.exists() else None
    if not trajectory_file or not trajectory_file.exists():
        trajectory_file = task_dir / "trajectory.json"
    if not trajectory_file or not trajectory_file.exists():
        return None

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


def get_task_name(logs_dir: str | Path | None) -> str | None:
    """Extract task name from a logs_dir path.

    logs_dir is like: .../job-name/task-name__suffix/agent
    Returns the task name with the ``__suffix`` stripped.
    """
    if not logs_dir:
        return None
    task_dir_name = Path(logs_dir).parent.name
    if "__" in task_dir_name:
        return task_dir_name.rsplit("__", 1)[0]
    return task_dir_name


def find_trajectory_folder(logs_dir: str | Path | None, base_folder: str | Path) -> Path | None:
    """Find the trajectory folder for a task by matching task name against hash-prefixed dirs.

    Args:
        logs_dir: Agent logs directory (used to derive the task name).
        base_folder: Root directory containing trajectory folders.

    Returns:
        Path to the matching trajectory folder, or None if not found.
    """
    task_name = get_task_name(logs_dir)
    if not task_name:
        logger.warning("Could not extract task name from logs_dir")
        return None

    base_path = Path(base_folder)
    if not base_path.exists():
        logger.warning(f"Trajectory folder not found: {base_path}")
        return None

    for item in base_path.iterdir():
        if not item.is_dir():
            continue
        # Dir format: <hash>-<task-name>__<suffix>
        dir_name = item.name
        # Strip hash prefix (8 hex chars + dash)
        if len(dir_name) > 9 and dir_name[8] == "-":
            dir_task = dir_name[9:]
        else:
            dir_task = dir_name
        # Strip __suffix
        if "__" in dir_task:
            dir_task = dir_task.rsplit("__", 1)[0]

        if dir_task == task_name:
            for traj_path in [item / "agent" / "trajectory.json", item / "trajectory.json"]:
                if traj_path.exists():
                    logger.debug(f"Found trajectory for task {task_name}: {item}")
                    return item

    logger.warning(f"No trajectory found for task {task_name} in {base_path}")
    return None


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


def calculate_cost(model_name: str, usage: dict) -> float:
    """Calculate cost in USD using litellm's pricing data.

    Accounts for cached tokens and reasoning tokens via
    ``litellm.completion_cost`` with a synthetic ``ModelResponse``.

    Args:
        model_name: LiteLLM model identifier (e.g. ``anthropic/claude-sonnet-4-5``).
        usage: Dict with token counts (prompt_tokens, completion_tokens, and
            optionally cached_input_tokens, cache_write_tokens, reasoning_tokens).

    Returns:
        Cost in USD, or 0.0 if pricing is unavailable.
    """
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    if not model_name or (prompt_tokens == 0 and completion_tokens == 0):
        return 0.0

    resp = ModelResponse()
    resp.usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=usage.get("cached_input_tokens", 0),
            cache_creation_tokens=usage.get("cache_write_tokens", 0),
        ),
        completion_tokens_details=CompletionTokensDetailsWrapper(
            reasoning_tokens=usage.get("reasoning_tokens", 0),
        ),
    )
    try:
        return float(completion_cost(completion_response=resp, model=model_name))
    except Exception:
        # Model not in litellm's pricing DB
        logger.debug(f"Could not calculate cost for model {model_name}", exc_info=True)
        return 0.0


def save_usage(
    logs_dir: Path | None,
    context: AgentContext,
    extra_fields: dict | None = None,
) -> None:
    """Save usage stats to a separate JSON file in the task dir.

    Args:
        logs_dir: Agent logs directory (usage.json is written to its parent).
        context: Harbor AgentContext with token counts and cost.
        extra_fields: Optional dict of additional fields (e.g. cached_input_tokens,
            cache_write_tokens, reasoning_tokens) to include in usage.json.
    """
    usage = {
        "prompt_tokens": context.n_input_tokens or 0,
        "completion_tokens": context.n_output_tokens or 0,
        "total_tokens": (context.n_input_tokens or 0) + (context.n_output_tokens or 0),
        "cost_usd": round(context.cost_usd or 0, 6),
    }
    if extra_fields:
        usage.update(extra_fields)

    if logs_dir:
        usage_path = Path(logs_dir).parent / "usage.json"
    else:
        usage_path = Path("usage.json")
    try:
        usage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(usage_path, "w") as f:
            json.dump(usage, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save usage: {e}")


_USAGE_AGGREGATE_KEYS = [
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cost_usd",
    "cached_input_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
]


def aggregate_usage(job_dir: str) -> dict:
    """Aggregate usage stats from per-task usage.json files across a job directory."""
    job_path = Path(job_dir)
    totals: dict = {k: 0 if k != "cost_usd" else 0.0 for k in _USAGE_AGGREGATE_KEYS}
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
            for key in _USAGE_AGGREGATE_KEYS:
                totals[key] += usage.get(key, 0)
            task_count += 1
        except (json.JSONDecodeError, KeyError):
            continue

    totals["cost_usd"] = round(totals["cost_usd"], 6)
    totals["tasks_with_usage"] = task_count

    # Drop zero-valued optional fields to keep output clean
    for key in ("cached_input_tokens", "cache_write_tokens", "reasoning_tokens"):
        if totals.get(key, 0) == 0:
            totals.pop(key, None)

    # Save to job directory
    usage_file = job_path / "usage.json"
    try:
        with open(usage_file, "w") as f:
            json.dump(totals, f, indent=2)
        logger.info(f"Usage saved to {usage_file}")
    except Exception as e:
        logger.error(f"Failed to save usage: {e}")

    return totals
