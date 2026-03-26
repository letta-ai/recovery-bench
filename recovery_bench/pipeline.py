"""Pipeline functions for generating initial and recovery traces.

This module contains the harbor execution wrappers and orchestration logic.
Functions are ordered low-level → high-level:

- generate_initial_traces: Run harbor for initial traces
- generate_recovery_traces: Run harbor for recovery traces
- run_recovery: Single recovery pass (reorganize → unsolved → recover → aggregate)
- run_pipeline: Full orchestrator (initial → reorganize → recovery)
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from .agents import AGENT_REGISTRY
from .prompts import build_recovery_instruction, format_messages_as_text
from .replay import extract_messages
from .utils import (
    aggregate_usage,
    find_trajectory_by_name,
    get_agent_name,
    get_unsolved_tasks,
    reorganize_directories,
    resolve_config,
    run_command,
    shorten_model_name,
)

logger = logging.getLogger(__name__)

# Modal's exec limit for CLI arguments (bytes).
MAX_INSTRUCTION_BYTES = 65_536

# Tasks excluded from recovery for non-size reasons.
ALWAYS_EXCLUDED_TASKS = {
    "qemu-alpine-ssh",  # setup timeout during trajectory replay (every agent)
}

# Import path for the generic recovery wrapper
_RECOVERY_INSTALLED_AGENT = "recovery_bench.agents.base:RecoveryInstalledAgent"


def _estimate_recovery_size(task_name: str, traces_folder: str) -> int:
    """Estimate the byte length of the recovery instruction for a task.

    Loads the previous trajectory messages, formats them as full text,
    wraps them in the recovery instruction template, and returns the
    encoded byte length.

    Returns 0 if the trajectory cannot be found (task won't be excluded).
    """
    folder = find_trajectory_by_name(task_name, traces_folder)
    if folder is None:
        return 0
    messages = extract_messages(folder)
    if not messages:
        return 0
    message_context = format_messages_as_text(messages)
    # Use a placeholder instruction — the original task text is small relative
    # to the message context, so this gives a close-enough estimate.
    instruction = build_recovery_instruction("(task instruction)", message_context)
    return len(instruction.encode("utf-8"))


def filter_oversized_tasks(
    task_ids: list[str],
    traces_folder: str,
    max_bytes: int = MAX_INSTRUCTION_BYTES,
) -> list[str]:
    """Filter out tasks whose recovery instruction would exceed *max_bytes*.

    Args:
        task_ids: Candidate task IDs.
        traces_folder: Path to initial traces (used to look up trajectories).
        max_bytes: Maximum allowed instruction size in bytes.

    Returns:
        Filtered list of task IDs that fit within the limit.
    """
    kept: list[str] = []
    for task_id in task_ids:
        size = _estimate_recovery_size(task_id, traces_folder)
        if size > max_bytes:
            logger.warning(
                f"Excluding {task_id}: recovery instruction is {size:,} bytes "
                f"(limit {max_bytes:,})"
            )
        else:
            kept.append(task_id)
    return kept


def resolve_agent(agent_str: str) -> tuple[str, bool, dict[str, str]]:
    """Resolve an agent string to (agent_ref, is_import_path, extra_agent_kwargs).

    Handles four formats:
    1. Import path (contains ':'): pass through as --agent-import-path
    2. Registry name (in AGENT_REGISTRY): look up import path
    3. 'installed:<name>': route to RecoveryInstalledAgent with wrapped_agent kwarg
    4. Plain Harbor agent name (e.g. 'terminus-2'): pass through as --agent

    Returns:
        (agent_ref, is_import_path, extra_kwargs)
        - agent_ref: import path or harbor agent name
        - is_import_path: True if agent_ref should be passed via --agent-import-path,
          False if it should be passed via --agent
        - extra_kwargs: additional --agent-kwarg flags to pass to harbor run
    """
    # Case 3: installed:<name>
    if agent_str.startswith("installed:"):
        harbor_agent_name = agent_str.split(":", 1)[1]
        return _RECOVERY_INSTALLED_AGENT, True, {"wrapped_agent": harbor_agent_name}

    # Case 2: registry name
    if agent_str in AGENT_REGISTRY:
        return AGENT_REGISTRY[agent_str], True, {}

    # Case 1: import path (contains ':')
    if ":" in agent_str:
        return agent_str, True, {}

    # Case 4: plain Harbor agent name (e.g. 'terminus-2')
    return agent_str, False, {}


def _build_harbor_cmd(
    model: str,
    n_concurrent: int,
    dataset_version: str = "2.0",
    agent: str | None = None,
    job_name: str | None = None,
    task_ids: list[str] | None = None,
    agent_kwargs: dict | None = None,
    harbor_env: str | None = None,
    message_mode: str | None = None,
) -> list[str]:
    """Build the ``harbor run`` command list.

    Handles agent resolution, dataset/model/concurrency flags, optional
    job-name, env, task-ids, and agent-kwargs.
    """
    cmd = ["harbor", "run", "--dataset", f"terminal-bench@{dataset_version}"]

    # Resolve agent → import path or Harbor name
    extra_kwargs: dict[str, str] = {}
    if agent:
        agent_ref, is_import_path, extra_kwargs = resolve_agent(agent)
        if is_import_path:
            cmd.extend(["--agent-import-path", agent_ref])
        else:
            cmd.extend(["--agent", agent_ref])
    else:
        cmd.extend(["--agent", "terminus-2"])

    cmd.extend(["--model", model, "--n-concurrent", str(n_concurrent)])

    if job_name:
        cmd.extend(["--job-name", job_name])
    if harbor_env:
        cmd.extend(["--env", harbor_env])
    if task_ids:
        for task_id in task_ids:
            cmd.extend(["--task-name", task_id])

    # Append --agent-kwarg flags
    all_kwargs = {**(agent_kwargs or {}), **(extra_kwargs or {})}
    if message_mode:
        all_kwargs["message_mode"] = message_mode
    for key, value in all_kwargs.items():
        serialized = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
        cmd.extend(["--agent-kwarg", f"{key}={serialized}"])

    return cmd


def generate_initial_traces(
    model: str,
    job_name: str,
    dataset_version: str = "2.0",
    n_concurrent: int = 6,
    task_ids: list[str] | None = None,
    agent: str | None = None,
    agent_kwargs: dict | None = None,
    harbor_env: str | None = None,
) -> str:
    """Generate initial traces using harbor run.

    Args:
        model: Model name for the initial agent.
        job_name: Job name for output directory.
        dataset_version: Terminal-Bench dataset version.
        n_concurrent: Number of concurrent processes.
        task_ids: Specific task IDs to run. None = all tasks.
        agent: Agent name, registry name, or import path. None = terminus-2.
        agent_kwargs: Extra kwargs forwarded to the agent via --agent-kwarg.
        harbor_env: Harbor sandbox backend (e.g. docker, daytona, modal).

    Returns:
        Path to the output job directory (e.g. "jobs/<job_name>").

    Raises:
        RuntimeError: If harbor run fails.
    """
    logger.info(f"Generating initial traces for {model}...")

    cmd = _build_harbor_cmd(
        model=model,
        n_concurrent=n_concurrent,
        dataset_version=dataset_version,
        agent=agent,
        job_name=job_name,
        task_ids=task_ids,
        agent_kwargs=agent_kwargs,
        harbor_env=harbor_env,
    )

    result = run_command(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Initial trace generation failed with exit code {result.returncode}")
    return f"jobs/{job_name}"


def generate_recovery_traces(
    traces_folder: str,
    model: str,
    task_ids: list[str],
    job_name: str | None = None,
    agent: str = "recovery-terminus",
    n_concurrent: int = 4,
    agent_kwargs: dict | None = None,
    harbor_env: str | None = None,
    dataset_version: str = "2.0",
    message_mode: str | None = None,
) -> int:
    """Run recovery agent on initial traces using harbor.

    Args:
        traces_folder: Path to initial traces directory (set as TRAJECTORY_FOLDER).
        model: Model name for the recovery agent.
        task_ids: Task IDs to run recovery on.
        job_name: Job name for output directory.
        agent: Recovery agent name, import path, or installed:<name>.
        n_concurrent: Number of concurrent processes.
        agent_kwargs: Extra kwargs forwarded to the agent via --agent-kwarg.
        harbor_env: Harbor sandbox backend (e.g. docker, daytona, modal).
        dataset_version: Terminal-Bench dataset version.
        message_mode: Message mode for recovery agent (full/none/summary).

    Returns:
        Exit code from harbor run.
    """
    env = os.environ.copy()
    env["TRAJECTORY_FOLDER"] = traces_folder

    cmd = _build_harbor_cmd(
        model=model,
        n_concurrent=n_concurrent,
        dataset_version=dataset_version,
        agent=agent,
        job_name=job_name,
        task_ids=task_ids,
        agent_kwargs=agent_kwargs,
        harbor_env=harbor_env,
        message_mode=message_mode,
    )

    result = run_command(cmd, env=env)
    return result.returncode


def run_recovery(
    traces_folder: str,
    model: str,
    job_name: str,
    agent: str = "recovery-terminus",
    n_concurrent: int = 4,
    task_ids: list[str] | None = None,
    agent_kwargs: dict | None = None,
    harbor_env: str | None = None,
    dataset_version: str = "2.0",
    reorganize: bool = True,
    message_mode: str | None = None,
) -> tuple[str, int]:
    """Run a single recovery pass: reorganize → find unsolved → recover → aggregate.

    Args:
        traces_folder: Path to initial traces directory.
        model: Model name for the recovery agent.
        job_name: Job name for output directory.
        agent: Recovery agent name, import path, or installed:<name>.
        n_concurrent: Number of concurrent processes.
        task_ids: Specific task IDs to recover. None = auto-detect unsolved.
        agent_kwargs: Extra kwargs forwarded to the agent via --agent-kwarg.
        harbor_env: Harbor sandbox backend (e.g. docker, daytona, modal).
        dataset_version: Terminal-Bench dataset version.
        reorganize: Whether to reorganize directories with hash prefixes first.
        message_mode: Message mode for recovery agent (full/none/summary).

    Returns:
        (output_job_dir, exit_code). output_job_dir is "" if no tasks to recover.
    """
    if reorganize:
        reorganize_directories(traces_folder)

    if task_ids is None:
        task_ids = get_unsolved_tasks(traces_folder)

    task_ids = [t for t in task_ids if t not in ALWAYS_EXCLUDED_TASKS]
    task_ids = filter_oversized_tasks(task_ids, traces_folder)

    if not task_ids:
        logger.info("No unsolved tasks found, skipping recovery.")
        return "", 0

    logger.info(f"Running recovery on {len(task_ids)} task(s) with {model}")

    rc = generate_recovery_traces(
        traces_folder=traces_folder,
        model=model,
        task_ids=task_ids,
        job_name=job_name,
        agent=agent,
        n_concurrent=n_concurrent,
        agent_kwargs=agent_kwargs,
        harbor_env=harbor_env,
        dataset_version=dataset_version,
        message_mode=message_mode,
    )

    # Aggregate usage
    job_dir = Path("jobs") / job_name
    if job_dir.exists():
        usage = aggregate_usage(str(job_dir))
        if usage["tasks_with_usage"] > 0:
            logger.info(
                f"Total usage: {usage['total_tokens']} tokens "
                f"({usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion), "
                f"cost: ${usage['cost_usd']:.4f} "
                f"across {usage['tasks_with_usage']} task(s)"
            )

    return f"jobs/{job_name}", rc


def run_pipeline(
    initial_model: str | None = None,
    recovery_model: str | None = None,
    resume_initial: str | None = None,
    initial_agent: str | None = None,
    recovery_agent: str = "recovery-terminus",
    task_ids: list[str] | None = None,
    n_concurrent: int = 8,
    dataset_version: str = "2.0",
    job_name: str | None = None,
    harbor_env: str | None = None,
    message_mode: str | None = None,
) -> int:
    """Run the full trace generation pipeline.

    Generates initial traces (or resumes from existing), then runs
    recovery on unsolved tasks.

    Args:
        initial_model: Model name or JSON config path for initial traces.
            Required unless resume_initial is set.
        recovery_model: Model name or JSON config path for recovery.
            None = skip recovery.
        resume_initial: Path to existing initial traces (skips generation).
        initial_agent: Agent name or import path for initial traces.
        recovery_agent: Recovery agent name, import path, or installed:<name>.
        task_ids: Specific task IDs to run.
        n_concurrent: Number of concurrent processes.
        dataset_version: Terminal-Bench dataset version.
        job_name: Custom job name for recovery output.
        harbor_env: Harbor sandbox backend (e.g. docker, daytona, modal).
        message_mode: Message mode for recovery agent (full/none/summary).

    Returns:
        Exit code (0 = success).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Resolve model configs (raw string or JSON path → model name + agent kwargs)
    initial_model_name, initial_agent_kwargs = (
        resolve_config(initial_model) if initial_model else (None, {})
    )
    recovery_model_name, recovery_agent_kwargs = (
        resolve_config(recovery_model) if recovery_model else (None, {})
    )

    # Step 1: Generate initial traces or resume from existing
    if resume_initial:
        if not Path(resume_initial).exists():
            logger.error(f"Traces directory does not exist: {resume_initial}")
            return 1
        initial_traces_dir = resume_initial
        logger.info(f"Resuming from existing initial trajectories at {initial_traces_dir}")
    else:
        initial_agent_name = get_agent_name(initial_agent) if initial_agent else "terminus-2"
        model_short = shorten_model_name(initial_model_name)
        initial_run_id = f"initial-{initial_agent_name}-{model_short}-{timestamp}"
        initial_traces_dir = generate_initial_traces(
            model=initial_model_name,
            job_name=initial_run_id,
            dataset_version=dataset_version,
            n_concurrent=n_concurrent,
            task_ids=task_ids,
            agent=initial_agent,
            agent_kwargs=initial_agent_kwargs,
            harbor_env=harbor_env,
        )

    # Aggregate usage for initial traces
    initial_job_dir = Path(initial_traces_dir)
    if initial_job_dir.exists():
        usage = aggregate_usage(str(initial_job_dir))
        if usage["tasks_with_usage"] > 0:
            logger.info(
                f"Initial usage: {usage['total_tokens']} tokens "
                f"({usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion), "
                f"cost: ${usage['cost_usd']:.4f} "
                f"across {usage['tasks_with_usage']} task(s)"
            )

    # Stop here if no recovery model specified
    if not recovery_model_name:
        logger.info(f"Initial traces complete: {initial_traces_dir}")
        return 0

    # Step 2: Reorganize initial traces with hash prefixes
    reorganize_directories(initial_traces_dir)

    # Step 3: Run recovery on unsolved tasks
    recovery_agent_name = get_agent_name(recovery_agent)
    recovery_model_short = shorten_model_name(recovery_model_name)
    recovery_job_name = (
        job_name or f"{recovery_agent_name}-{message_mode}-{recovery_model_short}-{timestamp}"
    )

    recovery_traces_dir, rc = run_recovery(
        traces_folder=initial_traces_dir,
        model=recovery_model_name,
        job_name=recovery_job_name,
        agent=recovery_agent,
        n_concurrent=n_concurrent,
        task_ids=task_ids if resume_initial else None,
        agent_kwargs=recovery_agent_kwargs,
        harbor_env=harbor_env,
        dataset_version=dataset_version,
        reorganize=False,  # Already reorganized above
        message_mode=message_mode,
    )

    if rc != 0:
        logger.error(f"Recovery failed with exit code {rc}")
        return rc

    logger.info("--- Pipeline completed successfully! ---")
    logger.info(f"Initial traces: {initial_traces_dir}")
    if recovery_traces_dir:
        logger.info(f"Recovery traces: {recovery_traces_dir}")
    return 0
