"""Pipeline functions for generating initial and recovery traces.

This module contains the harbor execution wrappers and orchestration logic.
Functions are ordered low-level → high-level:

- generate_initial_traces: Run harbor for initial traces
- generate_recovery_traces: Run harbor for recovery traces
- run_recovery: Single recovery pass (reorganize → unsolved → recover → aggregate)
- run_pipeline: Full orchestrator (initial → reorganize → iterative recovery)
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from .agents import AGENT_REGISTRY
from .utils import (
    aggregate_usage,
    cleanup_docker,
    get_agent_name,
    get_unsolved_tasks,
    reorganize_directories,
    run_command,
    shorten_model_name,
)

logger = logging.getLogger(__name__)

# Import path for the generic recovery wrapper
_RECOVERY_INSTALLED_AGENT = "recovery_bench.agents.base:RecoveryInstalledAgent"


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
    model_kwargs: dict | None = None,
    letta_code_model: str | None = None,
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
    if model_kwargs:
        cmd.extend(["--agent-kwarg", f"model_kwargs={json.dumps(model_kwargs)}"])
    if letta_code_model:
        cmd.extend(["--agent-kwarg", f"letta_code_model={letta_code_model}"])
    if message_mode:
        cmd.extend(["--agent-kwarg", f"message_mode={message_mode}"])
    if extra_kwargs:
        for key, value in extra_kwargs.items():
            cmd.extend(["--agent-kwarg", f"{key}={value}"])

    return cmd


def generate_initial_traces(
    model: str,
    job_name: str,
    dataset_version: str = "2.0",
    n_concurrent: int = 6,
    task_ids: list[str] | None = None,
    agent: str | None = None,
    model_kwargs: dict | None = None,
    letta_code_model: str | None = None,
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
        model_kwargs: Extra model kwargs (e.g. reasoning effort).
        letta_code_model: Letta Code model id (e.g. 'sonnet-4.6-xhigh').
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
        model_kwargs=model_kwargs,
        letta_code_model=letta_code_model,
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
    model_kwargs: dict | None = None,
    letta_code_model: str | None = None,
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
        model_kwargs: Extra model kwargs (e.g. reasoning effort).
        letta_code_model: Letta Code model id (e.g. 'sonnet-4.6-xhigh').
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
        model_kwargs=model_kwargs,
        letta_code_model=letta_code_model,
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
    model_kwargs: dict | None = None,
    letta_code_model: str | None = None,
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
        model_kwargs: Extra model kwargs (e.g. reasoning effort).
        letta_code_model: Letta Code model id (e.g. 'sonnet-4.6-xhigh').
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
        model_kwargs=model_kwargs,
        letta_code_model=letta_code_model,
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
    initial_model_kwargs: dict | None = None,
    initial_letta_code_model: str | None = None,
    recovery_model: str | None = None,
    recovery_model_kwargs: dict | None = None,
    recovery_letta_code_model: str | None = None,
    resume_initial: str | None = None,
    initial_agent: str | None = None,
    recovery_agent: str = "recovery-terminus",
    task_ids: list[str] | None = None,
    n_concurrent: int = 8,
    max_iterations: int = 1,
    dataset_version: str = "2.0",
    job_name: str | None = None,
    cleanup_container: bool = False,
    harbor_env: str | None = None,
    message_mode: str | None = None,
) -> int:
    """Run the full trace generation pipeline.

    Generates initial traces (or resumes from existing), then iteratively
    runs recovery on unsolved tasks.

    Args:
        initial_model: Model name for initial traces. Required unless resume_initial is set.
        initial_model_kwargs: Extra model kwargs for initial traces.
        initial_letta_code_model: Letta Code model id for initial traces.
        recovery_model: Model name for recovery. None = skip recovery.
        recovery_model_kwargs: Extra model kwargs for recovery.
        recovery_letta_code_model: Letta Code model id for recovery.
        resume_initial: Path to existing initial traces (skips generation).
        initial_agent: Agent name or import path for initial traces.
        recovery_agent: Recovery agent name, import path, or installed:<name>.
        task_ids: Specific task IDs to run.
        n_concurrent: Number of concurrent processes.
        max_iterations: Maximum number of recovery iterations.
        dataset_version: Terminal-Bench dataset version.
        job_name: Custom job name for recovery output.
        cleanup_container: Whether to cleanup Docker before running.
        harbor_env: Harbor sandbox backend (e.g. docker, daytona, modal).
        message_mode: Message mode for recovery agent (full/none/summary).

    Returns:
        Exit code (0 = success).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if cleanup_container:
        cleanup_docker()

    # Step 1: Generate initial traces or resume from existing
    if resume_initial:
        if not Path(resume_initial).exists():
            logger.error(f"Traces directory does not exist: {resume_initial}")
            return 1
        initial_traces_dir = resume_initial
        logger.info(f"Resuming from existing initial trajectories at {initial_traces_dir}")
    else:
        initial_agent_name = get_agent_name(initial_agent) if initial_agent else "terminus-2"
        model_short = shorten_model_name(initial_model)
        initial_run_id = f"initial-{initial_agent_name}-{model_short}-{timestamp}"
        initial_traces_dir = generate_initial_traces(
            model=initial_model,
            job_name=initial_run_id,
            dataset_version=dataset_version,
            n_concurrent=n_concurrent,
            task_ids=task_ids,
            agent=initial_agent,
            model_kwargs=initial_model_kwargs,
            letta_code_model=initial_letta_code_model,
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
    if not recovery_model:
        logger.info(f"Initial traces complete: {initial_traces_dir}")
        return 0

    # Step 2: Reorganize initial traces with hash prefixes
    reorganize_directories(initial_traces_dir)

    # Step 3: Iteratively run recovery on unsolved tasks
    all_trace_dirs = [initial_traces_dir]
    current_traces_dir = initial_traces_dir

    for iteration in range(1, max_iterations + 1):
        logger.info(f"--- Starting iteration {iteration} ---")

        # Build job name
        if job_name and max_iterations == 1:
            recovery_job_name = job_name
        else:
            recovery_agent_name = get_agent_name(recovery_agent)
            recovery_model_short = shorten_model_name(recovery_model)
            recovery_job_name = (
                job_name or f"{recovery_agent_name}-{recovery_model_short}-{timestamp}"
            )
            if max_iterations > 1:
                recovery_job_name = f"{recovery_job_name}-iter{iteration}"

        recovery_traces_dir, rc = run_recovery(
            traces_folder=current_traces_dir,
            model=recovery_model,
            job_name=recovery_job_name,
            agent=recovery_agent,
            n_concurrent=n_concurrent,
            task_ids=task_ids if resume_initial else None,
            model_kwargs=recovery_model_kwargs,
            letta_code_model=recovery_letta_code_model,
            harbor_env=harbor_env,
            dataset_version=dataset_version,
            reorganize=False,  # Already reorganized above (or by previous iteration)
            message_mode=message_mode,
        )

        if rc != 0:
            logger.error(f"Recovery iteration {iteration} failed with exit code {rc}")
            return rc

        if not recovery_traces_dir:
            logger.info(f"No unsolved tasks found in iteration {iteration}, stopping.")
            break

        # Hash reorganize the new traces for next iteration
        reorganize_directories(recovery_traces_dir)

        all_trace_dirs.append(recovery_traces_dir)
        current_traces_dir = recovery_traces_dir

    logger.info("--- Pipeline completed successfully! ---")
    logger.info(f"Initial traces: {initial_traces_dir}")
    logger.info(f"All trace directories: {all_trace_dirs}")
    return 0
