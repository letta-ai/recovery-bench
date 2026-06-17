"""Shared trajectory replay engine.

Provides the common recovery logic used by all recovery agents:
- Trajectory discovery and ATIF parsing
- Command extraction (shell commands + raw keystrokes)
- Replay executors (environment.exec for installed agents, tmux for terminus)
- Chat message extraction (for context injection)
"""

import asyncio
import json
import logging
import shlex
from dataclasses import dataclass
from pathlib import Path

from harbor.environments.base import BaseEnvironment

from recovery_bench.utils import find_trajectory_folder

logger = logging.getLogger(__name__)


@dataclass
class ReplayCommand:
    """A command extracted from a previous trajectory for replay."""

    command: str  # Shell command (control sequences stripped)
    keystrokes: str  # Raw keystrokes (for tmux replay)
    timeout_sec: float = 15.0


@dataclass
class FailedTest:
    """A single failed test extracted from CTRF verifier output."""

    name: str
    trace: str  # Truncated assertion text / failure detail


@dataclass
class VerifierSignal:
    """Ground-truth failure signature from the failed initial attempt's verifier.

    Captures what the *verifier* observed (independent of what the failed
    agent claimed).  This is fed into the summarizer so it can write a
    factual "what went wrong" section instead of inheriting the agent's
    self-narration, which is often confidently wrong.
    """

    reward: float | None
    failed_tests: list[FailedTest]
    test_summary: dict | None  # CTRF top-level summary: passed/failed/skipped counts
    raw_tail: str | None  # Last ~2000 chars of test-stdout.txt (fallback when no CTRF)


# Per-failed-test trace cap (CTRF traces can be huge — e.g. full pytest
# diffs).  600 chars captures the assertion + the first few diff lines,
# which is usually all that's actionable.
_FAILED_TEST_TRACE_CAP = 600

# Tail of test-stdout.txt to keep when no CTRF is available.  The
# beginning is almost always apt-install / download-progress noise.
_STDOUT_TAIL_CHARS = 2000

# Total verifier-signal cap, applied after formatting (defense in depth
# in case CTRF itself is huge — e.g. regex-chess has a 14KB ctrf).
_TOTAL_SIGNAL_CAP = 10_000


# ---------------------------------------------------------------------------
# Trajectory parsing
# ---------------------------------------------------------------------------


def _find_trajectory_file(trajectory_folder: Path) -> Path | None:
    """Locate the trajectory.json inside a trajectory folder."""
    for path in [
        trajectory_folder / "agent" / "trajectory.json",
        trajectory_folder / "trajectory.json",
    ]:
        if path.exists():
            return path
    return None


def _load_trajectory(trajectory_file: Path) -> list[dict]:
    """Load and return the steps from a trajectory file."""
    try:
        with open(trajectory_file, "r") as f:
            trajectory = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        logger.error(f"Failed to read trajectory: {trajectory_file}")
        return []

    return trajectory.get("steps", trajectory) if isinstance(trajectory, dict) else trajectory


def _extract_from_step(step: dict) -> tuple[list[ReplayCommand], dict | None]:
    """Extract replay commands and a chat message from a single trajectory step.

    Returns:
        (commands, message) where message is a dict with role/content or None.
    """
    source = step.get("source", step.get("role", ""))
    content = step.get("message", step.get("content", ""))

    commands: list[ReplayCommand] = []
    message: dict | None = None

    if source in ("agent", "assistant"):
        # Extract commands from tool_calls (ATIF v1.5+)
        tool_calls = step.get("tool_calls", [])
        for tool_call in tool_calls:
            args = tool_call.get("arguments", {})
            keystrokes = args.get("keystrokes", "")
            if keystrokes:
                cmd = keystrokes.rstrip("\n").rstrip("\r")
                commands.append(
                    ReplayCommand(
                        command=cmd if (cmd and not cmd.startswith("C-")) else "",
                        keystrokes=keystrokes,
                        timeout_sec=int(args.get("duration", 1) * 10) + 5,
                    )
                )

        # Fallback: parse commands from message content (old ATIF format)
        if not tool_calls:
            try:
                response = json.loads(content) if isinstance(content, str) else content
                if isinstance(response, dict) and "commands" in response:
                    for cmd_obj in response["commands"]:
                        ks = cmd_obj.get("keystrokes", "")
                        if ks:
                            cmd = ks.rstrip("\n").rstrip("\r")
                            commands.append(
                                ReplayCommand(
                                    command=cmd if (cmd and not cmd.startswith("C-")) else "",
                                    keystrokes=ks,
                                    timeout_sec=cmd_obj.get("timeout_sec", 120),
                                )
                            )
            except (json.JSONDecodeError, TypeError):
                pass

    # Build chat message for context injection
    role = "assistant" if source == "agent" else source
    if role in ("user", "assistant", "system"):
        message = {"role": role, "content": content}

    return commands, message


def extract_commands(trajectory_folder: Path) -> list[ReplayCommand]:
    """Parse ATIF trajectory and extract replay commands."""
    trajectory_file = _find_trajectory_file(trajectory_folder)
    if not trajectory_file:
        return []

    steps = _load_trajectory(trajectory_file)
    commands: list[ReplayCommand] = []
    for step in steps:
        step_commands, _ = _extract_from_step(step)
        commands.extend(step_commands)

    logger.info(f"Extracted {len(commands)} commands from {trajectory_file}")
    return commands


def extract_messages(trajectory_folder: Path) -> list[dict]:
    """Extract chat messages from trajectory for context injection."""
    trajectory_file = _find_trajectory_file(trajectory_folder)
    if not trajectory_file:
        return []

    steps = _load_trajectory(trajectory_file)
    messages: list[dict] = []
    for step in steps:
        _, message = _extract_from_step(step)
        if message:
            messages.append(message)

    return messages


def extract_verifier_signal(trajectory_folder: Path) -> VerifierSignal | None:
    """Extract ground-truth verifier output from a failed trajectory folder.

    The failed initial-run folder always has a ``result.json`` and usually
    a ``verifier/`` subdirectory with structured test results (CTRF JSON)
    and raw stdout.  This is what the *verifier* saw — independent of
    the failed agent's self-narration — and is the most reliable signal
    of what went wrong.

    Loading priority:
    1. Failed tests + summary from ``verifier/ctrf.json`` if present
       (CTRF format: ``results.tests[]`` with name/status/trace).
    2. Tail of ``verifier/test-stdout.txt`` as a fallback when CTRF is
       absent (some verifiers don't emit CTRF).
    3. Reward from ``result.json`` ``verifier_result.rewards.reward``.

    Returns ``None`` when no verifier artifacts are available — callers
    should fall back to message-only summarization.
    """
    if trajectory_folder is None or not trajectory_folder.exists():
        return None

    # Reward from result.json (always present in completed trajectories).
    reward: float | None = None
    result_file = trajectory_folder / "result.json"
    if result_file.exists():
        try:
            data = json.loads(result_file.read_text())
            vr = data.get("verifier_result") or {}
            rewards = vr.get("rewards") or {}
            reward = rewards.get("reward")
        except (json.JSONDecodeError, OSError) as e:
            logger.debug(f"Failed to read {result_file}: {e}")

    verifier_dir = trajectory_folder / "verifier"
    failed_tests: list[FailedTest] = []
    test_summary: dict | None = None
    raw_tail: str | None = None

    # Prefer CTRF (structured) when available.
    ctrf_file = verifier_dir / "ctrf.json"
    if ctrf_file.exists():
        try:
            ctrf = json.loads(ctrf_file.read_text())
            results = ctrf.get("results") or {}
            test_summary = results.get("summary")
            for t in results.get("tests") or []:
                if t.get("status") != "failed":
                    continue
                trace = t.get("trace") or t.get("message") or ""
                if len(trace) > _FAILED_TEST_TRACE_CAP:
                    trace = trace[:_FAILED_TEST_TRACE_CAP] + "...[truncated]"
                failed_tests.append(FailedTest(name=t.get("name", "?"), trace=trace))
        except (json.JSONDecodeError, OSError) as e:
            logger.debug(f"Failed to parse {ctrf_file}: {e}")

    # Fallback: tail of stdout when no CTRF tests parsed.
    if not failed_tests:
        stdout_file = verifier_dir / "test-stdout.txt"
        if stdout_file.exists():
            try:
                txt = stdout_file.read_text()
                if len(txt) > _STDOUT_TAIL_CHARS:
                    raw_tail = "...[earlier output truncated]\n" + txt[-_STDOUT_TAIL_CHARS:]
                else:
                    raw_tail = txt
            except OSError as e:
                logger.debug(f"Failed to read {stdout_file}: {e}")

    # Bail entirely if we have nothing useful.
    if reward is None and not failed_tests and not raw_tail:
        return None

    return VerifierSignal(
        reward=reward,
        failed_tests=failed_tests,
        test_summary=test_summary,
        raw_tail=raw_tail,
    )


def find_and_parse_trajectory(
    logs_dir: str | Path | None,
    base_folder: str | Path,
) -> tuple[list[ReplayCommand], list[dict], VerifierSignal | None]:
    """Find trajectory folder for task, parse it, return (commands, messages, verifier_signal).

    Convenience function that combines find_trajectory_folder + extraction.
    The verifier signal is the failed initial run's verifier output —
    used by the summarizer to ground its reflection in test-level truth.
    """
    folder = find_trajectory_folder(logs_dir, base_folder)
    if folder is None:
        return [], [], None
    return (
        extract_commands(folder),
        extract_messages(folder),
        extract_verifier_signal(folder),
    )


# ---------------------------------------------------------------------------
# Replay executors
# ---------------------------------------------------------------------------


def _find_interrupted_commands(commands: list[ReplayCommand]) -> set[int]:
    """Return indices of commands immediately followed by an interrupt signal.

    When the original agent sent C-c (or another control sequence) right after
    a command, it means the agent intentionally killed that command.  There is
    no point replaying it during recovery — it would just hang.
    """
    skip: set[int] = set()
    for i in range(len(commands) - 1):
        next_ks = commands[i + 1].keystrokes.strip()
        if next_ks.startswith("C-") and commands[i].command:
            skip.add(i)
    return skip


async def replay_via_exec(
    environment: BaseEnvironment,
    commands: list[ReplayCommand],
    timeout_sec: int = 15,
) -> None:
    """Replay commands using environment.exec() — for installed agents.

    Uses ``asyncio.wait_for`` as a safety net because some environment
    backends (e.g. Modal) do not reliably enforce ``timeout_sec``.
    Commands that were followed by an interrupt signal (C-c) in the
    original trajectory are skipped entirely.
    """
    skip_indices = _find_interrupted_commands(commands)
    if skip_indices:
        logger.info(f"Skipping {len(skip_indices)} commands followed by interrupt signals")

    replayed = 0
    skipped_interrupt = 0
    timed_out = 0
    total_executable = 0

    for i, cmd in enumerate(commands):
        if not cmd.command:
            continue
        total_executable += 1
        if i in skip_indices:
            skipped_interrupt += 1
            continue
        try:
            await asyncio.wait_for(
                environment.exec(
                    f"bash -lc {shlex.quote(cmd.command)}",
                    timeout_sec=timeout_sec,
                ),
                timeout=timeout_sec,
            )
            replayed += 1
        except (asyncio.TimeoutError, TimeoutError):
            timed_out += 1
            logger.warning(
                f"Replay command timed out after {timeout_sec}s, skipping: {cmd.command[:80]}"
            )
            continue
        except Exception as e:
            logger.error(f"Replay error: {e}")
            continue

    logger.info(
        f"Replay complete: {replayed}/{total_executable} succeeded, "
        f"{skipped_interrupt} skipped (interrupted), {timed_out} timed out"
    )


async def replay_via_tmux(
    session,
    commands: list[ReplayCommand],
) -> str:
    """Replay commands using tmux keystrokes — for terminus-based agents.

    Args:
        session: TmuxSession instance with send_keys/capture_pane methods.
        commands: Commands to replay (uses keystrokes field).

    Returns:
        Last terminal output after replay.
    """
    if not session:
        logger.warning("TmuxSession not initialized, cannot replay")
        return ""

    for command in commands:
        try:
            await session.send_keys(
                keys=command.keystrokes,
                min_timeout_sec=0.5,
                max_timeout_sec=float(command.timeout_sec),
            )
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            logger.error(f"Replay error: {e}")
            continue

    try:
        last_output = await session.capture_pane()
        return last_output or ""
    except Exception:
        return ""
