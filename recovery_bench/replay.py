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


def find_and_parse_trajectory(
    logs_dir: str | Path | None,
    base_folder: str | Path,
) -> tuple[list[ReplayCommand], list[dict]]:
    """Find trajectory folder for task, parse it, return (commands, messages).

    Convenience function that combines find_trajectory_folder + extraction.
    """
    folder = find_trajectory_folder(logs_dir, base_folder)
    if folder is None:
        return [], []
    return extract_commands(folder), extract_messages(folder)


# ---------------------------------------------------------------------------
# Replay executors
# ---------------------------------------------------------------------------


async def replay_via_exec(
    environment: BaseEnvironment,
    commands: list[ReplayCommand],
    timeout_sec: int = 60,
) -> None:
    """Replay commands using environment.exec() — for installed agents."""
    replayed = 0
    for cmd in commands:
        if not cmd.command:
            continue
        try:
            await environment.exec(
                f"bash -lc {shlex.quote(cmd.command)}",
                timeout_sec=timeout_sec,
            )
            replayed += 1
        except Exception as e:
            logger.error(f"Replay error: {e}")
            continue

    logger.info(f"Replayed {replayed}/{len(commands)} commands via exec")


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
