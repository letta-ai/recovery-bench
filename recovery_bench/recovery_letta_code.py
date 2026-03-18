"""
RecoveryLettaCode - LettaCode agent that replays failed ATIF trajectories before running.

This agent extends LettaCode to:
1. Find a failed ATIF trajectory matching the task (by task name from logs_dir)
2. Replay the bash commands from that trajectory to corrupt the environment
3. Run LettaCode with a recovery instruction
"""

import json
import logging
import os
import shlex
from pathlib import Path

from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from recovery_bench.letta_code_agent import LettaCode
from recovery_bench.utils import find_trajectory_folder

logger = logging.getLogger(__name__)


class RecoveryLettaCode(LettaCode):
    """
    LettaCode agent that replays a failed ATIF trajectory before running.

    This agent:
    1. Finds a previous failed trajectory by task name (from logs_dir)
    2. Replays bash commands to restore the corrupted state
    3. Runs LettaCode with a modified instruction indicating recovery
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trajectory_folder = os.getenv("TRAJECTORY_FOLDER", "./trajectories")

    @staticmethod
    def name() -> str:
        return "recovery-letta-code"

    async def setup(self, environment: BaseEnvironment) -> None:
        """Install LettaCode, then replay the failed trajectory."""
        await super().setup(environment)

        operations = self._extract_operations_from_trajectory()
        if not operations:
            logger.info("No operations found in trajectory, will run LettaCode fresh")
            return

        logger.info(f"Replaying {len(operations)} operations from previous trajectory...")
        for i, op in enumerate(operations):
            try:
                await self._replay_operation(environment, op)
            except Exception as e:
                logger.error(f"Replay operation {i+1} ({op['tool']}) failed: {e}")
                continue
        logger.info(f"Finished replaying {len(operations)} operations")

    def _extract_operations_from_trajectory(self) -> list[dict]:
        """Extract bash commands from a failed ATIF trajectory.

        Returns list of operation dicts: {"tool": "Bash", "args": {"command": ...}}
        """
        trajectory_folder = find_trajectory_folder(self.logs_dir, self._trajectory_folder)
        if trajectory_folder is None:
            return []

        for traj_path in [
            trajectory_folder / "agent" / "trajectory.json",
            trajectory_folder / "trajectory.json",
        ]:
            if traj_path.exists():
                return self._parse_atif_trajectory(traj_path)

        logger.warning(f"No trajectory found in {trajectory_folder}")
        return []

    def _parse_atif_trajectory(self, trajectory_file: Path) -> list[dict]:
        """Parse ATIF trajectory.json (terminus-2 format) to extract Bash operations.

        Mirrors the parsing logic from RecoveryTerminus._parse_trajectory() but
        returns operations in the format expected by _replay_operation().

        Returns list of operation dicts: {"tool": "Bash", "args": {"command": ...}}
        """
        logger.debug(f"Parsing ATIF trajectory from {trajectory_file}")

        try:
            with open(trajectory_file, "r") as f:
                trajectory = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.error(f"Failed to read ATIF trajectory: {trajectory_file}")
            return []

        operations = []
        steps = trajectory.get("steps", trajectory)

        for step in steps:
            source = step.get("source", step.get("role", ""))
            if source not in ("agent", "assistant"):
                continue

            content = step.get("message", step.get("content", ""))

            # Extract commands from tool_calls (ATIF v1.5+)
            tool_calls = step.get("tool_calls", [])
            for tool_call in tool_calls:
                args = tool_call.get("arguments", {})
                keystrokes = args.get("keystrokes", "")
                if keystrokes:
                    cmd = keystrokes.rstrip("\n").rstrip("\r")
                    # Skip control sequences and empty commands
                    if cmd and not cmd.startswith("C-"):
                        operations.append({"tool": "Bash", "args": {"command": cmd}})

            # Fallback: parse commands from message content (old ATIF format)
            if not tool_calls:
                try:
                    response = json.loads(content) if isinstance(content, str) else content
                    if isinstance(response, dict) and "commands" in response:
                        for cmd_obj in response["commands"]:
                            keystrokes = cmd_obj.get("keystrokes", "")
                            if keystrokes:
                                cmd = keystrokes.rstrip("\n").rstrip("\r")
                                if cmd and not cmd.startswith("C-"):
                                    operations.append({"tool": "Bash", "args": {"command": cmd}})
                except (json.JSONDecodeError, TypeError):
                    pass

        logger.info(f"Extracted {len(operations)} Bash operations from ATIF trajectory")
        return operations

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Prepend recovery prompt, then delegate to LettaCode."""
        recovery_instruction = (
            "RECOVERY MODE: The previous attempt to complete this task failed. "
            "The environment has been restored to the state after the failed attempt. "
            "Please analyze what went wrong and try a DIFFERENT approach.\n\n"
            "--- ORIGINAL TASK ---\n"
            f"{instruction}"
        )
        await super().run(recovery_instruction, environment, context)

    async def _replay_operation(self, environment: BaseEnvironment, op: dict) -> None:
        """Replay a single bash command in the environment."""
        cmd = op["args"].get("command", "")
        if cmd:
            await environment.exec(
                f"bash -lc {shlex.quote(cmd)}",
                timeout_sec=60,
            )
