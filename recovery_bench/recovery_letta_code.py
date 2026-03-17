"""
RecoveryLettaCode - LettaCode agent that replays failed ATIF trajectories before running.

This agent extends LettaCode to:
1. Find a failed ATIF trajectory matching the task (by instruction hash)
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
from recovery_bench.utils import create_task_hash

logger = logging.getLogger(__name__)


class RecoveryLettaCode(LettaCode):
    """
    LettaCode agent that replays a failed ATIF trajectory before running.

    This agent:
    1. Finds a previous failed trajectory by hashing the instruction
    2. Replays bash commands to restore the corrupted state
    3. Runs LettaCode with a modified instruction indicating recovery
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trajectory_folder = os.getenv("TRAJECTORY_FOLDER", "./trajectories")

    @staticmethod
    def name() -> str:
        return "recovery-letta-code"

    def _find_trajectory_folder(self, task_hash: str) -> Path | None:
        """Find the trajectory folder based on task hash prefix."""
        base_path = Path(self._trajectory_folder)

        if not base_path.exists():
            logger.warning(f"Trajectory folder not found: {base_path}")
            return None

        # Look for hash-prefixed directories (format: <hash>-<task-id>__<suffix>/)
        for item in base_path.iterdir():
            if item.is_dir() and item.name.startswith(f"{task_hash}-"):
                for traj_path in [item / "agent" / "trajectory.json", item / "trajectory.json"]:
                    if traj_path.exists():
                        logger.debug(f"Found ATIF trajectory for hash {task_hash}: {item}")
                        return item

        logger.warning(f"No trajectory found for hash {task_hash} in {base_path}")
        return None

    def _extract_operations_from_trajectory(self, instruction: str) -> list[dict]:
        """Extract bash commands from a failed ATIF trajectory.

        Returns list of operation dicts: {"tool": "Bash", "args": {"command": ...}}
        """
        task_hash = create_task_hash(instruction)
        logger.debug(f"Looking for trajectory with hash: {task_hash}")

        trajectory_folder = self._find_trajectory_folder(task_hash)
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
        """
        Run the replay LettaCode agent.

        1. Extract bash commands from failed ATIF trajectory
        2. Replay commands to corrupt the environment
        3. Run LettaCode with recovery instruction
        """
        # 1. Extract operations from failed trajectory
        operations = self._extract_operations_from_trajectory(instruction)
        
        if not operations:
            logger.info("No operations found in trajectory, running LettaCode fresh")
        else:
            # 2. Replay operations to corrupt the environment
            logger.info(f"Replaying {len(operations)} operations from previous trajectory...")
            for i, op in enumerate(operations):
                try:
                    await self._replay_operation(environment, op)
                except Exception as e:
                    # Continue even if an operation fails
                    logger.error(f"Replay operation {i+1} ({op['tool']}) failed: {e}")
                    continue
            
            logger.info(f"Finished replaying {len(operations)} operations")

        # 3. Create recovery instruction
        recovery_instruction = (
            "RECOVERY MODE: The previous attempt to complete this task failed. "
            "The environment has been restored to the state after the failed attempt. "
            "Please analyze what went wrong and try a DIFFERENT approach.\n\n"
            "--- ORIGINAL TASK ---\n"
            f"{instruction}"
        )

        # 4. Run LettaCode with recovery instruction
        await super().run(recovery_instruction, environment, context)

    async def _replay_operation(self, environment: BaseEnvironment, op: dict) -> None:
        """Replay a single bash command in the environment."""
        cmd = op["args"].get("command", "")
        if cmd:
            await environment.exec(
                f"bash -lc {shlex.quote(cmd)}",
                timeout_sec=60,
            )
