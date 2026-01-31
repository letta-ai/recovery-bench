"""
ReplayLettaCode - LettaCode agent that replays failed trajectories before running.

This agent extends LettaCode to:
1. Find a failed trajectory matching the task (by instruction hash)
2. Replay the commands from that trajectory to corrupt the environment
3. Run LettaCode with a recovery instruction
"""

import hashlib
import json
import os
import shlex
from pathlib import Path

from recovery_bench.letta_code_agent import LettaCode
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


def create_task_hash(task_description: str) -> str:
    """Create 8-character hash from task description."""
    return hashlib.sha256(task_description.encode("utf-8")).hexdigest()[:8]


class ReplayLettaCode(LettaCode):
    """
    LettaCode agent that replays a failed trajectory before running.
    
    This agent:
    1. Finds a previous failed trajectory by hashing the instruction
    2. Extracts and replays the commands to restore the corrupted state
    3. Runs LettaCode with a modified instruction indicating recovery
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trajectory_folder = os.getenv("TRAJECTORY_FOLDER", "./trajectories")

    @staticmethod
    def name() -> str:
        return "replay-letta-code"

    def _find_trajectory_folder(self, task_hash: str) -> Path | None:
        """Find the trajectory folder based on task hash prefix."""
        base_path = Path(self._trajectory_folder)
        
        if not base_path.exists():
            print(f"Trajectory folder not found: {base_path}")
            return None

        # Look for hash-prefixed directories (format: <hash>-<task-id>__<suffix>/)
        for item in base_path.iterdir():
            if item.is_dir() and item.name.startswith(f"{task_hash}-"):
                # Check agent/ subdirectory (Harbor output structure)
                trajectory_file = item / "agent" / "trajectory.json"
                if trajectory_file.exists():
                    print(f"Found trajectory for hash {task_hash}: {item}")
                    return item
                # Fall back to direct path
                trajectory_file = item / "trajectory.json"
                if trajectory_file.exists():
                    print(f"Found trajectory for hash {task_hash}: {item}")
                    return item

        print(f"No trajectory found for hash {task_hash} in {base_path}")
        return None

    def _extract_commands_from_trajectory(self, instruction: str) -> list[str]:
        """Extract bash commands from a failed trajectory.
        
        Supports two formats:
        1. LettaCode events JSONL (letta_events_*.jsonl)
        2. ATIF trajectory.json (for terminus-2 style agents)
        """
        task_hash = create_task_hash(instruction)
        print(f"Looking for trajectory with hash: {task_hash}")
        
        trajectory_folder = self._find_trajectory_folder(task_hash)
        if trajectory_folder is None:
            return []

        # Try LettaCode events JSONL first
        agent_dir = trajectory_folder / "agent"
        if agent_dir.exists():
            for f in agent_dir.iterdir():
                if f.name.startswith("letta_events_") and f.name.endswith(".jsonl"):
                    commands = self._parse_letta_events(f)
                    if commands:
                        return commands

        # Fall back to ATIF trajectory.json
        trajectory_file = agent_dir / "trajectory.json" if agent_dir.exists() else None
        if not trajectory_file or not trajectory_file.exists():
            trajectory_file = trajectory_folder / "trajectory.json"
        if trajectory_file and trajectory_file.exists():
            return self._parse_atif_trajectory(trajectory_file)

        print(f"No trajectory found in {trajectory_folder}")
        return []

    def _parse_letta_events(self, events_file: Path) -> list[str]:
        """Parse LettaCode events JSONL to extract bash commands.
        
        Events are streamed as JSON fragments grouped by tool_call_id.
        We need to concatenate fragments and parse to get the command.
        """
        print(f"Parsing LettaCode events from {events_file}")
        
        # Collect argument fragments by tool_call_id
        tool_call_fragments: dict[str, list[str]] = {}
        tool_call_names: dict[str, str] = {}
        
        try:
            with open(events_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line.startswith("{"):
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    # Look for approval_request_message with tool_call
                    if event.get("message_type") == "approval_request_message":
                        tool_call = event.get("tool_call", {})
                        if not tool_call:
                            continue
                        
                        tool_call_id = tool_call.get("tool_call_id")
                        tool_name = tool_call.get("name")
                        args_fragment = tool_call.get("arguments")
                        
                        if tool_call_id and args_fragment is not None:
                            if tool_call_id not in tool_call_fragments:
                                tool_call_fragments[tool_call_id] = []
                                tool_call_names[tool_call_id] = tool_name
                            tool_call_fragments[tool_call_id].append(args_fragment)
        except Exception as e:
            print(f"Error parsing events file: {e}")
            return []

        # Reconstruct and extract bash commands
        commands = []
        import re
        for tool_call_id, fragments in tool_call_fragments.items():
            tool_name = tool_call_names.get(tool_call_id, "")
            if tool_name != "Bash":
                continue
            
            # Concatenate fragments to get full JSON
            full_args = "".join(fragments)
            
            # Try full JSON parse first
            try:
                args_obj = json.loads(full_args)
                command = args_obj.get("command", "")
                if command and command.strip():
                    commands.append(command.strip())
                    continue
            except json.JSONDecodeError:
                pass
            
            # Fallback: extract command with regex (handles incomplete JSON)
            match = re.search(r'"command":\s*"((?:[^"\\]|\\.)*)"', full_args)
            if match:
                cmd = match.group(1)
                # Unescape common escape sequences
                cmd = cmd.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
                if cmd.strip():
                    commands.append(cmd.strip())

        print(f"Extracted {len(commands)} bash commands from LettaCode events")
        return commands

    def _parse_atif_trajectory(self, trajectory_file: Path) -> list[str]:
        """Parse ATIF trajectory.json to extract commands (terminus-2 style)."""
        print(f"Parsing ATIF trajectory from {trajectory_file}")
        
        try:
            with open(trajectory_file, "r") as f:
                trajectory = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading trajectory: {e}")
            return []

        commands = []
        
        # ATIF v1.5 format: steps array with tool_calls
        steps = trajectory.get("steps", trajectory)
        
        for step in steps:
            source = step.get("source", step.get("role", ""))
            
            if source in ("agent", "assistant"):
                # Extract commands from tool_calls
                tool_calls = step.get("tool_calls", [])
                for tool_call in tool_calls:
                    args = tool_call.get("arguments", {})
                    keystrokes = args.get("keystrokes", "")
                    if keystrokes and keystrokes.strip():
                        cmd = keystrokes.rstrip("\n")
                        if cmd:
                            commands.append(cmd)
                
                # Also try parsing message content for commands (old format)
                if not tool_calls:
                    content = step.get("message", step.get("content", ""))
                    try:
                        response = json.loads(content) if isinstance(content, str) else content
                        if isinstance(response, dict) and "commands" in response:
                            for cmd_obj in response["commands"]:
                                keystrokes = cmd_obj.get("keystrokes", "")
                                if keystrokes and keystrokes.strip():
                                    cmd = keystrokes.rstrip("\n")
                                    if cmd:
                                        commands.append(cmd)
                    except (json.JSONDecodeError, TypeError):
                        pass

        print(f"Extracted {len(commands)} commands from ATIF trajectory")
        return commands

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """
        Run the replay LettaCode agent.
        
        1. Extract commands from failed trajectory
        2. Replay commands to corrupt the environment
        3. Run LettaCode with recovery instruction
        """
        # 1. Extract commands from failed trajectory
        commands = self._extract_commands_from_trajectory(instruction)
        
        if not commands:
            print("No commands found in trajectory, running LettaCode fresh")
        else:
            # 2. Replay commands to corrupt the environment
            print(f"Replaying {len(commands)} commands from previous trajectory...")
            for i, cmd in enumerate(commands):
                try:
                    # Execute command via environment.exec()
                    # Wrap in bash -lc to ensure proper shell environment
                    await environment.exec(
                        f"bash -lc {shlex.quote(cmd)}",
                        timeout_sec=60,
                    )
                except Exception as e:
                    # Continue even if a command fails
                    print(f"Replay command {i+1} failed: {e}")
                    continue
            
            print(f"Finished replaying {len(commands)} commands")

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
