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
                agent_dir = item / "agent"
                
                # Check for LettaCode events JSONL
                if agent_dir.exists():
                    for f in agent_dir.iterdir():
                        if f.name.startswith("letta_events_") and f.name.endswith(".jsonl"):
                            print(f"Found trajectory for hash {task_hash}: {item}")
                            return item
                
                # Check for ATIF trajectory.json
                trajectory_file = agent_dir / "trajectory.json" if agent_dir.exists() else None
                if trajectory_file and trajectory_file.exists():
                    print(f"Found trajectory for hash {task_hash}: {item}")
                    return item
                
                # Fall back to direct path
                trajectory_file = item / "trajectory.json"
                if trajectory_file.exists():
                    print(f"Found trajectory for hash {task_hash}: {item}")
                    return item

        print(f"No trajectory found for hash {task_hash} in {base_path}")
        return None

    def _extract_operations_from_trajectory(self, instruction: str) -> list[dict]:
        """Extract state-modifying operations from a failed trajectory.
        
        Supports two formats:
        1. LettaCode events JSONL (letta_events_*.jsonl)
        2. ATIF trajectory.json (for terminus-2 style agents)
        
        Returns list of operation dicts: {"tool": "Bash|Write|Edit", "args": {...}}
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
                    operations = self._parse_letta_events(f)
                    if operations:
                        return operations

        # Fall back to ATIF trajectory.json (only extracts Bash commands)
        trajectory_file = agent_dir / "trajectory.json" if agent_dir.exists() else None
        if not trajectory_file or not trajectory_file.exists():
            trajectory_file = trajectory_folder / "trajectory.json"
        if trajectory_file and trajectory_file.exists():
            commands = self._parse_atif_trajectory(trajectory_file)
            # Convert to operation format
            return [{"tool": "Bash", "args": {"command": cmd}} for cmd in commands]

        print(f"No trajectory found in {trajectory_folder}")
        return []

    def _parse_letta_events(self, events_file: Path) -> list[dict]:
        """Parse LettaCode events JSONL to extract operations that modify state.
        
        Extracts Bash, Write, and Edit tool calls that could corrupt the environment.
        Events are streamed as JSON fragments grouped by tool_call_id.
        
        Returns list of operation dicts: {"tool": "Bash|Write|Edit", "args": {...}}
        """
        print(f"Parsing LettaCode events from {events_file}")
        
        # Collect argument fragments by tool_call_id (preserving order)
        tool_call_fragments: dict[str, list[str]] = {}
        tool_call_names: dict[str, str] = {}
        tool_call_order: list[str] = []  # Track order of first appearance
        
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
                                tool_call_order.append(tool_call_id)
                            tool_call_fragments[tool_call_id].append(args_fragment)
        except Exception as e:
            print(f"Error parsing events file: {e}")
            return []

        # Reconstruct and extract operations (Bash, Write, Edit)
        operations = []
        import re
        
        # Tools that modify environment state
        state_modifying_tools = {"Bash", "Write", "Edit"}
        
        for tool_call_id in tool_call_order:
            tool_name = tool_call_names.get(tool_call_id, "")
            if tool_name not in state_modifying_tools:
                continue
            
            fragments = tool_call_fragments[tool_call_id]
            full_args = "".join(fragments)
            
            # Try full JSON parse first
            args_obj = None
            try:
                args_obj = json.loads(full_args)
            except json.JSONDecodeError:
                # Fallback: extract with regex for incomplete JSON
                args_obj = self._extract_args_regex(tool_name, full_args)
            
            if args_obj:
                operations.append({"tool": tool_name, "args": args_obj})

        bash_count = sum(1 for op in operations if op["tool"] == "Bash")
        write_count = sum(1 for op in operations if op["tool"] == "Write")
        edit_count = sum(1 for op in operations if op["tool"] == "Edit")
        print(f"Extracted {len(operations)} operations: {bash_count} Bash, {write_count} Write, {edit_count} Edit")
        return operations

    def _extract_args_regex(self, tool_name: str, full_args: str) -> dict | None:
        """Extract tool arguments using regex for incomplete JSON."""
        import re
        
        def unescape(s: str) -> str:
            return s.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
        
        if tool_name == "Bash":
            match = re.search(r'"command":\s*"((?:[^"\\]|\\.)*)"', full_args)
            if match:
                return {"command": unescape(match.group(1))}
        
        elif tool_name == "Write":
            file_match = re.search(r'"file_path":\s*"((?:[^"\\]|\\.)*)"', full_args)
            content_match = re.search(r'"content":\s*"((?:[^"\\]|\\.)*)"', full_args)
            if file_match and content_match:
                return {
                    "file_path": unescape(file_match.group(1)),
                    "content": unescape(content_match.group(1))
                }
        
        elif tool_name == "Edit":
            file_match = re.search(r'"file_path":\s*"((?:[^"\\]|\\.)*)"', full_args)
            old_match = re.search(r'"old_string":\s*"((?:[^"\\]|\\.)*)"', full_args)
            new_match = re.search(r'"new_string":\s*"((?:[^"\\]|\\.)*)"', full_args)
            if file_match and old_match and new_match:
                return {
                    "file_path": unescape(file_match.group(1)),
                    "old_string": unescape(old_match.group(1)),
                    "new_string": unescape(new_match.group(1))
                }
        
        return None

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
        
        1. Extract operations from failed trajectory (Bash, Write, Edit)
        2. Replay operations to corrupt the environment
        3. Run LettaCode with recovery instruction
        """
        # 1. Extract operations from failed trajectory
        operations = self._extract_operations_from_trajectory(instruction)
        
        if not operations:
            print("No operations found in trajectory, running LettaCode fresh")
        else:
            # 2. Replay operations to corrupt the environment
            print(f"Replaying {len(operations)} operations from previous trajectory...")
            for i, op in enumerate(operations):
                try:
                    await self._replay_operation(environment, op)
                except Exception as e:
                    # Continue even if an operation fails
                    print(f"Replay operation {i+1} ({op['tool']}) failed: {e}")
                    continue
            
            print(f"Finished replaying {len(operations)} operations")

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
        """Replay a single operation (Bash, Write, or Edit) in the environment."""
        tool = op["tool"]
        args = op["args"]
        
        if tool == "Bash":
            cmd = args.get("command", "")
            if cmd:
                await environment.exec(
                    f"bash -lc {shlex.quote(cmd)}",
                    timeout_sec=60,
                )
        
        elif tool == "Write":
            file_path = args.get("file_path", "")
            content = args.get("content", "")
            if file_path and content is not None:
                # Create parent directories and write file
                import tempfile
                with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmpf:
                    tmpf.write(content)
                    local_path = tmpf.name
                try:
                    # Ensure parent directory exists
                    parent_dir = str(Path(file_path).parent)
                    await environment.exec(f"mkdir -p {shlex.quote(parent_dir)}", timeout_sec=30)
                    # Upload file
                    await environment.upload_file(local_path, file_path)
                finally:
                    Path(local_path).unlink(missing_ok=True)
        
        elif tool == "Edit":
            file_path = args.get("file_path", "")
            old_string = args.get("old_string", "")
            new_string = args.get("new_string", "")
            if file_path and old_string:
                # Use sed for simple string replacement
                # Escape special characters for sed
                old_escaped = old_string.replace("\\", "\\\\").replace("/", "\\/").replace("&", "\\&").replace("\n", "\\n")
                new_escaped = new_string.replace("\\", "\\\\").replace("/", "\\/").replace("&", "\\&").replace("\n", "\\n")
                sed_cmd = f"sed -i 's/{old_escaped}/{new_escaped}/g' {shlex.quote(file_path)}"
                await environment.exec(f"bash -lc {shlex.quote(sed_cmd)}", timeout_sec=30)
