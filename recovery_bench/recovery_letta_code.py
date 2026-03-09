"""
RecoveryLettaCode - LettaCode agent that replays failed trajectories before running.

This agent extends LettaCode to:
1. Find a failed trajectory matching the task (by instruction hash)
2. Replay the commands from that trajectory to corrupt the environment
3. Run LettaCode with a recovery instruction
"""

import json
import logging
import os
import shlex
from pathlib import Path

from recovery_bench.letta_code_agent import LettaCode

logger = logging.getLogger(__name__)
from recovery_bench.utils import create_task_hash
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class RecoveryLettaCode(LettaCode):
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
        return "recovery-letta-code"

    @staticmethod
    def _find_events_file(agent_dir: Path) -> Path | None:
        """Find a LettaCode events JSONL file in an agent directory.

        Checks both naming conventions:
        - ``letta_events_{ts}.jsonl`` (saved by run() on normal completion)
        - ``{ts}.events.jsonl`` (downloaded by harbor on timeout)
        """
        if not agent_dir.exists():
            return None
        for f in agent_dir.iterdir():
            if f.name.startswith("letta_events_") and f.name.endswith(".jsonl"):
                return f
        for f in agent_dir.iterdir():
            if f.name.endswith(".events.jsonl"):
                return f
        return None

    def _find_trajectory_folder(self, task_hash: str) -> Path | None:
        """Find the trajectory folder based on task hash prefix.

        Checks for LettaCode events files first, then falls back to
        ATIF trajectory.json (terminus-2 format).
        """
        base_path = Path(self._trajectory_folder)

        if not base_path.exists():
            logger.warning(f"Trajectory folder not found: {base_path}")
            return None

        # Look for hash-prefixed directories (format: <hash>-<task-id>__<suffix>/)
        # First pass: prefer directories with LettaCode events
        for item in base_path.iterdir():
            if item.is_dir() and item.name.startswith(f"{task_hash}-"):
                if self._find_events_file(item / "agent"):
                    logger.debug(f"Found LettaCode trajectory for hash {task_hash}: {item}")
                    return item

        # Second pass: accept directories with ATIF trajectory.json
        for item in base_path.iterdir():
            if item.is_dir() and item.name.startswith(f"{task_hash}-"):
                for traj_path in [item / "agent" / "trajectory.json", item / "trajectory.json"]:
                    if traj_path.exists():
                        logger.debug(f"Found ATIF trajectory for hash {task_hash}: {item}")
                        return item

        logger.warning(f"No trajectory found for hash {task_hash} in {base_path}")
        return None

    def _extract_operations_from_trajectory(self, instruction: str) -> list[dict]:
        """Extract state-modifying operations from a failed LettaCode trajectory.
        
        Returns list of operation dicts: {"tool": "Bash|Write|Edit", "args": {...}}
        """
        task_hash = create_task_hash(instruction)
        logger.debug(f"Looking for trajectory with hash: {task_hash}")
        
        trajectory_folder = self._find_trajectory_folder(task_hash)
        if trajectory_folder is None:
            return []

        events_file = self._find_events_file(trajectory_folder / "agent")
        if events_file:
            return self._parse_letta_events(events_file)

        # Fallback: try ATIF trajectory.json (terminus-2 format)
        for traj_path in [
            trajectory_folder / "agent" / "trajectory.json",
            trajectory_folder / "trajectory.json",
        ]:
            if traj_path.exists():
                return self._parse_atif_trajectory(traj_path)

        logger.warning(f"No trajectory found in {trajectory_folder}")
        return []

    def _parse_letta_events(self, events_file: Path) -> list[dict]:
        """Parse LettaCode events JSONL to extract operations that modify state.
        
        Extracts Bash, Write, and Edit tool calls that could corrupt the environment.
        Events are streamed as JSON fragments grouped by tool_call_id.
        
        Returns list of operation dicts: {"tool": "Bash|Write|Edit", "args": {...}}
        """
        logger.debug(f"Parsing LettaCode events from {events_file}")
        
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
            logger.error(f"Error parsing events file: {e}")
            return []

        # Reconstruct and extract operations (Bash, Write, Edit)
        operations = []
        import re
        
        # Tools that modify environment state (across all providers)
        # Anthropic: Bash, Write, Edit
        # OpenAI/Codex: ShellCommand, Shell, shell_command, shell, ApplyPatch, apply_patch
        # Gemini: RunShellCommand, run_shell_command, WriteFileGemini, write_file_gemini, Replace, replace
        shell_tools = {"Bash", "ShellCommand", "Shell", "shell_command", "shell", "RunShellCommand", "run_shell_command"}
        write_tools = {"Write", "WriteFileGemini", "write_file_gemini"}
        edit_tools = {"Edit", "ApplyPatch", "apply_patch", "Replace", "replace"}
        state_modifying_tools = shell_tools | write_tools | edit_tools
        
        for tool_call_id in tool_call_order:
            tool_name = tool_call_names.get(tool_call_id, "")
            if tool_name not in state_modifying_tools:
                continue
            
            fragments = tool_call_fragments[tool_call_id]
            full_args = "".join(fragments)
            
            # Normalize tool name to canonical operation type
            if tool_name in shell_tools:
                canonical_tool = "Bash"
            elif tool_name in write_tools:
                canonical_tool = "Write"
            elif tool_name in edit_tools:
                canonical_tool = "Edit"
            else:
                continue
            
            # Try full JSON parse first
            args_obj = None
            try:
                args_obj = json.loads(full_args)
            except json.JSONDecodeError:
                # Fallback: extract with regex for incomplete JSON
                args_obj = self._extract_args_regex(tool_name, full_args)
            
            if args_obj:
                operations.append({"tool": canonical_tool, "args": args_obj, "original_tool": tool_name})

        bash_count = sum(1 for op in operations if op["tool"] == "Bash")
        write_count = sum(1 for op in operations if op["tool"] == "Write")
        edit_count = sum(1 for op in operations if op["tool"] == "Edit")
        logger.info(f"Extracted {len(operations)} operations: {bash_count} Bash, {write_count} Write, {edit_count} Edit")
        return operations

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

    def _extract_args_regex(self, tool_name: str, full_args: str) -> dict | None:
        """Extract tool arguments using regex for incomplete JSON.
        
        Handles different argument names across providers:
        - Shell: command (all providers)
        - Write: file_path+content (Anthropic), file_path+content (Gemini)
        - Edit: file_path+old_string+new_string (Anthropic), patch (Codex), 
                file_path+old_str+new_str or find+replace (Gemini)
        """
        import re
        
        def unescape(s: str) -> str:
            return s.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
        
        def extract_field(pattern: str) -> str | None:
            match = re.search(pattern, full_args)
            return unescape(match.group(1)) if match else None
        
        # Shell tools (Bash, ShellCommand, Shell, RunShellCommand, etc.)
        shell_tools = {"Bash", "ShellCommand", "Shell", "shell_command", "shell", "RunShellCommand", "run_shell_command"}
        if tool_name in shell_tools:
            command = extract_field(r'"command":\s*"((?:[^"\\]|\\.)*)"')
            if command:
                return {"command": command}
        
        # Write tools (Write, WriteFileGemini, write_file_gemini)
        write_tools = {"Write", "WriteFileGemini", "write_file_gemini"}
        if tool_name in write_tools:
            file_path = extract_field(r'"file_path":\s*"((?:[^"\\]|\\.)*)"')
            content = extract_field(r'"content":\s*"((?:[^"\\]|\\.)*)"')
            if file_path and content is not None:
                return {"file_path": file_path, "content": content}
        
        # Edit tools (Edit, ApplyPatch, apply_patch, Replace, replace)
        edit_tools = {"Edit", "ApplyPatch", "apply_patch", "Replace", "replace"}
        if tool_name in edit_tools:
            # Try Anthropic format: file_path + old_string + new_string
            file_path = extract_field(r'"file_path":\s*"((?:[^"\\]|\\.)*)"')
            old_string = extract_field(r'"old_string":\s*"((?:[^"\\]|\\.)*)"')
            new_string = extract_field(r'"new_string":\s*"((?:[^"\\]|\\.)*)"')
            if file_path and old_string is not None and new_string is not None:
                return {"file_path": file_path, "old_string": old_string, "new_string": new_string}
            
            # Try Gemini Replace format: file_path + old_str + new_str
            old_str = extract_field(r'"old_str":\s*"((?:[^"\\]|\\.)*)"')
            new_str = extract_field(r'"new_str":\s*"((?:[^"\\]|\\.)*)"')
            if file_path and old_str is not None and new_str is not None:
                return {"file_path": file_path, "old_string": old_str, "new_string": new_str}
            
            # Try Codex ApplyPatch format: patch (unified diff)
            patch = extract_field(r'"patch":\s*"((?:[^"\\]|\\.)*)"')
            if patch:
                return {"patch": patch}
        
        return None

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
            # Check for patch format (Codex ApplyPatch)
            patch = args.get("patch", "")
            if patch:
                # Apply unified diff patch
                import tempfile
                with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".patch") as tmpf:
                    tmpf.write(patch)
                    local_patch = tmpf.name
                try:
                    remote_patch = "/tmp/replay_patch.patch"
                    await environment.upload_file(local_patch, remote_patch)
                    await environment.exec(
                        f"bash -lc 'patch -p1 < {remote_patch} || true'",
                        timeout_sec=30,
                    )
                finally:
                    Path(local_patch).unlink(missing_ok=True)
            else:
                # String replacement format (Anthropic Edit, Gemini Replace)
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
