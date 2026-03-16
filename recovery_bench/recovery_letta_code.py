"""
RecoveryLettaCode - LettaCode agent that replays failed trajectories before running.

This agent extends LettaCode to:
1. Find a failed trajectory matching the task (by instruction hash)
2. Replay the commands from that trajectory to corrupt the environment
3. Run LettaCode with a recovery instruction

Variants (matching RecoveryTerminus naming):
- RecoveryLettaCode: Full message history injected into the prompt
- RecoveryLettaCodeWithoutMessages: Environment-only recovery (no history)
- RecoveryLettaCodeWithMessageSummaries: LLM-summarized history injected into the prompt
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
    2. Extracts and replays state-modifying operations to restore corrupted state
    3. Injects prior message history and runs LettaCode in recovery mode
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

    def _extract_recovery_context_from_trajectory(self, instruction: str) -> tuple[list[dict], list[dict]]:
        """Extract operations + message history from a failed trajectory.

        Returns:
            (operations, messages)
            - operations: list[{"tool": "Bash|Write|Edit", "args": {...}}]
            - messages: list[{"role": "user|assistant|system", "content": str}]
        """
        task_hash = create_task_hash(instruction)
        logger.debug(f"Looking for trajectory with hash: {task_hash}")
        
        trajectory_folder = self._find_trajectory_folder(task_hash)
        if trajectory_folder is None:
            return [], []

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
        return [], []

    def _parse_letta_events(self, events_file: Path) -> tuple[list[dict], list[dict]]:
        """Parse LettaCode events JSONL to extract operations + messages.

        - Operations are reconstructed from approval_request_message fragments.
        - Messages are reconstructed from streamed assistant/user/system chunks.
        """
        logger.debug(f"Parsing LettaCode events from {events_file}")
        
        # Collect argument fragments by tool_call_id (preserving order)
        tool_call_fragments: dict[str, list[str]] = {}
        tool_call_names: dict[str, str] = {}
        tool_call_order: list[str] = []  # Track order of first appearance

        # Collect streamed message fragments by message id/otid
        message_fragments: dict[str, list[tuple[int, str]]] = {}
        message_roles: dict[str, str] = {}
        message_order: list[str] = []
        
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
                    
                    message_type = event.get("message_type")

                    # Collect assistant/user/system message chunks
                    if message_type in {"assistant_message", "user_message", "system_message"}:
                        role_map = {
                            "assistant_message": "assistant",
                            "user_message": "user",
                            "system_message": "system",
                        }
                        role = role_map.get(message_type)
                        content = event.get("content")
                        seq_raw = event.get("seq_id")
                        if isinstance(seq_raw, int):
                            seq_id = seq_raw
                        elif isinstance(seq_raw, str) and seq_raw.isdigit():
                            seq_id = int(seq_raw)
                        else:
                            seq_id = 0
                        if role and isinstance(content, str):
                            message_id = event.get("otid") or event.get("id") or f"seq-{seq_id}"
                            if message_id not in message_fragments:
                                message_fragments[message_id] = []
                                message_roles[message_id] = role
                                message_order.append(message_id)
                            message_fragments[message_id].append((seq_id, content))

                    # Look for approval_request_message with tool_call
                    if message_type == "approval_request_message":
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
            return [], []

        # Reconstruct and extract operations (Bash, Write, Edit)
        operations = []
        
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

        # Reconstruct full messages from streamed chunks
        messages: list[dict] = []
        for message_id in message_order:
            role = message_roles.get(message_id)
            fragments = message_fragments.get(message_id, [])
            if not role or not fragments:
                continue
            fragments.sort(key=lambda x: x[0])
            content = "".join(fragment for _, fragment in fragments)
            if content:
                messages.append({"role": role, "content": content})

        logger.info(f"Extracted {len(messages)} messages from Letta events")
        return operations, messages

    def _parse_atif_trajectory(self, trajectory_file: Path) -> tuple[list[dict], list[dict]]:
        """Parse ATIF trajectory.json to extract Bash operations + message history.

        Mirrors the parsing logic from RecoveryTerminus._parse_trajectory() but
        returns operations in the format expected by _replay_operation().

        Returns:
            (operations, messages)
        """
        logger.debug(f"Parsing ATIF trajectory from {trajectory_file}")

        try:
            with open(trajectory_file, "r") as f:
                trajectory = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.error(f"Failed to read ATIF trajectory: {trajectory_file}")
            return [], []

        operations = []
        messages = []
        steps = trajectory.get("steps", trajectory)

        for step in steps:
            source = step.get("source", step.get("role", ""))
            content = step.get("message", step.get("content", ""))

            # Collect messages for context injection
            role = "assistant" if source == "agent" else source
            if role in ("user", "assistant", "system"):
                messages.append({"role": role, "content": content})

            if source not in ("agent", "assistant"):
                continue

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
        logger.info(f"Extracted {len(messages)} messages from ATIF trajectory")
        return operations, messages

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
        3. Build recovery instruction (subclasses can override _build_recovery_instruction)
        4. Run LettaCode with recovery instruction
        """
        # 1. Extract operations + messages from failed trajectory
        self._replay_operations, self._replay_messages = self._extract_recovery_context_from_trajectory(instruction)

        if not self._replay_operations:
            logger.info("No operations found in trajectory, running LettaCode fresh")
        else:
            # 2. Replay operations to corrupt the environment
            logger.info(f"Replaying {len(self._replay_operations)} operations from previous trajectory...")
            for i, op in enumerate(self._replay_operations):
                try:
                    await self._replay_operation(environment, op)
                except Exception as e:
                    # Continue even if an operation fails
                    logger.error(f"Replay operation {i+1} ({op['tool']}) failed: {e}")
                    continue

            logger.info(f"Finished replaying {len(self._replay_operations)} operations")

        # 3. Build recovery instruction (hook for subclasses)
        recovery_instruction = await self._build_recovery_instruction(instruction)

        # 4. Run LettaCode with recovery instruction
        await super().run(recovery_instruction, environment, context)

    async def _build_recovery_instruction(self, instruction: str) -> str:
        """Build recovery instruction with full prior message history."""
        messages_text = self._format_messages_as_text(self._replay_messages)
        history_block = (
            "\n\n--- PREVIOUS ATTEMPT MESSAGES ---\n"
            f"{messages_text}\n"
            "--- END PREVIOUS ATTEMPT ---"
        ) if messages_text else ""

        return (
            "RECOVERY MODE: The previous attempt to complete this task failed. "
            "The environment has been restored to the state after the failed attempt. "
            "Below is the full message history from the failed attempt. "
            "Please analyze what went wrong and try a DIFFERENT approach.\n"
            f"{history_block}\n\n"
            "--- ORIGINAL TASK ---\n"
            f"{instruction}"
        )

    @staticmethod
    def _format_operations_as_text(operations: list[dict]) -> str:
        """Format extracted operations into a human-readable text log."""
        lines = []
        for i, op in enumerate(operations, 1):
            tool = op["tool"]
            args = op["args"]
            if tool == "Bash":
                lines.append(f"{i}. [Bash] {args.get('command', '')}")
            elif tool == "Write":
                path = args.get("file_path", "")
                content = args.get("content", "")
                preview = content[:200] + "..." if len(content) > 200 else content
                lines.append(f"{i}. [Write] {path}\n   Content: {preview}")
            elif tool == "Edit":
                if "patch" in args:
                    patch = args["patch"]
                    preview = patch[:200] + "..." if len(patch) > 200 else patch
                    lines.append(f"{i}. [Edit/Patch]\n   {preview}")
                else:
                    path = args.get("file_path", "")
                    old = args.get("old_string", "")
                    new = args.get("new_string", "")
                    lines.append(f"{i}. [Edit] {path}\n   - {old!r}\n   + {new!r}")
        return "\n".join(lines)

    @staticmethod
    def _format_messages_as_text(messages: list[dict]) -> str:
        """Format prior messages into role-tagged text for prompt injection."""
        lines = []
        for msg in messages:
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
            if content:
                lines.append(f"[{role}] {content}")
        return "\n\n".join(lines)

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


class RecoveryLettaCodeWithoutMessages(RecoveryLettaCode):
    """
    Recovery agent that restores environment state only (no prior message/operation history).

    Analogous to RecoveryTerminusWithoutMessages.
    """

    @staticmethod
    def name() -> str:
        return "recovery-letta-code-without-messages"

    async def _build_recovery_instruction(self, instruction: str) -> str:
        return (
            "RECOVERY MODE: The previous attempt to complete this task failed. "
            "The environment has been restored to the state after the failed attempt. "
            "Please analyze what went wrong and try a DIFFERENT approach.\n\n"
            "--- ORIGINAL TASK ---\n"
            f"{instruction}"
        )


class RecoveryLettaCodeWithMessageSummaries(RecoveryLettaCode):
    """
    Recovery agent that includes an LLM-generated summary of the previous
    attempt's message history in the instruction.

    Analogous to RecoveryTerminusWithMessageSummaries.
    """

    @staticmethod
    def name() -> str:
        return "recovery-letta-code-with-message-summaries"

    async def _build_recovery_instruction(self, instruction: str) -> str:
        messages_text = self._format_messages_as_text(self._replay_messages)

        if messages_text:
            summary = await self._summarize_messages(messages_text)
        else:
            summary = "Previous attempts to complete this task failed."

        return (
            "RECOVERY MODE: The previous attempt to complete this task failed. "
            "The environment has been restored to the state after the failed attempt. "
            "Below is a summary of what was tried previously. "
            "Please analyze what went wrong and try a DIFFERENT approach.\n\n"
            f"--- SUMMARY OF PREVIOUS ATTEMPT ---\n"
            f"{summary}\n"
            f"--- END SUMMARY ---\n\n"
            "--- ORIGINAL TASK ---\n"
            f"{instruction}"
        )

    async def _summarize_messages(self, messages_text: str) -> str:
        """Use an LLM to summarize the previous attempt's messages."""
        from litellm import acompletion

        prompt = (
            "Below is the full conversation history from a failed attempt to complete a coding task. "
            "Please provide a concise "
            "summary focusing on:\n"
            "1. What approach was taken\n"
            "2. What files were modified\n"
            "3. What likely went wrong\n\n"
            f"{messages_text}"
        )

        try:
            response = await acompletion(
                model=self.model_name or "anthropic/claude-haiku-4-5-20251001",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"Failed to summarize messages via LLM: {e}")
            return "Previous attempts to complete this task failed."
