"""
RecoveryTerminus - Extends Harbor's Terminus2 with trajectory replay for recovery.

This agent reads previous failed trajectories, replays commands to restore
the environment state during setup(), then runs Terminus2's agent loop
with a recovery-oriented prompt.

Variants:
- RecoveryTerminus: Full message history recovery
- RecoveryTerminusWithoutMessages: Environment-only recovery
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.environments.base import BaseEnvironment
from harbor.llms.chat import Chat
from harbor.models.agent.context import AgentContext

logger = logging.getLogger(__name__)


@dataclass
class ReplayCommand:
    """A command extracted from a previous trajectory for replay."""
    keystrokes: str
    timeout_sec: float = 15.0


class RecoveryTerminus(Terminus2):
    """
    Terminus2 agent extended with trajectory replay for recovery.

    During setup(), reads a previous failed trajectory, replays all commands
    to restore the corrupted environment state. During run(), injects prior
    messages and runs Terminus2's agent loop with a recovery prompt.
    """

    def __init__(self, model_kwargs: dict = None, **kwargs):
        # Extract recovery-specific kwargs before passing to Terminus2
        # model_kwargs contains things like reasoning_effort that Terminus2
        # accepts as direct params
        model_kwargs = model_kwargs or {}
        reasoning_effort = model_kwargs.pop("reasoning_effort", None)

        super().__init__(
            reasoning_effort=reasoning_effort,
            **kwargs,
        )

        self._base_folder = os.getenv("TRAJECTORY_FOLDER", "./trajectories")
        self._include_messages = True
        self._last_replay_output = ""
        self._replay_messages: list[dict] = []

    @staticmethod
    def name() -> str:
        return "recovery-terminus"

    async def setup(self, environment: BaseEnvironment) -> None:
        """Setup: start tmux (via Terminus2), then read and replay trajectories."""
        await super().setup(environment)

        # Read and replay trajectories during setup (not counted against agent timeout)
        commands, messages, n_episodes = self._read_trajectories()

        if commands:
            self._last_replay_output = await self._replay_commands(commands)
            logger.info(f"Replayed {len(commands)} commands from previous trajectory")
        else:
            logger.info("No commands found in trajectory, starting fresh")
            self._last_replay_output = ""

        self._replay_messages = messages

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Build a recovery prompt and run Terminus2's agent loop."""
        self._chat = Chat(self._llm, interleaved_thinking=self._interleaved_thinking)
        self._context = context

        if self._session is None:
            raise RuntimeError("Session is not set")

        # Inject prior conversation messages into chat history
        if self._include_messages and self._replay_messages:
            for msg in self._replay_messages:
                self._chat._messages.append(msg)

        # Build recovery prompt using terminus2's template format
        terminal_state = self._limit_output_length(self._last_replay_output)

        recovery_instruction = (
            f"{instruction}\n\n"
            "RECOVERY MODE: Previous attempt(s) to complete this task failed. "
            "The environment has been restored to the state after the failed attempt. "
            "Please analyze what went wrong and try a DIFFERENT approach."
        )

        initial_prompt = self._prompt_template.format(
            instruction=recovery_instruction,
            terminal_state=terminal_state,
        )

        from harbor.models.trajectories import Step
        from datetime import datetime, timezone

        self._trajectory_steps.append(
            Step(
                step_id=1,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="user",
                message=initial_prompt,
            )
        )

        actual_episodes = self._n_episodes
        try:
            actual_episodes = await self._run_agent_loop(
                initial_prompt=initial_prompt,
                chat=self._chat,
                logging_dir=self.logs_dir,
                original_instruction=instruction,
            )
        finally:
            # Populate context with metrics (same as Terminus2.run())
            context.rollout_details = (
                self._chat.rollout_details + self._subagent_rollout_details
            )
            context.n_input_tokens = (
                self._chat.total_input_tokens
                + self._subagent_metrics.total_prompt_tokens
            )
            context.n_output_tokens = (
                self._chat.total_output_tokens
                + self._subagent_metrics.total_completion_tokens
            )
            context.n_cache_tokens = (
                self._chat.total_cache_tokens
                + self._subagent_metrics.total_cached_tokens
            )
            total_cost = self._chat.total_cost + self._subagent_metrics.total_cost_usd
            context.cost_usd = total_cost if total_cost > 0 else None
            context.metadata = {
                "n_episodes": actual_episodes,
                "api_request_times_msec": self._api_request_times,
                "summarization_count": self._summarization_count,
            }
            if self._store_all_messages:
                context.metadata["all_messages"] = self._chat.messages

            self._dump_trajectory()

            # Also save usage.json for easy aggregation
            self._save_usage(context)

    def _save_usage(self, context: AgentContext) -> None:
        """Save usage stats to a separate JSON file in the task dir."""
        usage = {
            "prompt_tokens": context.n_input_tokens or 0,
            "completion_tokens": context.n_output_tokens or 0,
            "total_tokens": (context.n_input_tokens or 0) + (context.n_output_tokens or 0),
            "cost_usd": round(context.cost_usd or 0, 6),
        }

        if self.logs_dir:
            usage_path = Path(self.logs_dir).parent / "usage.json"
        else:
            usage_path = Path("usage.json")

        try:
            usage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(usage_path, "w") as f:
                json.dump(usage, f, indent=2)
            logger.info(f"Usage saved to {usage_path}")
        except Exception as e:
            logger.error(f"Failed to save usage: {e}")

    # ===== Trajectory replay helpers =====

    def _get_task_name(self) -> str | None:
        """Extract task name from logs_dir path.

        logs_dir is like: .../job-name/task-name__suffix/agent
        """
        if not self.logs_dir:
            return None
        task_dir_name = Path(self.logs_dir).parent.name
        if "__" in task_dir_name:
            return task_dir_name.rsplit("__", 1)[0]
        return task_dir_name

    def _find_trajectory_folder(self) -> Path | None:
        """Find the trajectory folder for this task in the traces directory."""
        task_name = self._get_task_name()
        if not task_name:
            logger.warning("Could not extract task name from logs_dir")
            return None

        base_path = Path(self._base_folder)
        if not base_path.exists():
            logger.warning(f"Trajectory folder not found: {base_path}")
            return None

        for item in base_path.iterdir():
            if not item.is_dir():
                continue
            # Dir format: <hash>-<task-name>__<suffix>
            dir_name = item.name
            # Strip hash prefix (8 hex chars + dash)
            if len(dir_name) > 9 and dir_name[8] == "-":
                dir_task = dir_name[9:]
            else:
                dir_task = dir_name
            # Strip __suffix
            if "__" in dir_task:
                dir_task = dir_task.rsplit("__", 1)[0]

            if dir_task == task_name:
                for traj_path in [item / "agent" / "trajectory.json", item / "trajectory.json"]:
                    if traj_path.exists():
                        logger.debug(f"Found trajectory for task {task_name}: {item}")
                        return item

        logger.warning(f"No trajectory found for task {task_name} in {base_path}")
        return None

    def _read_trajectories(self) -> tuple[list[ReplayCommand], list[dict], int]:
        """Read commands and messages from the previous trajectory."""
        trajectory_folder = self._find_trajectory_folder()
        if trajectory_folder is None:
            return [], [], 0
        return self._parse_trajectory(trajectory_folder)

    def _parse_trajectory(
        self, trajectory_folder: Path
    ) -> tuple[list[ReplayCommand], list[dict], int]:
        """Parse commands and messages from a trajectory folder."""
        trajectory_file = trajectory_folder / "agent" / "trajectory.json"
        if not trajectory_file.exists():
            trajectory_file = trajectory_folder / "trajectory.json"
        if not trajectory_file.exists():
            return [], [], 0

        try:
            with open(trajectory_file, "r") as f:
                trajectory = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return [], [], 0

        commands: list[ReplayCommand] = []
        messages: list[dict] = []
        n_episodes = 0

        steps = trajectory.get("steps", trajectory)

        for step in steps:
            source = step.get("source", step.get("role", ""))
            content = step.get("message", step.get("content", ""))

            if source in ("agent", "assistant"):
                n_episodes += 1
                # Extract commands from tool_calls (ATIF v1.5)
                tool_calls = step.get("tool_calls", [])
                for tool_call in tool_calls:
                    args = tool_call.get("arguments", {})
                    keystrokes = args.get("keystrokes", "")
                    if keystrokes:
                        commands.append(ReplayCommand(
                            keystrokes=keystrokes,
                            timeout_sec=int(args.get("duration", 1) * 10) + 5,
                        ))

                # Fallback: parse commands from message content (old format)
                if not tool_calls:
                    try:
                        response = json.loads(content) if isinstance(content, str) else content
                        if isinstance(response, dict) and "commands" in response:
                            for cmd in response["commands"]:
                                commands.append(ReplayCommand(
                                    keystrokes=cmd.get("keystrokes", ""),
                                    timeout_sec=cmd.get("timeout_sec", 120),
                                ))
                    except (json.JSONDecodeError, TypeError):
                        pass

            # Collect messages for context injection
            role = "assistant" if source == "agent" else source
            if role in ("user", "assistant", "system"):
                messages.append({"role": role, "content": content})

        return commands, messages, n_episodes

    async def _replay_commands(self, commands: list[ReplayCommand]) -> str:
        """Replay commands in the environment to restore state."""
        if not self._session:
            logger.warning("TmuxSession not initialized, cannot replay")
            return ""

        for command in commands:
            try:
                await self._session.send_keys(
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
            last_output = await self._session.capture_pane()
            return last_output or ""
        except Exception:
            return ""


class RecoveryTerminusWithoutMessages(RecoveryTerminus):
    """
    Recovery agent that only restores environment state (no message history).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._include_messages = False

    @staticmethod
    def name() -> str:
        return "recovery-terminus-without-messages"


class RecoveryTerminusWithMessageSummaries(RecoveryTerminus):
    """
    Recovery agent that injects a summarized version of the previous
    conversation history instead of the full messages.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._include_messages = True

    @staticmethod
    def name() -> str:
        return "recovery-terminus-with-summaries"

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Summarize replay messages before running the agent loop."""
        if self._replay_messages:
            summary = await self._summarize_messages(self._replay_messages)
            self._replay_messages = [
                {"role": "assistant", "content": f"Summary of previous attempts:\n{summary}"}
            ]
        await super().run(instruction, environment, context)

    async def _summarize_messages(self, messages: list[dict]) -> str:
        """Use the LLM to summarize previous conversation messages."""
        prompt = (
            "Please summarize the following conversation concisely, "
            "focusing on what was attempted and what went wrong:\n\n"
            + json.dumps(messages, indent=2)
        )
        try:
            response = await self._llm.call(prompt=prompt, message_history=[])
            return response.content
        except Exception:
            return "Previous attempts to complete this task failed."
