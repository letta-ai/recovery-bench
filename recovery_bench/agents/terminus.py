"""RecoveryTerminus - Extends Harbor's Terminus2 with trajectory replay for recovery.

This agent reads previous failed trajectories, replays commands to restore
the environment state during setup(), then runs Terminus2's agent loop
with a recovery-oriented prompt.

Variants:
- RecoveryTerminus: Full message history recovery
- RecoveryTerminusWithoutMessages: Environment-only recovery
- RecoveryTerminusWithMessageSummaries: Summarized history recovery
- BaselineTerminus: Fresh start (no replay) for comparison
"""

import json
import logging
import os
from datetime import datetime, timezone

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.environments.base import BaseEnvironment
from harbor.llms.chat import Chat
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import Step

from recovery_bench.replay import (
    extract_commands,
    extract_messages,
    replay_via_tmux,
)
from recovery_bench.utils import find_trajectory_folder, save_usage

logger = logging.getLogger(__name__)


class RecoveryTerminus(Terminus2):
    """Terminus2 agent extended with trajectory replay for recovery.

    During setup(), reads a previous failed trajectory, replays all commands
    to restore the corrupted environment state. During run(), injects prior
    messages and runs Terminus2's agent loop with a recovery prompt.
    """

    def __init__(self, model_kwargs: dict = None, **kwargs):
        # model_kwargs from config files (e.g. reasoning_effort, temperature)
        # are forwarded as direct params to Terminus2
        super().__init__(**(model_kwargs or {}), **kwargs)

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
        trajectory_folder = find_trajectory_folder(self.logs_dir, self._base_folder)
        if trajectory_folder is None:
            logger.info("No trajectory found, starting fresh")
            return

        commands = extract_commands(trajectory_folder)
        messages = extract_messages(trajectory_folder)

        if commands:
            self._last_replay_output = await replay_via_tmux(self._session, commands)
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
            context.rollout_details = self._chat.rollout_details + self._subagent_rollout_details
            context.n_input_tokens = (
                self._chat.total_input_tokens + self._subagent_metrics.total_prompt_tokens
            )
            context.n_output_tokens = (
                self._chat.total_output_tokens + self._subagent_metrics.total_completion_tokens
            )
            context.n_cache_tokens = (
                self._chat.total_cache_tokens + self._subagent_metrics.total_cached_tokens
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
            save_usage(self.logs_dir, context)


class BaselineTerminus(Terminus2):
    """Baseline agent: runs Terminus2 fresh on the task with no replay.

    Used to compare "strong model from scratch" vs "strong model with recovery".
    Accepts model_kwargs so it works with the same pipeline.
    """

    def __init__(self, model_kwargs: dict = None, **kwargs):
        super().__init__(**(model_kwargs or {}), **kwargs)

    @staticmethod
    def name() -> str:
        return "baseline-terminus"

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Run Terminus2, then save usage.json for aggregation."""
        await super().run(instruction, environment, context)
        save_usage(self.logs_dir, context)


class RecoveryTerminusWithoutMessages(RecoveryTerminus):
    """Recovery agent that only restores environment state (no message history)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._include_messages = False

    @staticmethod
    def name() -> str:
        return "recovery-terminus-without-messages"


class RecoveryTerminusWithMessageSummaries(RecoveryTerminus):
    """Recovery agent that injects a summarized version of the previous
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
