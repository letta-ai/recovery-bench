"""RecoveryLettaCode - Extends LettaCode with trajectory replay for recovery."""

import logging

from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from recovery_bench.agents.letta_code import LettaCode
from recovery_bench.agents.recovery_mixin import RecoveryMixin

logger = logging.getLogger(__name__)


class RecoveryLettaCode(RecoveryMixin, LettaCode):
    """LettaCode agent that replays a failed ATIF trajectory before running.

    This agent:
    1. Finds a previous failed trajectory by task name (from logs_dir)
    2. Replays bash commands to restore the corrupted state
    3. Runs LettaCode with a modified instruction indicating recovery

    Args:
        message_mode: How to use messages from the previous trajectory.
            ``"full"`` includes the full transcript, ``"none"`` skips them,
            ``"summary"`` summarizes via LLM first, ``"initial"`` skips
            both replay and preamble for a vanilla initial run.
            Default: ``"full"``.
    """

    def __init__(self, *args, message_mode: str = "full", **kwargs):
        super().__init__(*args, **kwargs)
        self._init_recovery(message_mode)

    @staticmethod
    def name() -> str:
        return "recovery-letta-code"

    async def setup(self, environment: BaseEnvironment) -> None:
        """Install LettaCode, then replay the failed trajectory."""
        await super().setup(environment)
        await self._maybe_replay_exec(environment)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Prepend recovery prompt, then delegate to LettaCode."""
        recovery_instruction = await self._build_recovery_instruction(instruction)
        await super().run(recovery_instruction, environment, context)
