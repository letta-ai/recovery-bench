"""Recovery agent for Harbor's Pi installed agent.

Reasoning effort is controlled via the ``--thinking`` CLI flag (levels:
off, minimal, low, medium, high, xhigh).  Pi's unified LLM layer maps
this to provider-specific params: ``reasoningEffort`` for OpenAI,
``thinkingBudgetTokens`` for Anthropic, ``thinking.budgetTokens`` for
Gemini.  ``xhigh`` maps to ``high`` on non-OpenAI providers.

Pass ``thinking`` as a top-level config key (auto-extracted by
BaseInstalledAgent's CLI_FLAGS system).
"""

import logging

from harbor.agents.installed.pi import Pi
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from recovery_bench.agents.recovery_mixin import RecoveryMixin
from recovery_bench.replay import replay_via_exec
from recovery_bench.utils import save_usage

logger = logging.getLogger(__name__)


class RecoveryPi(RecoveryMixin, Pi):
    """Pi agent extended with trajectory replay for recovery."""

    def __init__(self, message_mode: str = "full", model_kwargs: dict = None, **kwargs):
        super().__init__(**(model_kwargs or {}), **kwargs)
        self._init_recovery(message_mode)

    @staticmethod
    def name() -> str:
        return "recovery-pi"

    async def setup(self, environment: BaseEnvironment) -> None:
        await super().setup(environment)
        commands, _ = self._parse_trajectory()
        if commands:
            await replay_via_exec(environment, commands)
            logger.info(f"Replayed {len(commands)} commands from previous trajectory")

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        recovery_instruction = await self._build_recovery_instruction(instruction)
        await super().run(recovery_instruction, environment, context)
        save_usage(self.logs_dir, context)
