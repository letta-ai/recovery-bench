"""Recovery agent for Harbor's OpenHands SDK installed agent.

Works around a Harbor 0.4.0 gap: the adapter stores ``reasoning_effort``
but never forwards it to the runner script.  We patch the uploaded runner
to read ``REASONING_EFFORT`` and ``EXTENDED_THINKING_BUDGET`` env vars and
pass them to the SDK's ``LLM()`` constructor.

The V1 SDK's LLM class uses two levers:
- ``reasoning_effort`` — for OpenAI models (stripped for Claude by the SDK)
- ``extended_thinking_budget`` — for Anthropic models (default 200k tokens)
"""

import logging

from harbor.agents.installed.openhands_sdk import OpenHandsSDK
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from recovery_bench.agents.recovery_mixin import RecoveryMixin
from recovery_bench.replay import replay_via_exec
from recovery_bench.utils import save_usage

logger = logging.getLogger(__name__)


class RecoveryOpenHandsSDK(RecoveryMixin, OpenHandsSDK):
    """OpenHands SDK agent extended with trajectory replay for recovery."""

    def __init__(self, message_mode: str = "full", model_kwargs: dict = None, **kwargs):
        model_kwargs = dict(model_kwargs or {})
        self._extended_thinking_budget: int | None = model_kwargs.pop(
            "extended_thinking_budget", None
        )
        super().__init__(**model_kwargs, **kwargs)
        # Expose as env vars so the patched runner can forward to LLM().
        if self._reasoning_effort:
            self._extra_env["REASONING_EFFORT"] = self._reasoning_effort
        if self._extended_thinking_budget is not None:
            self._extra_env["EXTENDED_THINKING_BUDGET"] = str(self._extended_thinking_budget)
        self._init_recovery(message_mode)

    @staticmethod
    def name() -> str:
        return "recovery-openhands-sdk"

    async def install(self, environment: BaseEnvironment) -> None:
        await super().install(environment)
        # Patch the runner to forward REASONING_EFFORT (OpenAI) and
        # EXTENDED_THINKING_BUDGET (Anthropic) to LLM().
        local_runner = self.logs_dir / "run_agent.py"
        content = local_runner.read_text()
        content = content.replace(
            "    llm = LLM(**llm_kwargs)",
            '    _re = os.environ.get("REASONING_EFFORT")\n'
            "    if _re:\n"
            '        llm_kwargs["reasoning_effort"] = _re\n'
            '    _etb = os.environ.get("EXTENDED_THINKING_BUDGET")\n'
            "    if _etb:\n"
            '        llm_kwargs["extended_thinking_budget"] = int(_etb)\n'
            "    llm = LLM(**llm_kwargs)",
        )
        local_runner.write_text(content)
        await environment.upload_file(
            source_path=local_runner,
            target_path="/installed-agent/run_agent.py",
        )
        await environment.exec(
            command="chmod +x /installed-agent/run_agent.py",
            user="root",
        )

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
