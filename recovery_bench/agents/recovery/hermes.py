"""Recovery agent for Harbor's Hermes installed agent.

Hermes supports ``agent.reasoning_effort`` in its config.yaml (levels:
none, minimal, low, medium, high, xhigh).  The Harbor adapter's static
``_build_config_yaml`` does not expose this, so we override it as an
instance method to inject the setting when ``reasoning_effort`` is
provided via ``model_kwargs``.
"""

import logging

import yaml
from harbor.agents.installed.hermes import Hermes
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from recovery_bench.agents.recovery_mixin import RecoveryMixin
from recovery_bench.replay import replay_via_exec
from recovery_bench.utils import save_usage

logger = logging.getLogger(__name__)


class RecoveryHermes(RecoveryMixin, Hermes):
    """Hermes agent extended with trajectory replay for recovery."""

    def __init__(self, message_mode: str = "full", model_kwargs: dict = None, **kwargs):
        model_kwargs = dict(model_kwargs or {})
        self._hermes_reasoning_effort: str | None = model_kwargs.pop("reasoning_effort", None)
        super().__init__(**model_kwargs, **kwargs)
        self._init_recovery(message_mode)

    @staticmethod
    def name() -> str:
        return "recovery-hermes"

    # Override the parent's @staticmethod as an instance method so we can
    # inject agent.reasoning_effort.  Hermes.run() calls
    # ``self._build_config_yaml(cli_model)`` — Python MRO dispatches here.
    def _build_config_yaml(self, model: str) -> str:
        config_yaml = Hermes._build_config_yaml(model)
        if self._hermes_reasoning_effort:
            config = yaml.safe_load(config_yaml)
            config.setdefault("agent", {})["reasoning_effort"] = self._hermes_reasoning_effort
            return yaml.dump(config, default_flow_style=False)
        return config_yaml

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
