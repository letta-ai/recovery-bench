"""Recovery agent for Harbor's ClaudeCode installed agent.

Widens the ``--effort`` enum to include ``xhigh`` and ``max`` — Harbor
0.4.0's adapter pins ``choices=["low", "medium", "high"]`` even though
Claude Code 2.1+ also accepts ``xhigh`` and ``max`` (verified against
``claude --help`` on 2.1.132).  Without this override Harbor's
``_coerce_value`` raises ``ValueError`` before ever invoking the CLI.

Remove this override once the upstream Harbor adapter is updated.
"""

import logging

from harbor.agents.installed.base import CliFlag
from harbor.agents.installed.claude_code import ClaudeCode
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from recovery_bench.agents.recovery_mixin import RecoveryMixin
from recovery_bench.utils import save_usage

logger = logging.getLogger(__name__)


# Widened choices for the --effort flag.  Mirrors `claude --help` output
# from Claude Code 2.1.132: low, medium, high, xhigh, max.
_EFFORT_CHOICES = ["low", "medium", "high", "xhigh", "max"]


def _patch_effort_flag(flags: list[CliFlag]) -> list[CliFlag]:
    """Return a copy of ``flags`` with reasoning_effort's choices widened."""
    out: list[CliFlag] = []
    for flag in flags:
        if flag.kwarg == "reasoning_effort":
            # CliFlag is a frozen dataclass; rebuild via its constructor.
            out.append(
                CliFlag(
                    kwarg=flag.kwarg,
                    cli=flag.cli,
                    type=flag.type,
                    choices=_EFFORT_CHOICES,
                    default=flag.default,
                    env_fallback=flag.env_fallback,
                    format=flag.format,
                )
            )
        else:
            out.append(flag)
    return out


class RecoveryClaudeCode(RecoveryMixin, ClaudeCode):
    """ClaudeCode agent extended with trajectory replay for recovery."""

    # Shadow the parent's CLI_FLAGS to widen --effort choices.  See
    # module docstring.
    CLI_FLAGS = _patch_effort_flag(ClaudeCode.CLI_FLAGS)

    def __init__(self, message_mode: str = "full", model_kwargs: dict = None, **kwargs):
        super().__init__(**(model_kwargs or {}), **kwargs)
        self._init_recovery(message_mode)

    @staticmethod
    def name() -> str:
        return "recovery-claude-code"

    async def setup(self, environment: BaseEnvironment) -> None:
        await super().setup(environment)
        await self._maybe_replay_exec(environment)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        recovery_instruction = await self._build_recovery_instruction(instruction)
        await super().run(recovery_instruction, environment, context)
        save_usage(self.logs_dir, context)
