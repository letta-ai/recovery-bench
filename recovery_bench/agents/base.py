"""Generic recovery wrapper for any Harbor installed agent.

RecoveryInstalledAgent uses dynamic subclassing via __new__ so that the
returned instance IS-A the wrapped agent class.  All Harbor lifecycle
methods (populate_context_post_run, SUPPORTS_ATIF, cleanup, etc.) are
inherited automatically — no manual delegation required.
"""

import logging
import os

from harbor.agents.factory import AgentFactory
from harbor.models.agent.name import AgentName

from recovery_bench.prompts import (
    build_recovery_instruction,
    format_messages_as_text,
    summarize_messages,
)
from recovery_bench.replay import (
    find_and_parse_trajectory,
    replay_via_exec,
)
from recovery_bench.utils import save_usage

logger = logging.getLogger(__name__)

# Cache dynamic classes so type(a) is type(b) for same wrapped agent.
_recovery_class_cache: dict[type, type] = {}


def resolve_harbor_agent(name: str) -> type:
    """Resolve a Harbor agent name (e.g. 'claude-code') to its class.

    Uses Harbor's AgentFactory registry which maps AgentName enum values
    to agent classes.

    Args:
        name: Harbor agent name (must match an AgentName enum value).

    Returns:
        The agent class (e.g. ClaudeCode).

    Raises:
        ValueError: If the name doesn't match any known Harbor agent.
    """
    try:
        agent_name = AgentName(name)
    except ValueError:
        valid = [e.value for e in AgentName]
        raise ValueError(f"Unknown Harbor agent: {name!r}. Valid names: {', '.join(valid)}")

    if agent_name not in AgentFactory._AGENT_MAP:
        raise ValueError(
            f"Harbor agent {name!r} is registered in AgentName but not in AgentFactory. "
            "This may be a Harbor version mismatch."
        )

    return AgentFactory._AGENT_MAP[agent_name]


class RecoveryInstalledAgent:
    """Generic recovery wrapper for any Harbor installed agent.

    Uses ``__new__`` to return a dynamic subclass of the wrapped agent.
    Harbor sees the result as a proper ``BaseInstalledAgent`` instance —
    all lifecycle methods are inherited automatically.

    Usage (via CLI)::

        --recovery-agent installed:claude-code
        --recovery-agent installed:codex
        --recovery-agent installed:gemini-cli

    Internally resolved to::

        harbor run --agent-import-path recovery_bench.agents.base:RecoveryInstalledAgent \\
                   --agent-kwarg wrapped_agent=claude-code
    """

    def __new__(cls, wrapped_agent: str, message_mode: str = "full", **kwargs):
        inner_cls = resolve_harbor_agent(wrapped_agent)

        if inner_cls not in _recovery_class_cache:

            class Recovery(inner_cls):
                _trajectory_folder = os.getenv("TRAJECTORY_FOLDER", "./trajectories")

                def __init__(self, _message_mode: str = "full", **kw):
                    super().__init__(**kw)
                    self._message_mode = _message_mode
                    self._replay_messages: list[dict] = []

                @staticmethod
                def name() -> str:
                    return f"recovery-{inner_cls.name()}"

                async def setup(self, environment):
                    await super().setup(environment)
                    commands, messages = find_and_parse_trajectory(
                        self.logs_dir, self._trajectory_folder
                    )
                    self._replay_messages = messages
                    if commands:
                        await replay_via_exec(environment, commands)
                        logger.info(f"Replayed {len(commands)} commands from previous trajectory")

                async def _build_message_context(self) -> str | None:
                    if self._message_mode == "none" or not self._replay_messages:
                        return None
                    if self._message_mode == "summary":
                        model = getattr(self, "model_name", "") or ""
                        return await summarize_messages(self._replay_messages, model)
                    return format_messages_as_text(self._replay_messages)

                async def run(self, instruction, environment, context):
                    message_context = await self._build_message_context()
                    await super().run(
                        build_recovery_instruction(instruction, message_context),
                        environment,
                        context,
                    )
                    # Ensure usage.json exists for pipeline aggregation
                    save_usage(self.logs_dir, context)

            Recovery.__name__ = f"Recovery{inner_cls.__name__}"
            Recovery.__qualname__ = f"Recovery{inner_cls.__name__}"
            _recovery_class_cache[inner_cls] = Recovery

        return _recovery_class_cache[inner_cls](_message_mode=message_mode, **kwargs)
