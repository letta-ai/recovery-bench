"""Generic recovery wrapper for any Harbor installed agent.

RecoveryInstalledAgent uses dynamic subclassing via __new__ so that the
returned instance IS-A the wrapped agent class.  All Harbor lifecycle
methods (populate_context_post_run, SUPPORTS_ATIF, cleanup, etc.) are
inherited automatically — no manual delegation required.
"""

import logging
import tempfile
from pathlib import Path

from harbor.agents.factory import AgentFactory
from harbor.models.agent.name import AgentName

from recovery_bench.agents.recovery_mixin import RecoveryMixin
from recovery_bench.replay import replay_via_exec
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

            class Recovery(RecoveryMixin, inner_cls):
                def __init__(self, _message_mode: str = "full", **kw):
                    super().__init__(**kw)
                    self._init_recovery(_message_mode)

                @staticmethod
                def name() -> str:
                    return f"recovery-{inner_cls.name()}"

                async def setup(self, environment):
                    await super().setup(environment)
                    commands, _ = self._parse_trajectory()
                    if commands:
                        await replay_via_exec(environment, commands)
                        logger.info(f"Replayed {len(commands)} commands from previous trajectory")

                def create_run_agent_commands(self, instruction):
                    # Build commands with a short placeholder, then rewrite
                    # to pipe the instruction via stdin instead of as a
                    # positional arg (avoids Linux MAX_ARG_STRLEN of 128 KB).
                    _PH = "__RECOVERY_BENCH_INSTRUCTION__"
                    cmds = inner_cls.create_run_agent_commands(self, _PH)
                    for cmd in cmds:
                        if _PH not in cmd.command:
                            continue
                        # Remove the positional instruction argument
                        cmd.command = cmd.command.replace(f"-- {_PH}", "")
                        cmd.command = cmd.command.replace(_PH, "")
                        # Redirect stdin from the instruction file instead of /dev/null
                        for devnull in ("</dev/null", "< /dev/null"):
                            cmd.command = cmd.command.replace(devnull, f"< {self._instruction_file}")
                        # Fallback: if the agent doesn't redirect from /dev/null, pipe via cat
                        if self._instruction_file not in cmd.command:
                            cmd.command = f"cat {self._instruction_file} | {cmd.command}"
                    return cmds

                async def run(self, instruction, environment, context):
                    recovery_instruction = await self._build_recovery_instruction(instruction)

                    # Upload instruction to the container as a file;
                    # create_run_agent_commands will redirect stdin from it.
                    self._instruction_file = "/installed-agent/recovery-instruction.txt"
                    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
                        f.write(recovery_instruction)
                        local_path = f.name
                    try:
                        await environment.upload_file(local_path, self._instruction_file)
                    finally:
                        Path(local_path).unlink(missing_ok=True)

                    # create_run_agent_commands ignores the instruction arg
                    # (uses a placeholder), so pass the original short one.
                    await super().run(instruction, environment, context)
                    save_usage(self.logs_dir, context)

            Recovery.__name__ = f"Recovery{inner_cls.__name__}"
            Recovery.__qualname__ = f"Recovery{inner_cls.__name__}"
            _recovery_class_cache[inner_cls] = Recovery

        return _recovery_class_cache[inner_cls](_message_mode=message_mode, **kwargs)
