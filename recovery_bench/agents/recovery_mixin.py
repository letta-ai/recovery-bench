"""Shared recovery mixin for all recovery agents.

Provides the common trajectory parsing, message storage, and recovery
instruction building used by all agents in recovery_bench.agents.recovery.
"""

import logging
import os

from harbor.environments.base import BaseEnvironment

from recovery_bench.prompts import build_message_context, build_recovery_instruction
from recovery_bench.replay import (
    ReplayCommand,
    VerifierSignal,
    find_and_parse_trajectory,
    replay_via_exec,
    replay_via_tmux,
)

logger = logging.getLogger(__name__)

DEFAULT_TRAJECTORY_FOLDER = "./trajectories"

# Set of valid message modes.  ``"initial"`` turns the agent into a
# pure initial-run baseline: no trajectory replay, no recovery
# preamble, no prior message context.  Useful for head-to-head
# comparisons against the recovery variants.
VALID_MESSAGE_MODES = frozenset({"full", "none", "summary", "initial"})


class RecoveryMixin:
    """Mixin providing shared recovery setup and instruction building.

    Subclasses should:
    1. Call ``_init_recovery(message_mode)`` during ``__init__``
    2. Call ``await _maybe_replay_exec(environment)`` during ``setup``
       (or ``_maybe_replay_tmux(session)`` for tmux-based agents)
    3. Call ``await _build_recovery_instruction(instruction)`` during ``run``

    When ``message_mode == "initial"``, replay is skipped and the raw
    instruction is passed through unchanged — the agent behaves
    exactly like a fresh initial run.
    """

    _message_mode: str
    _trajectory_folder: str
    _replay_messages: list[dict]
    _verifier_signal: VerifierSignal | None

    def _init_recovery(self, message_mode: str = "full") -> None:
        """Initialize common recovery state."""
        if message_mode not in VALID_MESSAGE_MODES:
            raise ValueError(
                f"Invalid message_mode {message_mode!r}; "
                f"expected one of {sorted(VALID_MESSAGE_MODES)}"
            )
        self._message_mode = message_mode
        self._trajectory_folder = os.getenv("TRAJECTORY_FOLDER", DEFAULT_TRAJECTORY_FOLDER)
        self._replay_messages: list[dict] = []
        self._verifier_signal = None

    def is_initial_mode(self) -> bool:
        """True when this agent should behave as a pure initial-run baseline."""
        return self._message_mode == "initial"

    def _parse_trajectory(self) -> tuple[list[ReplayCommand], list[dict]]:
        """Find and parse trajectory for the current task.

        Stores messages in ``_replay_messages`` and the verifier signal
        in ``_verifier_signal`` for later use in
        ``_build_recovery_instruction``.

        Returns:
            (commands, messages) tuple.  The verifier signal is stored
            on ``self`` rather than returned to keep call-site changes
            minimal — replay only needs commands+messages.
        """
        commands, messages, verifier_signal = find_and_parse_trajectory(
            self.logs_dir, self._trajectory_folder
        )
        self._replay_messages = messages
        self._verifier_signal = verifier_signal
        return commands, messages

    async def _maybe_replay_exec(self, environment: BaseEnvironment) -> None:
        """Parse + replay the failed trajectory via ``environment.exec``.

        No-op when ``message_mode == "initial"``.  Used by every
        recovery agent except :class:`RecoveryTerminus` (which uses
        tmux).
        """
        if self.is_initial_mode():
            return
        commands, _ = self._parse_trajectory()
        if commands:
            await replay_via_exec(environment, commands)
            logger.info(f"Replayed {len(commands)} commands from previous trajectory")

    async def _maybe_replay_tmux(self, session) -> str:
        """Parse + replay the failed trajectory via tmux.

        Returns the captured terminal output for use in the recovery
        prompt's ``terminal_state`` field.  Returns ``""`` when
        ``message_mode == "initial"`` or no commands are found.
        """
        if self.is_initial_mode():
            return ""
        commands, _ = self._parse_trajectory()
        if not commands:
            logger.info("No commands found in trajectory, starting fresh")
            return ""
        output = await replay_via_tmux(session, commands)
        logger.info(f"Replayed {len(commands)} commands from previous trajectory")
        return output

    async def _build_recovery_instruction(self, instruction: str) -> str:
        """Build a recovery-aware instruction using prior attempt context.

        When ``message_mode == "initial"`` returns the raw instruction
        unchanged (no preamble, no prior context) — the agent runs as
        a fresh initial baseline.

        When ``message_mode == "summary"`` and ``self.logs_dir`` is
        available, the summarizer's prompt and response are persisted
        under that directory for post-hoc inspection.
        """
        if self.is_initial_mode():
            return instruction
        model = getattr(self, "model_name", "") or ""
        log_dir = getattr(self, "logs_dir", None)
        message_context = await build_message_context(
            self._replay_messages,
            self._message_mode,
            model,
            log_dir=log_dir,
            verifier_signal=self._verifier_signal,
        )
        return build_recovery_instruction(instruction, message_context)
