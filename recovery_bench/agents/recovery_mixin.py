"""Shared recovery mixin for all recovery agents.

Provides the common trajectory parsing, message storage, and recovery
instruction building used by RecoveryTerminus, RecoveryLettaCode, and
RecoveryInstalledAgent.
"""

import os

from recovery_bench.prompts import build_message_context, build_recovery_instruction
from recovery_bench.replay import ReplayCommand, find_and_parse_trajectory

DEFAULT_TRAJECTORY_FOLDER = "./trajectories"


class RecoveryMixin:
    """Mixin providing shared recovery setup and instruction building.

    Subclasses should:
    1. Call ``_init_recovery(message_mode)`` during ``__init__``
    2. Call ``_parse_trajectory()`` during ``setup`` to get commands/messages
    3. Call ``await _build_recovery_instruction(instruction)`` during ``run``
    """

    _message_mode: str
    _trajectory_folder: str
    _replay_messages: list[dict]

    def _init_recovery(self, message_mode: str = "full") -> None:
        """Initialize common recovery state."""
        self._message_mode = message_mode
        self._trajectory_folder = os.getenv("TRAJECTORY_FOLDER", DEFAULT_TRAJECTORY_FOLDER)
        self._replay_messages: list[dict] = []

    def _parse_trajectory(self) -> tuple[list[ReplayCommand], list[dict]]:
        """Find and parse trajectory for the current task.

        Stores messages in ``_replay_messages`` for later use in
        ``_build_recovery_instruction``.

        Returns:
            (commands, messages) tuple.
        """
        commands, messages = find_and_parse_trajectory(self.logs_dir, self._trajectory_folder)
        self._replay_messages = messages
        return commands, messages

    async def _build_recovery_instruction(self, instruction: str) -> str:
        """Build a recovery-aware instruction using prior attempt context."""
        model = getattr(self, "model_name", "") or ""
        message_context = await build_message_context(
            self._replay_messages, self._message_mode, model
        )
        return build_recovery_instruction(instruction, message_context)
