"""Recovery agents package.

Each module provides a recovery variant of a specific agent harness,
extending it with trajectory replay and recovery-aware prompting via
RecoveryMixin.
"""

from recovery_bench.agents.recovery.claude_code import RecoveryClaudeCode
from recovery_bench.agents.recovery.codex import RecoveryCodex
from recovery_bench.agents.recovery.gemini_cli import RecoveryGeminiCli
from recovery_bench.agents.recovery.letta_code import RecoveryLettaCode
from recovery_bench.agents.recovery.terminus import RecoveryTerminus

__all__ = [
    "RecoveryClaudeCode",
    "RecoveryCodex",
    "RecoveryGeminiCli",
    "RecoveryLettaCode",
    "RecoveryTerminus",
]

# Maps friendly agent names to import paths for CLI resolution.
RECOVERY_REGISTRY: dict[str, str] = {
    "recovery-terminus": "recovery_bench.agents.recovery.terminus:RecoveryTerminus",
    "recovery-letta-code": "recovery_bench.agents.recovery.letta_code:RecoveryLettaCode",
    "recovery-claude-code": "recovery_bench.agents.recovery.claude_code:RecoveryClaudeCode",
    "recovery-codex": "recovery_bench.agents.recovery.codex:RecoveryCodex",
    "recovery-gemini-cli": "recovery_bench.agents.recovery.gemini_cli:RecoveryGeminiCli",
}
