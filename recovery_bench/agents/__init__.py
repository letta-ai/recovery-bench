"""Recovery-Bench agents package.

Provides initial agents, baseline agents, and recovery agents for
evaluating LLM agent recovery capabilities.
"""

from recovery_bench.agents.letta_code import LettaCode
from recovery_bench.agents.recovery import (
    RECOVERY_REGISTRY,
    RecoveryClaudeCode,
    RecoveryCodex,
    RecoveryGeminiCli,
    RecoveryLettaCode,
    RecoveryTerminus,
)
from recovery_bench.agents.terminus import BaselineTerminus

__all__ = [
    "BaselineTerminus",
    "LettaCode",
    "RecoveryClaudeCode",
    "RecoveryCodex",
    "RecoveryGeminiCli",
    "RecoveryLettaCode",
    "RecoveryTerminus",
]

# Maps friendly agent names to import paths for CLI resolution.
AGENT_REGISTRY: dict[str, str] = {
    # Recovery agents (replay + modified instruction)
    **RECOVERY_REGISTRY,
    # Baseline agents (fresh start, no replay)
    "baseline-terminus": "recovery_bench.agents.terminus:BaselineTerminus",
    # Initial agents
    "letta-code": "recovery_bench.agents.letta_code:LettaCode",
}
