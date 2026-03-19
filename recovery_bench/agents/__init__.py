"""Recovery-Bench agents package.

Provides recovery agents, baseline agents, and the generic
RecoveryInstalledAgent wrapper for any Harbor installed agent.
"""

from recovery_bench.agents.base import RecoveryInstalledAgent, resolve_harbor_agent
from recovery_bench.agents.letta_code import LettaCode, RecoveryLettaCode
from recovery_bench.agents.terminus import (
    BaselineTerminus,
    RecoveryTerminus,
    RecoveryTerminusWithMessageSummaries,
    RecoveryTerminusWithoutMessages,
)

__all__ = [
    "BaselineTerminus",
    "LettaCode",
    "RecoveryInstalledAgent",
    "RecoveryLettaCode",
    "RecoveryTerminus",
    "RecoveryTerminusWithMessageSummaries",
    "RecoveryTerminusWithoutMessages",
    "resolve_harbor_agent",
]

# Maps friendly agent names to import paths for CLI resolution.
AGENT_REGISTRY: dict[str, str] = {
    # Recovery agents (replay + modified instruction)
    "recovery-terminus": "recovery_bench.agents.terminus:RecoveryTerminus",
    "recovery-terminus-no-messages": "recovery_bench.agents.terminus:RecoveryTerminusWithoutMessages",
    "recovery-terminus-summaries": "recovery_bench.agents.terminus:RecoveryTerminusWithMessageSummaries",
    "recovery-letta-code": "recovery_bench.agents.letta_code:RecoveryLettaCode",
    # Baseline agents (fresh start, no replay)
    "baseline-terminus": "recovery_bench.agents.terminus:BaselineTerminus",
    # Initial agents
    "letta-code": "recovery_bench.agents.letta_code:LettaCode",
}
