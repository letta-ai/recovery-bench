"""Prompt constants and instruction builders for recovery-bench agents.

All prompt text lives here so every agent uses the same wording.
"""

# ---------------------------------------------------------------------------
# Initial-run prompts
# ---------------------------------------------------------------------------

LETTA_CODE_PROMPT_PREFIX = (
    "Complete the task. Do NOT ask clarification questions, you have "
    "enough information to complete the task. Make sure to finish the "
    "task to the best of your ability and do not stop at an intermediate step."
)

# ---------------------------------------------------------------------------
# Recovery prompts
# ---------------------------------------------------------------------------

RECOVERY_PREAMBLE = (
    "RECOVERY MODE: The previous attempt to complete this task failed. "
    "The environment has been restored to the state after the failed attempt. "
    "Please analyze what went wrong and try a DIFFERENT approach."
)


def build_recovery_instruction(instruction: str) -> str:
    """Wrap a task instruction with the standard recovery preamble."""
    return f"{RECOVERY_PREAMBLE}\n\n--- ORIGINAL TASK ---\n{instruction}"


# ---------------------------------------------------------------------------
# Message summarization prompts (used by RecoveryTerminusWithMessageSummaries)
# ---------------------------------------------------------------------------

SUMMARIZE_MESSAGES_PROMPT = (
    "Please summarize the following conversation concisely, "
    "focusing on what was attempted and what went wrong:\n\n"
)

SUMMARY_MESSAGE_TEMPLATE = "Summary of previous attempts:\n{summary}"

SUMMARY_FALLBACK = "Previous attempts to complete this task failed."
