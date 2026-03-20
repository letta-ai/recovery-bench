"""Prompt constants and instruction builders for recovery-bench agents.

All prompt text lives here so every agent uses the same wording.
"""

import json
import logging

logger = logging.getLogger(__name__)

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


def build_recovery_instruction(
    instruction: str,
    message_context: str | None = None,
) -> str:
    """Wrap a task instruction with the standard recovery preamble.

    Args:
        instruction: The original task instruction.
        message_context: Optional formatted text from the previous attempt's
            conversation (full transcript or summary).  Inserted between the
            recovery preamble and the original task.
    """
    parts = [RECOVERY_PREAMBLE]
    if message_context:
        parts.append(f"--- PREVIOUS ATTEMPT CONTEXT ---\n{message_context}")
    parts.append(f"--- ORIGINAL TASK ---\n{instruction}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Message formatting and summarization
# ---------------------------------------------------------------------------

SUMMARIZE_MESSAGES_PROMPT = (
    "Please summarize the following conversation concisely, "
    "focusing on what was attempted and what went wrong:\n\n"
)

SUMMARY_MESSAGE_TEMPLATE = "Summary of previous attempts:\n{summary}"

SUMMARY_FALLBACK = "Previous attempts to complete this task failed."


def format_messages_as_text(messages: list[dict]) -> str:
    """Format trajectory messages as a readable text transcript."""
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        # Truncate very long messages to keep instruction manageable
        if len(content) > 2000:
            content = content[:2000] + "\n... [truncated]"
        lines.append(f"[{role}]: {content}")
    return "\n\n".join(lines)


async def summarize_messages(messages: list[dict], model: str) -> str:
    """Summarize trajectory messages using litellm.

    Args:
        messages: Conversation messages from the previous trajectory.
        model: LiteLLM model identifier (e.g. 'anthropic/claude-sonnet-4-5').

    Returns:
        Summary text, or SUMMARY_FALLBACK if summarization fails.
    """
    from litellm import acompletion

    prompt = SUMMARIZE_MESSAGES_PROMPT + json.dumps(messages, indent=2)
    try:
        response = await acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.warning(f"Failed to summarize messages: {e}")
        return SUMMARY_FALLBACK
