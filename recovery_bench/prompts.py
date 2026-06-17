"""Prompt constants and instruction builders for recovery-bench agents.

All prompt text lives here so every agent uses the same wording.
"""

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from recovery_bench.replay import VerifierSignal

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

# Model used for the summary mode.  Hardcoded (not the agent's model) so
# that smaller / weaker recovery agents still get a high-quality
# reflection.  Adapted from
# letta-train/envs/memory_reflection/system_prompt.txt — Phase 1
# (transcript reading) and Phase 5 (commit/push) removed since this is a
# single LiteLLM call that returns text, not a memory subagent.
SUMMARIZER_MODEL = "anthropic/claude-opus-4-7"

# Reasoning effort for the summarizer.  Without an explicit thinking
# config Opus 4.7 runs with thinking OFF (silent regression from 4.6)
# — see
# https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-7
#
# We bypass LiteLLM's ``reasoning_effort`` mapping and pass the native
# Anthropic ``thinking`` + ``output_config`` shape directly: the
# LiteLLM version pinned by Harbor (1.81.x) maps ``reasoning_effort``
# for Opus 4.7 to the now-rejected ``thinking.type="enabled"`` shape.
#
# We use "high" rather than the Anthropic-recommended "xhigh" because
# LiteLLM 1.81.x's ``output_config.effort`` allowlist is
# ``{high, medium, low, max}`` only.  Bump to "xhigh" once Harbor
# moves to LiteLLM >=1.83.9.
SUMMARIZER_REASONING_EFFORT = "high"

SUMMARIZE_MESSAGES_PROMPT = """\
You are a reflection agent reviewing a FAILED attempt at a \
task, so your job is to surface what went wrong. You run autonomously and return a single \
reflection report. You CANNOT ask questions — make reasonable assumptions \
based on context.

**The previous attempt FAILED verification.** Even if the primary agent \
claims success at the end of the transcript (e.g. "task_complete: true", \
"the test passes", "done"), those claims are WRONG — the verifier rejected \
the result. Treat the agent's self-assessment as suspect, especially any \
final-state claims, and look for what they got wrong.

**You are NOT the primary agent.** You are reviewing a conversation that already \
happened:
- "system" messages are the primary agent's system prompt — use them only to \
understand the agent's identity and what it was trying to do.
- "assistant" messages (including ones with tool calls) are from the primary agent.
- "user" messages are from the primary agent's user.

## Your Job

Produce a reflection that captures what was attempted and what went \
wrong. Follow the phases below in order.

### Phase 1 — Extract

Identify candidate observations worth surfacing. Prioritize in this order:

1. **Mistakes and corrections** — errors made, dead ends hit, failed \
   retries, misread requirements
2. **State changes** — what was changed, created, deleted, or left in a \
   partially completed state during the conversation
3. **Approaches tried** — strategies, tools, and command patterns that were \
   attempted (and whether they worked, partially worked, or failed)
4. **Constraints discovered** — non-obvious requirements, gotchas, or \
   environment quirks the primary agent learned the hard way

For each candidate, apply these filters:

- **Durable or ephemeral?** Specific timestamps, exact intermediate variable \
  values, or one-off scratch paths are ephemeral. Distill the underlying \
  pattern instead.
- **Generalizable?** Prefer reusable patterns over event transcripts. \
  "The verifier requires output on stdout, not a file" is durable. \
  "Ran `python solve.py` at step 14" is not. The raw conversation is \
  already available — don't re-record it.

If nothing survives filtering, say so explicitly in the report.

### Phase 2 — Reflect

For each observation that survived Phase 1, write a surgical, well-placed \
note. Be concrete: name the specific commands, files, or assumptions \
involved.

**Contradiction handling**: If the primary agent contradicted itself \
mid-conversation (e.g., changed strategies, abandoned an approach), surface \
the final decision and the reason for it.

### Phase 3 — Review

Quick sanity pass before returning the reflection:

- **Stale content**: Did anything in your reflection get superseded by a later \
  step in the conversation? Remove or update it.
- **Signal check**: Is anything you wrote trivially obvious or low-signal? \
  Drop it.

## Output Format

Return a reflection with:

1. **What was attempted** — 1-3 sentences on the high-level approach the \
   primary agent took.
2. **What went wrong** — concrete failures, errors, dead ends, with the \
   specific commands/files involved.
3. **State at end of conversation** — what was created, modified, or left \
   behind.
4. **Observations and recommendations** — concrete takeaways. Avoid vague \
   advice like "be more careful"; prefer specific notes ("use ffmpeg \
   `-c:v copy` instead of re-encoding").
5. **Skipped** — anything you considered surfacing but decided against, with \
   a one-line reason.

## Critical Reminders

1. **Not the primary agent** — Don't respond as if you're continuing the work.
2. **Be selective** — Few high-signal observations > many trivial ones.
3. **No relative dates** — Use absolute dates if dates matter.
4. **Concrete, not abstract** — Name files, commands, error strings.

---

"""

SUMMARY_FALLBACK = "Previous attempts to complete this task failed."


def format_messages_as_text(messages: list[dict]) -> str:
    """Format trajectory messages as a readable text transcript."""
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        lines.append(f"[{role}]: {content}")
    return "\n\n".join(lines)


def format_verifier_signal(signal: "VerifierSignal | None") -> str | None:
    """Format a :class:`~recovery_bench.replay.VerifierSignal` as a markdown
    block for inclusion in the summarizer prompt.

    Returns ``None`` when the signal carries no useful information.
    The output deliberately leads with ``# VERIFIER OUTPUT`` so the
    summarizer can find it without parsing — it's prepended *before*
    the conversation transcript.
    """
    if signal is None:
        return None

    lines: list[str] = ["# VERIFIER OUTPUT (ground-truth failure signature)", ""]

    if signal.reward is not None:
        verdict = "PASS" if signal.reward > 0 else "FAIL"
        lines.append(f"**Reward**: {signal.reward} ({verdict})")
        lines.append("")

    if signal.test_summary:
        s = signal.test_summary
        parts = []
        for key in ("tests", "passed", "failed", "skipped", "pending"):
            if key in s:
                parts.append(f"{key}={s[key]}")
        if parts:
            lines.append(f"**Test summary**: {', '.join(parts)}")
            lines.append("")

    if signal.failed_tests:
        lines.append(f"**Failed tests** ({len(signal.failed_tests)}):")
        lines.append("")
        for t in signal.failed_tests:
            lines.append(f"### `{t.name}`")
            lines.append("```")
            lines.append(t.trace.rstrip())
            lines.append("```")
            lines.append("")
    elif signal.raw_tail:
        lines.append("**Verifier stdout (tail)**:")
        lines.append("")
        lines.append("```")
        lines.append(signal.raw_tail.rstrip())
        lines.append("```")
        lines.append("")

    body = "\n".join(lines).rstrip() + "\n"

    # Defense-in-depth cap (matches replay._TOTAL_SIGNAL_CAP semantics).
    if len(body) > 10_000:
        body = body[:10_000] + "\n...[verifier signal truncated]\n"

    return body


async def build_message_context(
    messages: list[dict],
    message_mode: str,
    model: str,
    log_dir: str | os.PathLike | None = None,
    verifier_signal: "VerifierSignal | None" = None,
) -> str | None:
    """Build message context text from trajectory messages.

    Args:
        messages: Conversation messages from the previous trajectory.
        message_mode: One of ``"full"``, ``"none"``, ``"summary"``.
        model: LiteLLM model identifier for the *agent*. Ignored by the
            summarizer, which uses :data:`SUMMARIZER_MODEL` regardless of
            the agent's model so weaker recovery agents still get a
            high-quality reflection.
        log_dir: Optional directory for persisting summarizer artifacts
            (the prompt sent to the summarizer and its raw response).
            When provided and ``message_mode == "summary"``, writes
            ``summarizer_input.txt`` and ``summarizer_output.md`` under
            ``log_dir`` for post-hoc inspection.
        verifier_signal: Optional ground-truth failure signature from the
            failed initial run's verifier (test results, assertion text).
            Only used in ``"summary"`` mode — fed to the summarizer so it
            can ground its reflection in test-level truth instead of the
            failed agent's self-narration.

    Returns:
        Formatted text for ``full``, summarized text for ``summary``,
        or ``None`` for ``none`` / empty messages.
    """
    if message_mode == "none" or not messages:
        return None
    if message_mode == "summary":
        return await summarize_messages(
            messages, log_dir=log_dir, verifier_signal=verifier_signal
        )
    # full
    return format_messages_as_text(messages)


async def summarize_messages(
    messages: list[dict],
    model: str = SUMMARIZER_MODEL,
    log_dir: str | os.PathLike | None = None,
    verifier_signal: "VerifierSignal | None" = None,
) -> str:
    """Summarize trajectory messages using litellm.

    Always uses :data:`SUMMARIZER_MODEL` by default — the summarizer
    model is intentionally decoupled from the recovery agent's model so
    a small / weak agent still receives a high-quality reflection.

    Args:
        messages: Conversation messages from the previous trajectory.
        model: LiteLLM model identifier. Defaults to
            :data:`SUMMARIZER_MODEL`.
        log_dir: Optional directory for persisting summarizer artifacts
            (the prompt sent to the summarizer and its raw response).
            When provided, writes ``summarizer_input.txt`` and
            ``summarizer_output.md`` for post-hoc inspection.
        verifier_signal: Optional ground-truth failure signature from
            the failed initial run's verifier.  When present, prepended
            to the conversation transcript so the summarizer sees what
            the *verifier* observed (often more accurate than the
            failed agent's own self-narration).  Also persisted as
            ``verifier_signal.txt`` under ``log_dir``.

    Returns:
        Summary text, or SUMMARY_FALLBACK if summarization fails.
    """
    from litellm import acompletion

    verifier_block = format_verifier_signal(verifier_signal)
    transcript_section = (
        "# CONVERSATION TRANSCRIPT\n\n"
        "The conversation follows as a JSON array of messages.\n\n"
        + json.dumps(messages, indent=2)
    )
    prompt = SUMMARIZE_MESSAGES_PROMPT
    if verifier_block is not None:
        prompt += verifier_block + "\n---\n\n"
    prompt += transcript_section

    if log_dir is not None:
        try:
            ld = Path(log_dir)
            ld.mkdir(parents=True, exist_ok=True)
            (ld / "summarizer_input.txt").write_text(prompt)
            if verifier_block is not None:
                (ld / "verifier_signal.txt").write_text(verifier_block)
        except Exception as e:
            logger.warning(f"Failed to persist summarizer_input.txt: {e}")

    # Bypass LiteLLM's reasoning_effort mapping (broken on this older
    # LiteLLM for Opus 4.7) by passing the native Anthropic shape as
    # top-level kwargs — LiteLLM forwards unknown provider kwargs to
    # the API.  ``display="summarized"`` opts back into visible
    # reasoning content; Opus 4.7 omits it by default.
    try:
        response = await acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": SUMMARIZER_REASONING_EFFORT},
        )
        summary = response.choices[0].message.content
    except Exception as e:
        logger.warning(f"Failed to summarize messages: {e}")
        summary = SUMMARY_FALLBACK

    if log_dir is not None:
        try:
            ld = Path(log_dir)
            ld.mkdir(parents=True, exist_ok=True)
            header = (
                f"<!-- summarizer model={model} effort={SUMMARIZER_REASONING_EFFORT} "
                f"n_input_messages={len(messages)} -->\n\n"
            )
            (ld / "summarizer_output.md").write_text(header + (summary or ""))
        except Exception as e:
            logger.warning(f"Failed to persist summarizer_output.md: {e}")

    return summary
