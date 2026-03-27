#!/usr/bin/env python3
"""
CLI entry point for the recovery-bench trace generation pipeline.

Usage:
    # Full pipeline: initial traces + recovery
    python -m recovery_bench.generate_traces \
        --initial-model anthropic/claude-haiku-4-5-20251001 \
        --recovery-model anthropic/claude-opus-4-5-20251101 \
        --task-id cancel-async-tasks

    # Recovery only on existing traces
    python -m recovery_bench.generate_traces \
        --recovery-model configs/terminus/opus-46-max.json \
        --resume-initial jobs/initial-haiku-xxx

    # Initial only (no recovery)
    python -m recovery_bench.generate_traces \
        --initial-model openai/gpt-4o-mini \
        --task-id constraints-scheduling
"""

import argparse
import logging
import sys

from dotenv import load_dotenv

from .pipeline import run_pipeline

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate traces pipeline")
    parser.add_argument(
        "--initial-model",
        type=str,
        default=None,
        help="Model name or JSON config for initial traces. Required unless --resume-initial is used.",
    )
    parser.add_argument(
        "--recovery-model",
        type=str,
        default=None,
        help="Model name or JSON config for recovery. Required for recovery; skips recovery if omitted.",
    )
    parser.add_argument(
        "--n-concurrent", type=int, default=8, help="Number of concurrent processes"
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default="2.0",
        help="Dataset version for initial trace generation (e.g., 2.0)",
    )
    parser.add_argument(
        "--resume-initial",
        type=str,
        default=None,
        help="Path to existing initial traces (skips initial generation, runs recovery only)",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        action="append",
        dest="task_ids",
        help="Specific task ID(s) to run (can be specified multiple times)",
    )
    parser.add_argument(
        "--initial-agent",
        type=str,
        default=None,
        help="Agent for initial runs: built-in name (e.g., terminus-2) or import path. Defaults to terminus-2.",
    )
    parser.add_argument(
        "--recovery-agent",
        type=str,
        default="recovery-terminus",
        help=(
            "Recovery agent: registry name (e.g. recovery-terminus), "
            "import path (e.g. module:Class), or installed:<name> "
            "(e.g. installed:claude-code) to wrap a Harbor agent"
        ),
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default=None,
        help="Custom job name for recovery output (auto-generated if not specified)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Harbor sandbox backend (e.g., docker, daytona, modal)",
    )
    parser.add_argument(
        "--message-mode",
        type=str,
        choices=["full", "none", "summary"],
        default="full",
        help="How recovery agents use messages from previous trajectory (default: full)",
    )
    args = parser.parse_args()

    # Validate flags
    if args.resume_initial and args.initial_model:
        parser.error("--initial-model and --resume-initial are mutually exclusive")

    if not args.resume_initial and not args.initial_model:
        parser.error("--initial-model is required unless --resume-initial is used")

    if args.resume_initial and not args.recovery_model:
        parser.error("--recovery-model is required when using --resume-initial")

    return run_pipeline(
        initial_model=args.initial_model,
        recovery_model=args.recovery_model,
        resume_initial=args.resume_initial,
        initial_agent=args.initial_agent,
        recovery_agent=args.recovery_agent,
        task_ids=args.task_ids,
        n_concurrent=args.n_concurrent,
        dataset_version=args.dataset_version,
        job_name=args.job_name,
        harbor_env=args.env,
        message_mode=args.message_mode,
    )


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    sys.exit(main())
