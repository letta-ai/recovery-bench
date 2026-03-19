# Recovery-Bench

Recovery-Bench is a benchmark for evaluating the capability of LLM agents to recover from mistakes. This repository provides tools to generate Recovery-Bench traces and run recovery agents on terminal-based tasks.

Built on [Harbor](https://github.com/laude-institute/harbor) and [Terminal-Bench 2.0](https://harborframework.com/docs/running-tbench).

Read more on our blog: [letta.com/blog/recovery-bench](https://www.letta.com/blog/recovery-bench)

## Prerequisites

1. Install dependencies:
```bash
pip install -e .
```

2. Install [Git LFS](https://git-lfs.com/) and pull the initial traces:
```bash
git lfs install
git lfs pull
```
This fetches the initial haiku 4.5 traces in `runs/` used for recovery experiments.

3. Set up your API keys:
```bash
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
```

4. Ensure Docker is installed and running.

## Quick Start

### 1. Generate Initial Traces + Run Recovery
```bash
python -m recovery_bench.generate_traces \
    --initial-model anthropic/claude-haiku-4-5-20251001 \
    --recovery-model anthropic/claude-opus-4-5-20251101 \
    --task-id sqlite-db-truncate
```

### 2. Run Recovery on Existing Traces
```bash
python -m recovery_bench.generate_traces \
    --recovery-model anthropic/claude-opus-4-5-20251101 \
    --resume-initial jobs/initial-haiku-xxx
```

## Usage

Run the full pipeline (initial traces → recovery) or recovery only on existing traces:

```bash
# Full pipeline
python -m recovery_bench.generate_traces \
    --initial-model <model-or-config> \
    --recovery-model <model-or-config> \
    --task-id <task-name>

# Recovery only on existing traces
python -m recovery_bench.generate_traces \
    --recovery-model <model-or-config> \
    --resume-initial <path-to-initial-traces>

# Using a JSON model config
python -m recovery_bench.generate_traces \
    --recovery-model configs/models/opus-4.6-high.json \
    --resume-initial jobs/initial-haiku-xxx
```

Options:
- `--initial-model`: Model name or JSON config for initial traces (required unless `--resume-initial`)
- `--recovery-model`: Model name or JSON config for recovery (required for recovery, skips recovery if omitted)
- `--task-id`: Specific task ID(s) to run (can be repeated)
- `--recovery-agent`: Recovery agent name, import path, or `installed:<name>` (default: `recovery-terminus`)
- `--initial-agent`: Initial agent import path (default: `terminus-2`)
- `--n-concurrent`: Number of concurrent processes (default: 8)
- `--max-iterations`: Maximum recovery iterations (default: 1)
- `--resume-initial`: Path to existing traces (skips initial generation)
- `--job-name`: Custom job name for recovery output
- `--cleanup-container`: Cleanup Docker containers before running
- `--dataset-version`: Dataset version for initial trace generation (e.g., `2.0`)
- `--env`: Harbor sandbox backend (e.g., `docker`, `daytona`, `modal`)

## Agents

### Evaluating new models (no code)

Use `RecoveryTerminus` (default) with different models:

```bash
python -m recovery_bench.generate_traces \
    --recovery-model anthropic/claude-opus-4-5-20251101 \
    --resume-initial runs/initial-claude-haiku-4-5-20251001-20260303_194859
```

### Evaluating any Harbor installed agent (no code)

Wrap any Harbor installed agent with `installed:<name>`:

```bash
# Claude Code as recovery agent
python -m recovery_bench.generate_traces \
    --recovery-agent installed:claude-code \
    --recovery-model anthropic/claude-sonnet-4-6 \
    --resume-initial runs/initial-claude-haiku-4-5-20251001-20260303_194859

# Codex as recovery agent
python -m recovery_bench.generate_traces \
    --recovery-agent installed:codex \
    --recovery-model openai/gpt-5.3-codex \
    --resume-initial runs/initial-claude-haiku-4-5-20251001-20260303_194859

# Any Harbor agent: aider, gemini-cli, goose, openhands, etc.
python -m recovery_bench.generate_traces \
    --recovery-agent installed:gemini-cli \
    --recovery-model google/gemini-3.1-pro \
    --resume-initial runs/initial-claude-haiku-4-5-20251001-20260303_194859
```

### Terminus-2 (ATIF format)

Recovery agents extend Harbor's [Terminus-2](https://github.com/laude-institute/harbor/blob/main/src/harbor/agents/terminus_2/terminus_2.py), inheriting its LLM calling (with retry and exponential backoff), context summarization, response parsing, and ATIF trajectory format.

```bash
# Recovery (default)
--recovery-agent recovery-terminus

# Variants
--recovery-agent recovery-terminus-no-messages   # Environment replay only
--recovery-agent recovery-terminus-summaries      # Summarized history

# Baseline (fresh start, no replay)
--recovery-agent baseline-terminus
```

### LettaCode (events JSONL format)

```bash
# Initial
--initial-agent recovery_bench.agents.letta_code:LettaCode

# Recovery
--recovery-agent recovery-letta-code
```

## Example

Test haiku failing, opus recovering:

```bash
python -m recovery_bench.generate_traces \
    --initial-model anthropic/claude-haiku-4-5-20251001 \
    --initial-agent recovery_bench.agents.letta_code:LettaCode \
    --recovery-agent recovery-letta-code \
    --recovery-model anthropic/claude-opus-4-5-20251101 \
    --task-id sqlite-db-truncate \
    --max-iterations 1
```

## How It Works

1. **Generate initial traces**: Run initial model on Terminal-Bench tasks
2. **Collect failures**: Keep trajectories where the agent failed (reward=0)
3. **Replay operations**: Re-execute failed agent's commands to corrupt environment
4. **Test recovery**: Run recovery model starting from corrupted state
5. **Measure**: Compare recovery success rate across models

## Citation

If you use Recovery-Bench in your research, please cite our blog post and the Terminal-Bench paper.
