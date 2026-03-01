# Recovery-Bench

Recovery-Bench is a benchmark for evaluating the capability of LLM agents to recover from mistakes. This repository provides tools to generate Recovery-Bench traces and run recovery agents on terminal-based tasks.

Built on [Harbor](https://github.com/laude-institute/harbor) and [Terminal-Bench 2.0](https://harborframework.com/docs/running-tbench).

Read more on our blog: [letta.com/blog/recovery-bench](https://www.letta.com/blog/recovery-bench)

## Prerequisites

1. Install dependencies:
```bash
pip install -e .
```

2. Set up your API keys:
```bash
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
```

3. Ensure Docker is installed and running.

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
- `--recovery-agent`: Recovery agent import path
- `--initial-agent`: Initial agent import path
- `--n-concurrent`: Number of concurrent processes (default: 8)
- `--max-iterations`: Maximum recovery iterations (default: 1)
- `--run-initial`: Only run initial traces, skip recovery
- `--resume-initial`: Path to existing traces (skips initial generation)
- `--job-name`: Custom job name for recovery output

## Agents

### Terminus-2 (ATIF format)

Recovery agents extend Harbor's [Terminus-2](https://github.com/laude-institute/harbor/blob/main/src/harbor/agents/terminus_2/terminus_2.py), inheriting its LLM calling (with retry and exponential backoff), context summarization, response parsing, and ATIF trajectory format.

```bash
# Initial
--initial-agent terminus-2

# Recovery
--recovery-agent recovery_bench.recovery_terminus:RecoveryTerminus
```

Recovery variants:
- `RecoveryTerminus`: Full message history from failed attempt injected into context
- `RecoveryTerminusWithoutMessages`: Environment replay only, no prior messages
- `RecoveryTerminusWithMessageSummaries`: LLM-generated summary of prior messages

Baseline (no replay, fresh start on the same failed tasks):
- `BaselineTerminus`: Runs Terminus2 from scratch for comparison

```bash
# Recovery run
python -m recovery_bench.generate_traces \
    --recovery-model configs/models/sonnet-46-max.json \
    --resume-initial jobs/initial-haiku-xxx

# Baseline run (fresh start, no replay)
python -m recovery_bench.generate_traces \
    --recovery-model configs/models/sonnet-46-max.json \
    --resume-initial jobs/initial-haiku-xxx \
    --recovery-agent recovery_bench.recovery_terminus:BaselineTerminus
```

### LettaCode (events JSONL format)

For LettaCode agents that output `letta_events_*.jsonl`:

```bash
# Initial
--initial-agent recovery_bench.letta_code_agent:LettaCode

# Recovery
--recovery-agent recovery_bench.recovery_letta_code:RecoveryLettaCode
```

## Example

Test haiku failing, opus recovering:

```bash
python -m recovery_bench.generate_traces \
    --initial-model anthropic/claude-haiku-4-5-20251001 \
    --initial-agent recovery_bench.letta_code_agent:LettaCode \
    --recovery-agent recovery_bench.recovery_letta_code:RecoveryLettaCode \
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
