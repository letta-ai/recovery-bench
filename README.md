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
python -m recovery_bench.generate_traces anthropic/claude-haiku-4-5-20251001 \
    --recovery-model anthropic/claude-opus-4-5-20251101 \
    --task-id sqlite-db-truncate
```

### 2. Run Recovery on Existing Traces
```bash
python -m recovery_bench.run_recovery \
    --traces jobs/initial-haiku-xxx \
    --model anthropic/claude-opus-4-5-20251101
```

## Usage

### Generate Traces Pipeline

Run the full pipeline: initial traces → recovery:

```bash
python -m recovery_bench.generate_traces <initial-model> \
    --recovery-model <recovery-model> \
    --task-id <task-name>
```

Options:
- `--task-id`: Specific task ID(s) to run (can be repeated)
- `--recovery-model`: Model for recovery (defaults to initial model)
- `--recovery-agent`: Recovery agent import path
- `--initial-agent`: Initial agent import path
- `--n-concurrent`: Number of concurrent processes (default: 4)
- `--max-iterations`: Maximum recovery iterations (default: 3)
- `--run-initial`: Only run initial traces, skip recovery

### Run Recovery Separately

Run recovery on existing initial traces:

```bash
python -m recovery_bench.run_recovery \
    --traces <path-to-initial-traces> \
    --model <recovery-model>
```

Options:
- `--traces`: Path to initial traces folder (required)
- `--model`: Model for recovery (required)
- `--agent`: Recovery agent import path
- `--task-id`: Specific task ID(s) to recover
- `--job-name`: Custom job name for output
- `--n-concurrent`: Number of concurrent processes

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
python -m recovery_bench.run_recovery \
    --traces jobs/initial-haiku-xxx \
    --model-config configs/models/sonnet-46-max.json

# Baseline run (fresh start, no replay)
python -m recovery_bench.run_recovery \
    --traces jobs/initial-haiku-xxx \
    --model-config configs/models/sonnet-46-max.json \
    --agent recovery_bench.recovery_terminus:BaselineTerminus
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
python -m recovery_bench.generate_traces anthropic/claude-haiku-4-5-20251001 \
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
