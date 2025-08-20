# Recovery Bench

A benchmarking framework for evaluating recovery-bench capabilities of language models. This repository provides tools to generate recovery-bench traces and run replay/recovery agents on terminal-based tasks.

## Quick Start

### 1. Generate Recover Bench Traces
```bash
python3 -m recovery-bench.generate_traces openai/gpt-4o-mini
```

### 2. Run Replay/Recovery Agent
```bash
python3 -m recovery-bench.run_replay_agent \
    --trajectory-folder runs/gpt-4o-mini-collected-20250714_232243 \
    --model-name anthropic/claude-sonnet-4-20250514
```

## Usage

### Generate Traces

Generate complete recovery-bench traces for a model:

```bash
python3 -m recovery-bench.generate_traces openai/gpt-4o-mini \
    --dataset-version 0.2.15 \
    --min-episodes 10 \
    --n-concurrent 4 \
    --max-iterations 3
```

Key options:
- `--dataset-version`: Dataset version (default: 0.2.15)
- `--min-episodes`: Minimum episodes per task (default: 10)
- `--n-concurrent`: Number of concurrent processes (default: 4)
- `--max-iterations`: Maximum replay iterations (default: 3)
- `--run-initial`: Only run initial traces, skip replay iterations
- `--task-folder`: Path to task definitions folder (default: ./terminal-bench/tasks)

### Run Replay Agent

Run the replay/recovery agent on collected traces:

```bash
python3 -m recovery-bench.run_replay_agent \
    --trajectory-folder runs/gpt-4o-mini-collected-20250714_232243 \
    --model-name anthropic/claude-sonnet-4-20250514 \
    --run-id sonnet-correction-1 \
    --n-concurrent 4
```

#### Options:
- `--trajectory-folder`: Path to the trajectory folder (required)
- `--model-name`: Model name to use (required)
- `--run-id`: Custom run identifier
- `--n-concurrent`: Number of concurrent processes
- `--task-folder`: Path to task definitions folder (default: ./terminal-bench/tasks)
- `--cleanup-container`: Clean up Docker containers before running

TODO:
- [ ] Bumping up terminal-bench version
