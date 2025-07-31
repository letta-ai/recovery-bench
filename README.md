# Self-Correction Bench

A benchmarking framework for evaluating self-correction capabilities of language models. This repository provides tools to generate self-correction traces and run replay/recovery agents on terminal-based tasks.

## Quick Start

### 1. Generate Self-Correction Traces
```bash
python3 -m self-correction.generate_traces anthropic/claude-3-5-haiku-20241022
```

### 2. Run Replay/Recovery Agent
```bash
python3 -m self-correction.run_replay_agent \
    --trajectory-folder runs/collected-traces \
    --model-name anthropic/claude-sonnet-4-20250514
```

## Usage

### Generate Traces

Generate complete self-correction traces for a model:

```bash
python3 -m self-correction.generate_traces anthropic/claude-3-5-haiku-20241022 \
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
python3 -m self-correction.run_replay_agent \
    --trajectory-folder runs/collected-traces \
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