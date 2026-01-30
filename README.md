# Recovery-Bench

Recovery-Bench is a benchmark for evaluating the capability of LLM agents to recover from mistakes. This repository provides tools to generate Recovery-Bench traces and run replay/recovery agents on terminal-based tasks.

Built on [Harbor](https://github.com/laude-institute/harbor) and [Terminal-Bench 2.0](https://harborframework.com/docs/running-tbench).

Read more on our blog: [letta.com/blog/recovery-bench](https://www.letta.com/blog/recovery-bench)

## Prerequisites

1. Install Harbor:
```bash
pip install harbor
# or
uv tool install harbor
```

2. Set up your API keys:
```bash
export ANTHROPIC_API_KEY="your-key"
# or
export OPENAI_API_KEY="your-key"
```

3. Ensure Docker is installed and running.

## Quick Start

### 1. Generate Recovery-Bench Traces
```bash
python3 -m recovery_bench.generate_traces openai/gpt-4o-mini
```

### 2. Run Replay/Recovery Agent
```bash
python3 -m recovery_bench.run_replay_agent \
    --trajectory-folder runs/gpt-4o-mini-collected-TIMESTAMP \
    --model-name anthropic/claude-sonnet-4-20250514
```

## Usage

### Generate Traces

Generate complete recovery-bench traces for a model:

```bash
python3 -m recovery_bench.generate_traces openai/gpt-4o-mini \
    --dataset-version 2.0 \
    --min-episodes 10 \
    --n-concurrent 4 \
    --max-iterations 3
```

Key options:
- `--task-id`: Specific task ID(s) to run (can be specified multiple times)
- `--dataset-version`: Dataset version (default: 2.0)
- `--min-episodes`: Minimum episodes per task (default: 10)
- `--n-concurrent`: Number of concurrent processes (default: 4)
- `--max-iterations`: Maximum replay iterations (default: 3)
- `--run-initial`: Only run initial traces, skip replay iterations
- `--task-folder`: Path to task definitions folder

### Run Replay Agent

Run the replay/recovery agent on collected traces:

```bash
python3 -m recovery_bench.run_replay_agent \
    --trajectory-folder runs/collected-traces \
    --model-name anthropic/claude-sonnet-4-20250514 \
    --run-id recovery-test-1 \
    --n-concurrent 4
```

#### Options:
- `--trajectory-folder`: Path to the trajectory folder (required)
- `--model-name`: Model name to use (required)
- `--run-id`: Custom run identifier
- `--n-concurrent`: Number of concurrent processes
- `--cleanup-container`: Clean up Docker containers before running

### Agent Variants

Three replay agent variants are available:

1. **ReplayAgent** (default): Uses full message history from failed attempts
   ```bash
   --agent-import-path recovery_bench.replay_agent:ReplayAgent
   ```

2. **ReplayAgentWithoutMessages**: Only restores environment state, no message history
   ```bash
   --agent-import-path recovery_bench.replay_agent:ReplayAgentWithoutMessages
   ```

3. **ReplayAgentWithMessageSummaries**: Uses summarized message history
   ```bash
   --agent-import-path recovery_bench.replay_agent:ReplayAgentWithMessageSummaries
   ```

## How It Works

Recovery-Bench evaluates agents on their ability to recover from previous failures:

1. **Generate initial traces**: Run a weaker model (e.g., gpt-4o-mini) on Terminal-Bench tasks
2. **Collect failures**: Keep trajectories where the agent failed to complete the task
3. **Test recovery**: Run stronger models starting from the failed state
4. **Measure recovery capability**: Compare performance in recovery vs. fresh-start scenarios

Key insight: The best-performing models for recovery differ from the top performers on fresh tasks. For example, GPT-5 shows significant improvement in recovery rankings compared to fresh-state performance.

## Architecture

- **Harbor**: Modern evaluation framework for Terminal-Bench 2.0
- **Terminus-2**: Reference agent implementation with tmux-based terminal interaction
- **ATIF**: Agent Trajectory Interchange Format for standardized trajectory storage

## Citation

If you use Recovery-Bench in your research, please cite our blog post and the Terminal-Bench paper.
