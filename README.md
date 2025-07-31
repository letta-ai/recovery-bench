# Self-Correction Bench

A benchmarking framework for evaluating self-correction capabilities of language models. This repository provides tools to generate self-correction traces and run replay/recovery agents on terminal-based tasks.

## Quick Start

The repository provides two main functionalities through Python module commands:

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

## Main Components

### Core Functionality
- **Trace Generation**: Automatically generates initial traces, identifies failed tasks, and creates correction attempts
- **Replay Agent**: Runs collected traces with a recovery agent to attempt self-correction
- **Analysis Tools**: Compare results and analyze performance across different models

### File Structure
```
self-correction-bench/
├── self-correction/
│   ├── generate_traces.py    # Self-correction trace generation pipeline
│   ├── run_replay_agent.py   # Replay/recovery agent runner
│   ├── replay_agent.py       # Core replay agent implementation
│   ├── utils.py              # Consolidated utility functions
│   ├── analysis.py           # Results analysis tools
│   └── compare_results.py    # Results comparison tools
```

## Detailed Usage

### Generate Traces (Full Pipeline)

Generate complete self-correction traces for a model:

```bash
python3 -m self-correction.generate_traces anthropic/claude-3-5-haiku-20241022 \
    --dataset-version 0.2.15 \
    --min-episodes 10 \
    --n-concurrent 4 \
    --max-iterations 3
```

This command will:
1. Generate initial traces using terminal-bench
2. Identify unsolved tasks 
3. Run replay agent iterations to generate more traces that meet the minimum episode threshold (and still failed)
4. Collect all traces into a final directory

#### Options:
- `--dataset-version`: Dataset version to use for initial trace generation (default: 0.2.15)
- `--min-episodes`: Minimum episodes per task for filtering and collection (default: 10)
- `--n-concurrent`: Number of concurrent processes for both initial traces and replay iterations (default: 4)  
- `--max-iterations`: Maximum replay iterations (default: 3)
- `--run-initial`: Only run initial traces, skip replay iterations
- `--task-folder`: Path to task definitions folder

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
- `--task-folder`: Path to task definitions
- `--cleanup-container`: Clean up Docker containers before running

### Direct Function Usage

All functionality is available as Python functions in `utils.py`:

```python
from self_correction.utils import (
    get_unsolved_tasks,
    collect_traces, 
    reorganize_directories,
    run_replay_agent_tb
)

# Get unsolved task IDs
task_ids = get_unsolved_tasks("runs/my-run", min_episodes_desired=10)

# Collect traces from multiple runs
collect_traces([Path("runs/run1"), Path("runs/run2")], Path("runs/collected"))
```

## Requirements

- Python 3.8+
- terminal-bench framework
- Docker (for task execution)
- Required Python packages (install with `pip install -r requirements.txt`)

## Environment Setup

Set the task folder environment variable:
```bash
export TASK_FOLDER=/path/to/terminal-bench/tasks
```

## Examples

### Complete Workflow Example

1. **Generate traces for a model:**
   ```bash
   python3 -m self-correction.generate_traces anthropic/claude-3-5-haiku-20241022
   ```

2. **Run correction with a stronger model:**
   ```bash
   python3 -m self-correction.run_replay_agent \
       --trajectory-folder runs/claude-3-5-haiku-collected-20250730_123456 \
       --model-name anthropic/claude-sonnet-4-20250514
   ```

3. **Analyze results:**
   ```bash
   python3 -m self-correction.analysis --run-dirs runs/initial-run runs/corrected-run
   ```

### Batch Processing

Generate traces for multiple models:
```bash
for model in "anthropic/claude-3-5-haiku-20241022" "anthropic/claude-3-5-sonnet-20241022"; do
    python3 -m self-correction.generate_traces "$model"
done
```

### Additional Utility Commands

Get unsolved task IDs:
```bash
python3 -c "
from self_correction.utils import get_unsolved_tasks
tasks = get_unsolved_tasks('runs/my-run', print_output=True)
"
```

Reorganize directories by task hash:
```bash
python3 -c "
from self_correction.utils import reorganize_directories
reorganize_directories('runs/my-run', '/path/to/tasks')
"
```

Collect traces from multiple directories:
```bash
python3 -c "
from pathlib import Path
from self_correction.utils import collect_traces
collect_traces([Path('runs/run1'), Path('runs/run2')], Path('runs/collected'))
"
```

## Output Structure

Generated runs are stored in the `runs/` directory with timestamps:
```
runs/
├── initial-claude-3-5-haiku-20250730_123456/     # Initial traces
├── replay-claude-3-5-haiku-20250730_123456-iter1/ # First replay iteration  
├── claude-3-5-haiku-collected-20250730_123456/    # Final collected traces
└── sonnet-4-correction-1/                        # Correction results
```

## Contributing

1. Follow the existing code structure in `self-correction/`
2. Add new utilities to `utils.py` as functions
3. Update this README for any new functionality
4. Test with multiple models and task types

## Troubleshooting

- **Docker issues**: Use `--cleanup-container` flag to clean up containers
- **Task folder not found**: Set `TASK_FOLDER` environment variable or use `--task-folder` flag  
- **Memory issues**: Reduce `--n-concurrent` parameter
- **Import errors**: Ensure you're running from the repository root directory
- **Module not found**: Make sure you're using `python3 -m self-correction.module_name` format
