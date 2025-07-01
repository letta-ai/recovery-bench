# Hash Reorganization Script

Clean, minimal script to reorganize benchmark results by task description hash using canonical task definitions from terminal-bench.

## Requirements

- **tbench conda environment**: The script requires the `tbench` conda environment to access terminal-bench task definitions.

## Usage

```bash
# Activate the tbench environment
conda activate tbench

# Single model (dry run)
python hash_reorganize.py runs/sample-claude-3-5-haiku

# Single model (execute)
python hash_reorganize.py runs/sample-claude-3-5-haiku --execute
```

## What it does

Transforms structure from:
```
runs/model/task-name/...
```

To:
```
runs/model/hash123/task-name/...
```

Where `hash123` is an 8-character SHA256 hash of the canonical task instruction from `/home/kevinlin/terminal-bench/tasks/*/task.yaml`. 