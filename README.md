# self-correction-bench

# TODOs:
- [x] Add a way to run a single task
- [x] Add a way to run a single task with customized agent
- [x] collect traces from the agent
- [x] add a way to replay the traces with docker 
- [ ] replay bootstrapped traces to lengthen the traces
- [ ] registry the traces inside registry.json

# Generate the initial traces

```
 time tb run \
    --dataset-name terminal-bench-core \
    --dataset-version head \
    --agent terminus \
    --model-name anthropic/claude-3-5-haiku-20241022 \
    --run-id sample-claude-3-5-haiku --n-concurrent 6
```

# Usage:
To reorganize the results by task description hash:
```
TASK_FOLDER=/home/kevinlin/terminal-bench/tasks python3 -m self-correction.hash_reorganize runs/sample-claude-3-5-haiku
```

To filter and register the tasks that are not solved:
```
python3 -m self-correction.register_unsolved_tasks --logs-dir runs/sample-claude-3-5-haiku --registry-path ./registry.json --max-episodes 10
```

To regenerate the traces for the unsolved tasks:
```
 TRAJECTORY_FOLDER=runs/sample-claude-3-5-haiku tb run \
    --dataset-name self-correction-bench \
    --dataset-version 0.0.1 \
    --agent-import-path self-correction.replay_agent:ReplayAgent \
    --model-name anthropic/claude-3-5-haiku-20241022 \
    --run-id=self-claude-3-5-fix-1 \
    --n-concurrent=4 \
```

To filter again:
```
python3 -m self-correction.register_unsolved_tasks --logs-dir runs/self-claude-3-5-fix-1 --max-episodes 10
```

To reorgnaize the traces by task description hash:
```
TASK_FOLDER=/home/kevinlin/terminal-bench/tasks python3 -m self-correction.hash_reorganize runs/self-claude-3-5-fix-1
```

To generate the traces yet again:
```
TRAJECTORY_FOLDER=runs/self-claude-3-5-fix-1 tb run --local-registry-path=./registry.json \
    --dataset-name self-correction-bench \
    --dataset-version 0.0.1 \
    --agent-import-path self-correction.replay_agent:ReplayAgent \
    --model-name anthropic/claude-3-5-haiku-20241022 \
    --run-id=self-claude-3-5-fix-2 \
    --n-concurrent=4 
```

To reorgnaize the traces by task description hash:
```
TASK_FOLDER=/home/kevinlin/terminal-bench/tasks python3 -m self-correction.hash_reorganize runs/self-claude-3-5-fix-2
```

To collect the traces from multiple trajectories (make sure that all runs are hashed-reorganized):
```
python3 -m self-correction.collect_traces --logs-dirs runs/sample-claude-3-5-haiku runs/self-claude-3-5-fix-1 runs/self-claude-3-5-fix-2 --output-dir runs/self-claude-3-5-collected
```

To run the collected traces:
```
python3 -m self-correction.run_replay_agent --trajectory-folder runs/self-claude-3-5-collected --model-name anthropic/claude-sonnet-4-20250514 --run-id sonnet-4-correction-1
```


To run the replay agent:
```
TRAJECTORY_FOLDER=runs/sample-claude-3-5-haiku tb run --local-registry-path=./registry.json \
    --dataset-name self-correction-bench \
    --dataset-version 0.0.1 \
    --agent-import-path self-correction.replay_agent:ReplayAgent \
    --model-name anthropic/claude-sonnet-4-20250514 --run-id=...
```

# New Task-Specific Commands

## Get List of Unsolved Task IDs

To get only the list of unsolved task IDs (without creating registry):
```
python3 -m self-correction.register_unsolved_tasks --logs-dir runs/self-claude-3-5-fix-1 --max-episodes 10 --list-only
```

## Run Replay Agent for Unsolved Tasks

To run the replay agent for all unsolved tasks in a trajectory folder:
```
python3 -m self-correction.run_replay_agent \
    --trajectory-folder runs/self-claude-3-5-collected \
    --model-name anthropic/claude-sonnet-4-20250514 \
    --run-id sonnet-4-correction-1 \
    --n-concurrent 4
```

This will automatically get all unsolved task IDs from the trajectory folder and run the replay agent for them.

## Run TB Command with Specific Task IDs

To run the tb command directly with specific task IDs:
```
TRAJECTORY_FOLDER=runs/self-claude-3-5-collected tb run \
    --dataset-name terminal-bench-core \
    --dataset-version head \
    --agent-import-path self-correction.replay_agent:ReplayAgent \
    --model-name anthropic/claude-sonnet-4-20250514 \
    --task-id taskid1 \
    --task-id taskid2 \
    --task-id taskid3 \
    --run-id sonnet-4-correction-1
```

## Automatic End-to-End Workflow

To automatically get unsolved task IDs and run the replay agent (integrated into run_replay_agent):

```
# Automatically get unsolved task IDs from trajectory folder and run replay agent
python3 -m self-correction.run_replay_agent \
    --trajectory-folder runs/self-claude-3-5-collected \
    --model-name anthropic/claude-sonnet-4-20250514 \
    --run-id sonnet-4-correction-1 \
    --n-concurrent 4
```
