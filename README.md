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

To run the replay agent:
```
TRAJECTORY_FOLDER=runs/sample-claude-3-5-haiku tb run --local-registry-path=./registry.json \
    --dataset-name self-correction-bench \
    --dataset-version 0.0.1 \
    --agent-import-path self-correction.replay_agent:ReplayAgent \
    --model-name anthropic/claude-sonnet-4-20250514 --run-id=...
```