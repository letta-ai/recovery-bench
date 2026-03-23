# Recovery-Bench

Recovery-Bench is a benchmark for evaluating how well LLM agents recover from mistakes. An agent (with a weak model) attempts a terminal-based task and fails. Recovery-Bench replays the failed agent's commands to reproduce the corrupted environment, then drops a recovery agent into that state to see if it can fix the mess and complete the original task.

Built on [Harbor](https://github.com/laude-institute/harbor) and [Terminal-Bench 2.0](https://harborframework.com/docs/running-tbench). Read more on our [blog post](https://www.letta.com/blog/recovery-bench).

## How It Works

```
Initial agent runs task → fails → trajectory saved
                                        ↓
                          Replay failed commands in fresh env
                                        ↓
                          Recovery agent starts from corrupted state
                                        ↓
                          Measure: did it recover? (reward > 0)
```

1. **Initial traces** — An agent (e.g. Haiku) runs Terminal-Bench tasks. Some succeed, some fail.
2. **Filter failures** — Keep only trajectories where the agent failed (reward = 0).
3. **Replay** — Re-execute the failed agent's commands in a fresh Docker container to reproduce the corrupted state.
4. **Recovery** — A recovery agent gets the corrupted environment, context from the failed attempt, and the original task. It tries a different approach.
5. **Score** — Compare recovery success rates across models and agents.

## Setup

```bash
pip install -e .

# Pull the bundled initial traces (requires Git LFS)
git lfs install
git lfs pull

# API keys for the models you're testing
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
```

Docker must be running.

### Shared failure set

The `git lfs pull` fetches pre-generated Haiku 4.5 initial traces into `runs/`. These traces are the common baseline for all experiments — every model and agent is evaluated against the same set of failed tasks and corrupted environments, making results directly comparable across runs.

## Evaluating Models

Pick a model and run it against the shared failure set:

```bash
python -m recovery_bench.generate_traces \
    --recovery-model anthropic/claude-opus-4-5-20251101 \
    --resume-initial runs/initial-claude-haiku-4-5-20251001-20260303_194859
```

That's it. This replays each failed Haiku trajectory, runs Opus as the recovery agent, and scores the results. Swap `--recovery-model` to any [LiteLLM model](https://docs.litellm.ai/docs/providers):

```bash
--recovery-model openai/gpt-5.3-codex
--recovery-model google/gemini-3.1-pro
--recovery-model anthropic/claude-sonnet-4-6
```

For model-specific kwargs (reasoning effort, temperature, etc.), pass a JSON config instead:

```bash
--recovery-model configs/terminus/sonnet-46-max.json
```

```json
{
  "model": "anthropic/claude-sonnet-4-6",
  "model_kwargs": { "reasoning_effort": "max", "temperature": 1.0 }
}
```

### Message modes

`--message-mode` controls how much of the failed attempt the recovery agent sees:

| Mode | What the agent gets |
|------|---------------------|
| `full` (default) | Full transcript of the previous conversation |
| `none` | Nothing — only the replayed environment and original task |
| `summary` | LLM-generated summary of what was tried and what went wrong |

## Evaluating Agents

By default, recovery uses `RecoveryTerminus` (a Terminus-2 agent with replay and recovery instructions). You can swap the agent independently of the model.

### Built-in recovery agents

| Name | Description |
|------|-------------|
| `recovery-terminus` | Default. Terminus-2 + replay + recovery instruction |
| `recovery-letta-code` | LettaCode recovery agent |
| `baseline-terminus` | Fresh start, no replay (baseline comparison) |

```bash
python -m recovery_bench.generate_traces \
    --recovery-agent recovery-letta-code \
    --recovery-model anthropic/claude-sonnet-4-6 \
    --resume-initial runs/initial-claude-haiku-4-5-20251001-20260303_194859
```

### Any Harbor-installed agent

Use `installed:<name>` to wrap any Harbor agent for recovery. This dynamically creates a recovery variant that inherits the agent's full behavior and adds replay + recovery instructions:

```bash
python -m recovery_bench.generate_traces \
    --recovery-agent installed:claude-code \
    --recovery-model anthropic/claude-sonnet-4-6 \
    --resume-initial runs/initial-claude-haiku-4-5-20251001-20260303_194859
```

Works with any Harbor agent: `installed:codex`, `installed:gemini-cli`, `installed:aider`, etc.

---

## Advanced Usage

### Generating your own initial traces

Instead of using the bundled traces, generate fresh ones:

```bash
# Initial traces only
python -m recovery_bench.generate_traces \
    --initial-model anthropic/claude-haiku-4-5-20251001 \
    --task-id sqlite-db-truncate

# Full pipeline: initial + recovery in one command
python -m recovery_bench.generate_traces \
    --initial-model anthropic/claude-haiku-4-5-20251001 \
    --recovery-model anthropic/claude-opus-4-5-20251101 \
    --task-id sqlite-db-truncate
```

### Custom agents via import path

```bash
--recovery-agent my_module.agents:MyRecoveryAgent
--initial-agent my_module.agents:MyInitialAgent
```

### Full CLI reference

| Flag | Description | Default |
|------|-------------|---------|
| `--initial-model` | Model for initial traces | Required unless `--resume-initial` |
| `--recovery-model` | Model for recovery | Omit to skip recovery |
| `--resume-initial` | Path to existing initial traces | — |
| `--task-id` | Task ID (repeatable) | All tasks |
| `--recovery-agent` | Registry name, import path, or `installed:<name>` | `recovery-terminus` |
| `--initial-agent` | Registry name or import path | `terminus-2` |
| `--message-mode` | `full`, `none`, or `summary` | `full` |
| `--n-concurrent` | Parallel processes | `8` |
| `--job-name` | Custom output directory name | Auto-generated |
| `--dataset-version` | Terminal-Bench version | `2.0` |
| `--env` | Harbor backend (`docker`, `daytona`, `modal`) | — |

### Project structure

```
recovery_bench/
  generate_traces.py    CLI entry point
  pipeline.py           Orchestrator: initial → reorganize → recovery → aggregate
  prompts.py            Prompt text and instruction builders
  replay.py             Trajectory parsing and environment replay
  utils.py              Config resolution, task hashing, usage tracking
  agents/
    __init__.py         Agent registry
    base.py             RecoveryInstalledAgent (generic Harbor agent wrapper)
    recovery_mixin.py   Shared recovery logic across all recovery agents
    terminus.py         RecoveryTerminus, BaselineTerminus
    letta_code.py       LettaCode, RecoveryLettaCode
```

## Citation

If you use Recovery-Bench in your research, please cite our [blog post](https://www.letta.com/blog/recovery-bench) and the [Terminal-Bench paper](https://harborframework.com).
