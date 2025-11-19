# Recovery-SWE-Agent

Recovery-SWE-Agent is a wrapper for testing LLM agents' ability to recover from mistakes on SWE-bench tasks. This tool allows you to generate trajectories and run replay/recovery agents on terminal-based software engineering tasks with partial execution states.

## Setup

### Install SWE-bench
SWE-bench uses Docker for reproducible evaluations. If you don't have Docker installed, follow the [Docker setup guide](https://docs.docker.com/engine/install/) to install Docker on your machine.

```bash
git clone git@github.com:princeton-nlp/SWE-bench.git
cd SWE-bench
pip install -e .
```

### Install mini-swe-agent
Use pip to install mini-swe-agent in your current environment

```bash
pip install mini-swe-agent
```

### Install datasets
Use pip install datasets in your current environment

```bash
pip install datasets
```

## Quick Start

### Generating trajectories
Use command from toplevel to generate trajectories
```bash
python -m recovery-bench.recovery-swe-agent.src.generate_traces \
  --dataset SWE-bench_Verified \
  --num_trajs 10 \
  --model openai/gpt-5-nano \
  --subset verified \
  --workers 4 \
  --run_id 0.0.1 
```

After generating trajectories, add --clean to the command to parse for dirty states 
and organize the dataset for the recovery agent to be ran

```bash
python -m recovery-bench.recovery-swe-agent.src.generate_traces \
  --dataset SWE-bench_Verified \
  --num_trajs 10 \
  --model openai/gpt-5-nano \
  --subset verified \
  --workers 4 \
  --run_id 0.0.1 \
  --clean
```

### Options:
- `--dataset`: Name of SWE Bench dataset (required)
- `num_trajs`: Number of dirty states to generate (required)
- `--model`: Model to use (required)
- `--subset`: SWE Bench dataset subset (required)
- `--run-id`: Custom run identifier (required)
- `--workers`: Number of trajectories to run in parallel (optional, default = 1)
- `--clean`: Boolean flag to indicate cleaning/organizing of freshly generate trajectories


### Run Mini SWE Recovery Agent
Use command from toplevel directory
```bash
python -m recovery-bench.recovery-swe-agent.src.run_mini_swe_agent \
  --trajectory-folder recovery-bench/recovery-swe-agent/swe-runs \
  --model-name openai/gpt-5-mini \
  --run-id gpt-5-correction-1 \
  --recovery-mode environment_only
```

For avaliable models, see [mini-swe-agent models](https://mini-swe-agent.com/latest/models/quickstart/)

### Options:
- `--trajectory-folder`: Path to the trajectory folder (required)
- `--model-name`: Model name to use (required)
- `--run-id`: Custom run identifier (required)
- `--recovery-mode`: Recovery mode - options: "environment_only", "action_summary", "full_history" (optional, default = "full_history")
- `--max-workers`: Number of concurrent processes for SWE-bench evaluation (optional, default = 1)


## TODOs:
- [ ] Look into adversarial error injection after strong agent runs
- [ ] Add compatibility for other SWE Bench datasets
- [ ] Add comprehensive trajectory checks to ensure quality dirty states
