# Recovery-Bench-SWE

Recovery-Bench-SWE is a benchmark for evaluating the capability of LLM agents to recover from mistakes in the Software Engineering setting. This repository provides the tools to generate Recovery-Bench-SWE traces and run replay/recovery agents on software-engineering-based tasks.

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

## Quick Start

### 1. Generate Recovery-Bench-SWE Traces

```bash
python -m recovery-bench-swe.src.generate_traces \
  --dataset SWE-bench_Verified \
  --num-trajs 50 \
  --model openai/gpt-5-nano \
  --subset verified \
  --max-workers 4  \
  --run-id 0.0.1
```

### Options:
- `--dataset`: SWEBench dataset to generate traces from (required)
- `--num_trajs`: number of trajectories to generate (optional, default = 50)
- `--model`: LiteLLM compatible model (required)
- `--subset`: SWEBench subset to use (required)
- `--max-workers`: Number of concurrent processes for SWE-bench evaluation (optional, default = 4)
- `--run_id`: Custom run identifer (required)
- `--clean` : Cleans recovery-bench-swe directory of trajectories in the case generation errors (optional, useful when generate_traces can't pull docker images)

### 2. Run SWE Recovery Agent
```bash
python -m recovery-bench-swe.src.run_swe_agent \
  --trajectory-folder recovery-bench-swe/swe-runs/openai__gpt-5-mini/20260216_014037 \
  --model-name openai/gpt-5 \
  --run-id 0.0.2 \
  --recovery-mode environment_only
```

For available models, see [mini-swe-agent models](https://mini-swe-agent.com/latest/models/quickstart/)

### Options:
- `--trajectory-folder`: Path to the trajectory folder (required)
- `--model`: Model name to use (required)
- `--run-id`: Custom run identifier (required)
- `--recovery-mode`: Recovery mode - options: `environment_only`, `action_summary`, `full_history` (optional, default = "full_history")
- `--max-workers`: Number of concurrent processes for SWE-bench evaluation (optional, default = 1)
- `--default-agent`: Toggle to run the DefaultAgent on Recovery-Bench-SWE dataset (optional, default = False)


