# Recovery-Bench-SWE

Recovery-Bench is a benchmark for evaluating the capability of LLM agents to recover from mistakes in the Software Engineering setting. This repository provides the tools to generate Recovery-Bench-SWE traces and run replay/recovery agents on software-engineering-based tasks.

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
  -- dataset` SWE-bench_Verified \
  -- num_trajs 50 \
  -- model openai/gpt-5-nano` \
  -- subset verified` \
  -- workers 1 \
  -- run_id 0.0.1
```

### Options:
- `--dataset`: SWEBench dataset to generate traces from (required)
- `--num_trajs`: number of trajectories to generate (optional, default = 50)
- `--run-id`: Custom run identifier (required)
- `--model`: LiteLLM compatible model (required)
- `--subset`: SWEBench subset to use 
- `--workers`: Number of concurrent processes for SWE-bench evaluation (optional, default = 4)
- `run_id` 0.0.2: Custom run identifer (required)

Additionally, in the case generate_traces can't pull docker images, clean the populated instances with the --clean flag.


### 2. Run SWE Recovery Agent
Use command from toplevel directory
```bash
python -m recovery-bench-swe.src.run_mini_swe_agent \
  --trajectory-folder recovery-bench/recovery-swe-agent/swe-runs/openai__gpt-5-nano \
  --model-name openai/gpt-5 \
  --run-id 0.0.3 \
  --recovery-mode environment_only
```

For avaliable models, see [mini-swe-agent models](https://mini-swe-agent.com/latest/models/quickstart/)

### Options:
- `--trajectory-folder`: Path to the trajectory folder (required)
- `--model-name`: Model name to use (required)
- `--run-id`: Custom run identifier (required)
- `--recovery-mode`: Recovery mode - options: "environment_only", "action_summary", "full_history" (optional, default = "full_history")
- `--max-workers`: Number of concurrent processes for SWE-bench evaluation (optional, default = 1)
- `--default-agent`: Toggle to run the DefaultAgent on Recovery-Bench dataset



