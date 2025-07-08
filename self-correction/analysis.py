import json
from pathlib import Path

import argparse


task_ids = [
    "blind-maze-explorer-algorithm",
    "cartpole-rl-training",
    "chem-property-targeting",
    "configure-git-webserver",
    "crack-7z-hash",
    "cross-entropy-method",
    "csv-to-parquet",
    "decommissioning-service-with-sensitive-data",
    "download-youtube",
    "extract-elf",
    "extract-safely",
    "fix-pandas-version",
    "form-filling",
    "git-workflow-hack",
    "grid-pattern-transform",
    "heterogeneous-dates",
    "hf-model-inference",
    "mlflow-register",
    "modernize-fortran-build",
    "new-encrypt-command",
    "openssl-selfsigned-cert",
    "polyglot-c-py",
    "prove-plus-comm",
    "pytorch-model-cli",
    "simple-web-scraper",
    "stable-parallel-kmeans",
    "swe-bench-astropy-1",
    "swe-bench-fsspec",
    "swe-bench-langcodes",
    "tmux-advanced-workflow",
    "vim-terminal-task" 
]


parser = argparse.ArgumentParser()
parser.add_argument("--trajectory_folder", type=str, required=True)
parser.add_argument("--experiment_folder", type=str, required=True)
args = parser.parse_args()

experiment_base = Path(args.experiment_folder)

# read experiment_base / results.json
with open(experiment_base / "results.json", "r") as f:
    results = json.load(f)

# get resolved ids
resolved_ids = results["resolved_ids"]

trajectory_base = Path(args.trajectory_folder)
trajectory_folder = None

experiment_results = []

for task_id in task_ids:
    # hash/task_name/task_name.1-of-1
    for path in trajectory_base.glob(f"*/{task_id}/*.1-of-1"):
        if path.is_dir():
            trajectory_folder = path
            break

    if trajectory_folder is None:
        raise FileNotFoundError("Could not find trajectory folder ending in .1-of-1")


    agent_logs_dir = trajectory_folder / "agent-logs"

    episode_dirs = []
    for item in agent_logs_dir.iterdir():
        if item.is_dir() and item.name.startswith("episode-"):
            try:
                episode_num = int(item.name.split("-")[1])
                episode_dirs.append((episode_num, item))
            except ValueError:
                continue

    episode_dirs.sort(key=lambda x: x[0])

    for path in experiment_base.glob(f"{task_id}/*.1-of-1*"):
        if path.is_dir():
            experiment_trajectory_folder = path / "agent-logs"
            break

    experiment_episode_dirs = []
    for item in experiment_trajectory_folder.iterdir():
        if item.is_dir() and item.name.startswith("episode-"):
            try:
                episode_num = int(item.name.split("-")[1])
                experiment_episode_dirs.append((episode_num, item))
            except ValueError:
                continue

    experiment_episode_dirs.sort(key=lambda x: x[0])

    # print task_name, and the number of episodes that were successful
    experiment_results.append([task_id, len(episode_dirs), len(experiment_episode_dirs), "resolved" if task_id in resolved_ids else "unresolved"])

# rank experiment_results by resolved, then by n_episodes
experiment_results.sort(key=lambda x: (x[3] == "resolved", x[1]))

# print experiment_results
print("Task ID".ljust(50), "steps (haiku)".ljust(20), "steps (sonnet)".ljust(20), "Resolved".ljust(20))
for result in experiment_results:
    print(f"{result[0]:<50} {result[1]:<20} {result[2]:<20} {result[3]:<20}")


