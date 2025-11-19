import sys
import argparse
import os
import subprocess
import hashlib
from . import swe_utils
import shutil
import json
from datasets import load_dataset
from pathlib import Path
from datetime import datetime 
from swebench.harness.run_evaluation import main as run_swe_bench


P_NLP = "princeton-nlp"
RUNS_DIR = "swe-runs"
DATASET = "princeton-nlp/SWE-bench_Verified"

def get_dataset(swebench_data : Path):
    #Check if user inputted the full dataset path, concatenate full path otherwise
    if str(swebench_data)[:len(P_NLP)] != P_NLP:
        swebench_data = P_NLP / swebench_data

    #Try to load and and return dataset, otherwise print error message and return 1
    try:
        dataset = load_dataset(str(swebench_data))
        print(f"Sucessfully loaded dataset: {swebench_data}\n")
        return dataset
    except Exception as e:
        print(f"Failed to load dataset: {swebench_data}\n")
        return 1


def filter_easy_tasks(data) -> list:
    hard_inst = []
    for i in range(data.num_rows):
        task = data[i]
        if task["difficulty"] != "15 min - 1 hour":
            hard_inst.append(task)
    return hard_inst


def create_task_hash(task_description: str) -> str:
    hash = hashlib.sha256(task_description.encode("utf-8")).hexdigest()[:8]
    return hash


def get_runs_dir() -> Path:
    """
    This function assumes the trajectories will be written to a directory
    situated in the parent of the src directory named 'swe-runs'. The relative path
    from generate_traces.py script would be '../../swe-runs' where swe-runs
    is written to global variable RUNS_DIR
    """
    swe_recovery_dir = Path(__file__).parent.parent / RUNS_DIR
    return swe_recovery_dir

def get_swe_root_dir() -> Path:
    """
    This function returns the directory path to the recovery swe agent directory
    """
    swe_root_dir = Path(__file__).parent.parent
    return swe_root_dir


def gen_single_state(swe_task: dict, subset: str, model: str, runs_dir: Path):
    # Get instance id, used for running mini swe agent instance and required for testing harness
    inst_id = swe_task["instance_id"]
    cmd = [
        "mini-extra",
        "swebench-single",
        "--subset", subset,
        "--split", "test",
        "--model", model,
        "-i", inst_id
    ]
    
    # Create unique task identifier 
    task_hash = create_task_hash(swe_task["problem_statement"])
    print(f"Hash: {task_hash} for task {inst_id}")
    return 0


def gen_dirty_states(num_trajs: int, model: str, subset: str, workers: int):
    swe_root = get_swe_root_dir()
    print(f"Swe root dir: {swe_root}")
    cmd = [
        "mini-extra", "swebench",
        "--model", model,
        "--subset", subset,
        "--split", "test",
        "--workers", str(workers),
        "--slice", f"0:{num_trajs}",
        "--output", str(swe_root),
    ]
    cmd.append("--shuffle")
    result = subprocess.run(cmd)
    return result


def get_repos(data: dict) -> list:
    repos = set()
    flat_repos = []
    try:
        for row in data:
            repo = row["repo"]
            if row["repo"] not in repos:
                repos.add(repo)
                flat_repos.append(repo.replace("/", "__"))
        return flat_repos
    except Exception as e:
        print("Couldn't get the repos from the SWE Bench dataset")
        return 1

def get_miniswe_preds():
    """
    This routine assumes the predictions.json file returned by the minisweagent are currently 
    in swe_root
    """
    swe_root = get_swe_root_dir()
    predictions = swe_root / "preds.json"
    if not predictions.exists():
        print("Error: {predictions} does not exist")
        return 1
    try:
        with open(str(predictions), "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error: Couldn't read predictions json file from {predictions}")
        return 1

def run_preds(dataset: str, max_workers: int, run_id : str):
    """
    This routine will run the predictions from the weak mini swe agent
    to check if the mini swe agent failed
    """
    #Edit this to be a try block later for pull request
    cwd = get_swe_root_dir()
    preds = get_miniswe_preds()
    if preds == 1:
        return 1

    instances = []
    for inst, p in preds.items():
        inst_id = p["instance_id"]
        instances.append(inst_id)
        model = p["model_name_or_path"]
        patch = p["model_patch"]
        swe_utils.write_predictions(cwd, inst_id, model, patch, run_id)

    print("KOBE" * 15)
    print(cwd)
    
    predictions = cwd / "predictions" / (run_id + ".jsonl")
    return run_swe_bench(
                dataset_name = DATASET,
                split = "test",
                instance_ids=instances,
                predictions_path = str(predictions),
                max_workers = max_workers,
                force_rebuild = False,
                cache_level = "env",
                clean = True,
                open_file_limit = 4096,
                run_id = run_id,
                timeout = 1800,
                namespace = "swebench",
                rewrite_reports = False,
                modal = False,
                instance_image_tag= "latest",
                env_image_tag = "latest",
            )

def get_unresolved_tasks(model: str, run_id: str):
    """
    Routine that gets the unresolved tasks from swebench newly generated trajectories
    """
    root = Path(os.getcwd())
    #swe_root = get_swe_root_dir()

    model_safe = model.replace('/', '__')
    filename = f"{model_safe}.{run_id}.json"
    path = root / filename
    try:
        with open(str(path), "r") as f:
            data = json.load(f)
        return data["unresolved_ids"] + data["empty_patch_ids"]
    except Exception as e:
        print(f"Couldn't get unresolved files from {path}")
        return 1

def clean_gen_trajs(dirty_states: list, repos: list): 
    """
    Done quadratically to ensure we don't remove any source files pertaining to the project
    """
    swe_root = get_swe_root_dir()
    swe_runs = get_runs_dir()
    if not swe_root.exists():
        print("Error: swe root path does not exist")
        return 1
    if not swe_runs.exists():
        print("Error: swe runs path does not exist")
        return 1

    #Output directory for weak mini swe agent is assumed to by ../../__file__
    try:
        for p in swe_root.iterdir():
            p_str = str(p)   
            for repo in repos:
                if repo in p_str:
                    inst = p.name #the instance id essentially
                    if str(inst) in dirty_states:
                        shutil.move(p_str, str(swe_runs))
                    else:
                        shutil.rmtree(p)
        return 0
    except Exception as e:
        print("Error: Couldn't clean the generating trajcetories")
        return 1


#Before calling, need to check which trajectories failed
def organize_directory(
        swe_data: dict, 
        model: str,
        run_id: str
        ):
    swe_root = get_swe_root_dir()
    swe_runs = get_runs_dir()
    if not swe_root.exists():
        print("Error: swe root path does not exist")
        return 1
    if not swe_runs.exists():
        print("Error: swe runs path does not exist")
        return 1

    #First move trajectories that are unresolved to swe_runs directory, delete ones that are resolved
    dirty_states = get_unresolved_tasks(model, run_id)
    if dirty_states == 1:
        return 1

    repos = get_repos(swe_data)
    if repos == 1:
        return 1

    if clean_gen_trajs(dirty_states, repos):
        return 1
    return 0

def get_tsk(swe_data: dict, inst_id: str):
    for row in swe_data:
        if row["instance_id"] == inst_id:
            return row["problem_statement"]
    return 1


def write_cmds_and_tsk(swe_data: dict):
    swe_runs = get_runs_dir()
    if not swe_runs.exists():
        print("Error: swe runs path does not exist")
        return 1
    
    for p in swe_runs.iterdir():
        if (not p.is_dir()):
            continue
        for sub_p in p.iterdir():
            curr = str(sub_p).split(".")
            print(curr)
            if curr[-1] == "json" and "commands" not in curr[0]:
                traj = swe_utils.get_data(sub_p, "json")
                if traj == 1:
                    return 1
                cmds = swe_utils.get_commands(traj)
                if swe_utils.write_cmds(p, traj["instance_id"], cmds):
                    return 1

                try:                   
                    tsk = get_tsk(swe_data, traj["instance_id"])
                    if tsk == 1:
                        print("Error: Couldn't find task")
                        return 1
                    with open(str(p / "task.txt"), "w") as f:
                        f.write(tsk)
                except Exception as e:
                    print("Error: Couldn't write task")
                    return 1
    return 0


def pred_and_org(
        dataset: str,
        swe_data: dict, 
        model: str, 
        max_workers: int,
        run_id: str
        ):
    """
    A routine for making predictions off the generated trajectories, but for cleaning up
    and organizing trajectories
    """
    
    # Run the predictions
    result = run_preds(dataset, max_workers, run_id)
    if result == 1:
        return 1
    
    #Organize the directory, moving unsolved tasks into
    if organize_directory(swe_data, model, run_id):
        return 1
    
    #Write the commands
    if write_cmds_and_tsk(swe_data):
        return 1
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate dirty states for SWE Bench instances"
    )

    # SWE-Bench dataset name, concatenate the full path in get_dataset
    parser.add_argument(
        "--dataset",
        type = str,
        required = True,
        help = "SWEBench dataset",
    )
    
    parser.add_argument(
        "--num_trajs",
        type = int,
        default = 50,
        help = "Number of dirty states out of swebench dataset to generate",
    )

    parser.add_argument(
        "--model",
        type = str,
        required = True,
        help = "Mini swe agent compatabile model",
    )

    parser.add_argument(
        "--subset",
        type = str,
        required = True,
        help = "SWE Bench subset to use",
    )

    parser.add_argument(
        "--workers",
        type = int,
        default = 4,
        help = "Number of worker threads for parallel processing",
    )

    parser.add_argument(
        "--run_id",
        type = str,
        required = True,
        help = "Unique run id to be used for trajectory generation and organization",
    )

    parser.add_argument(
        "--clean",
        action = "store_true",
        help = "Clean working directory and dataset instead of generating trajectories",
    )

    args, unkown = parser.parse_known_args()    
    swebench_data = Path(args.dataset)
    dataset = get_dataset(swebench_data)
    if dataset == 1:
        return 1
    
    # Get test split from dataset
    test_data = dataset["test"]
    
    # Get parameters, may not need filt_data
    filt_data = filter_easy_tasks(test_data) 
    num_trajs = args.num_trajs
    model = args.model
    subset = args.subset
    workers = args.workers
    run_id = args.run_id
    
    if args.clean:
        return pred_and_org(str(swebench_data), test_data, model, workers, run_id)
    return gen_dirty_states(num_trajs, model, subset, workers)


if __name__ == "__main__":
    sys.exit(main())   
