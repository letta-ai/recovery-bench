import sys
import argparse
import os
import subprocess
import shutil
import json
from datasets import load_dataset
from datetime import datetime
from pathlib import Path
from swebench.harness.run_evaluation import main as run_swe_bench
from .swe_utils import (get_runs_dir,
                        get_swe_root_dir,
                        get_result_file, 
                        get_results_dir,
                        predictions_dir, 
                        write_cmds,
                        write_predictions,
                        cat_dir_and_model,
                        cat_dir_and_time,
                        get_commands,
                        get_data,
                        move_result)

P_NLP = "princeton-nlp"

def get_dataset(swebench_data : Path):
    """
    Get's desired SWEBench dataset from user inputted SWE Bench 
    dataset specification

    Inputs:
        swebench_data (Path) - SWEBench dataset to generate traces from

    Returns: 
        dict - SWEBench dataset 
        1 - indicating unsuccessful load_dataset() call
    """
    #Check if user inputted the full dataset path, concatenate full path otherwise
    if str(swebench_data)[:len(P_NLP)] != P_NLP:
        swebench_data = P_NLP / swebench_data
    #Try to load and and return dataset, otherwise print error message and return 1
    try:
        dataset = load_dataset(str(swebench_data))
        print(f"Sucessfully loaded dataset: {swebench_data}\n")
        return swebench_data, dataset
    except Exception as e:
        print(f"Failed to load dataset: {swebench_data}\n")
        return 1


def gen_dirty_states(num_trajs: int, model: str, subset: str, workers: int):
    """
    Generates dirty state trajectories via command line 
    minisweagent call.

    Inputs:
        num_trajs (int) - number of trajectories to generate
        model (str) - model to generate trajectories
        subset (str) - SWEBench subset
        workers (int) - number of threads to parallelize evaluation

    Returns:
        CompleteProcess instance
    """
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
    """
    Gets repo instance ids from SWEBench dataset.

    Inputs:
        data (dict) - SWEBench dataset

    Returns:
        list - SWEBench task instance ids
    """
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
    This routine assumes the preds.json file returned by the minisweagent 
    is currently in the Recovery-Bench swe_root.

    Inputs:
        None

    Returns:
        data (dict) - on successful read of the preds.json file
        1 (int) - indicating unsuccessful read
    """
    swe_root = get_swe_root_dir()
    predictions = swe_root / "preds.json"
    if not predictions.exists():
        print(f"Error: {predictions} does not exist")
        return 1
    try:
        with open(str(predictions), "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error: Couldn't read predictions json file from {predictions}")
        return 1

def run_preds(max_workers: int, run_id : str, dataset_name: Path):
    """
    Runs the predictions from the weak mini swe agent checking 
    if the mini swe agent failed to apply a correcting patches.

    Inputs:
        max_workers (int) - number of threads to parallelize evaluation
        run_id (str) - user inputting unique run identifer
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
        write_predictions(cwd, inst_id, model, patch, run_id)

    pred_dir = predictions_dir(cwd)
    if not pred_dir.exists():
        pred_dir.mkdir(parents=True, exist_ok=True)
    
    predictions = pred_dir / (run_id + ".jsonl")
    return run_swe_bench(
                dataset_name = str(dataset_name),
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
                report_dir = str(cwd)
            )

def get_unresolved_tasks(model: str, run_id: str):
    """
    Routine that gets the unresolved tasks from SWEBench generated trajectories.

    Inputs:
        model (str) - model used to generate dirty states
        run_id (str) - user inputted unique run identifier

    Returns:
        list - failed tasks by 'model'
        1 (int) - unsuccessful path read
    """
    results_dir = get_results_dir()
    path = get_result_file(results_dir, model, run_id)
    try:
        with open(str(path), "r") as f:
            data = json.load(f)
        return data["unresolved_ids"] + data["empty_patch_ids"]
    except Exception as e:
        print(f"Couldn't get unresolved files from {path}")
        return 1

def clean_gen_trajs(dirty_states: list, repos: list, model: str, time: str): 
    """
    Moves trajectories that are unresolved to dirty state subdirectory
    based on the model used to generate instances.

    Inputs:
        dirty_states (list) - trajectory instances that failed to resolve task
        repos (list) - all repos names based on targeted SWEBench dataset
        model (str) - model used to generate trajectories (ex: openai/gpt-5-nano)

    Returns:
        0 (int) - successful trajectory moving
        1 (int) - unsuccessful trajectory moving
    """
    swe_root = get_swe_root_dir()
    swe_runs = cat_dir_and_time(cat_dir_and_model(get_runs_dir(), model), time)
    if not swe_root.exists():
        print("Error: swe root path does not exist")
        return 1
    if not swe_runs.exists():
        os.mkdir(str(swe_runs))

    #Output directory for weak minisweagent is assumed to by ../../generate_traces.py
    try:
        for p in swe_root.iterdir():
            p_str = str(p)   
            for repo in repos:
                if repo in p_str:
                    inst = p.name #the instance id
                    if str(inst) in dirty_states:
                        print(f"Moving {inst} to {swe_runs}")
                        shutil.move(p_str, str(swe_runs))
                        print("Successfully moved")
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
    run_id: str,
    time: str
) -> int:
    """
    Organizes swe-root directory by getting unresolved tasks
    generated by minisweagent, and moving them to the targetted 
    swe-runs subdirectory for the model used.

    Inputs:
        swe_data (dict) - SWEBench dataset loaded into memory
        model (str) - model used to generate trajectories
        run_id (str) - user inputted unique run identifier

    Returns:
        0 (int) - indicated successful organization
        1 (int) - indicated unsuccessful organization
    """

    swe_root, swe_runs  = get_swe_root_dir(), get_runs_dir()
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
    
    clean_failed = clean_gen_trajs(dirty_states, repos, model, time)
    if clean_failed:
        return 1
    
    return 0

def get_tsk(swe_data: dict, inst_id: str):
    """
    Get's the task description from desired instance id. Important
    for reconstructing dirty states to run the Recovery Agent.

    Inputs:
        swe_data (dict) - SWEBench dataset loaded into memory
        inst_id (str) - instance id associated with desired task to extract

    Returns:
        str - problem statement of task for minisweagent to complete
        1 (int) - indicates unsuccessful task extraction
    """
    for row in swe_data:
        if row["instance_id"] == inst_id:
            return row["problem_statement"]
    return 1


def write_cmds_and_tsk(swe_data: dict, model: str, time):
    """
    Assumes trajectories will be saved to get_runs_dir()/model.
    get_runs_dir() is designed to be the recovery-bench-swe directory.

    Parameters:
        swe_data (dict): test data from huggingface swebench verified
        model (str): Litellm model specification [openai/gpt-5-nano for example]

    Returns:
        0 if the routine completes without errors
        1 if routine errors
    """
    swe_runs = cat_dir_and_time(cat_dir_and_model(get_runs_dir(), model), time)
    if not swe_runs.exists():
        print("Error: swe runs path does not exist")
        return 1
    
    for p in swe_runs.iterdir():
        if (not p.is_dir()):
            continue
        for sub_p in p.iterdir():
            curr = str(sub_p).split(".")
            if curr[-1] == "json" and "commands" not in curr[0]:
                traj = get_data(sub_p, "json")
                if traj == 1:
                    return 1
                cmds = get_commands(traj)
                if write_cmds(p, traj["instance_id"], cmds):
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
    swe_data: dict, 
    model: str, 
    max_workers: int,
    run_id: str,
    dataset_name: str
) -> int:
    """
    A routine for making predictions off the generated trajectories, 
    cleaning up successful trajectories, then organizing and formatting
    trajectories for dirty state agent evaluation.

    Inputs:
        swe_data (dict) - SWEBench dataset loaded into memory
        model (str) - model used to generate dirty states
        max_workers (int) - number of threads to parallelize evalaution of weak agent trajectories
        run_ids (str) - user inputted unique run identifier

    Returns:
        0 (int) - indicates successful execution
        1 (int) - indicates unsuccessful execution
    """
    
    # Run the predictions
    result = run_preds(max_workers, run_id, dataset_name)
    if result == 1:
        return 1
    
    #Move the openai__gpt-5.nano.test.json to a recovery-bench-swe subdirectory here
    moved = move_result(model, run_id)
    if moved == 1:
        return 1
    
    time = str(datetime.now().strftime("%Y%m%d_%H%M%S"))
    # Organize the directory, moving unsolved tasks into
    if organize_directory(swe_data, model, run_id, time):
        return 1
    
    # Write the commands
    if write_cmds_and_tsk(swe_data, model, time):
        return 1
    return 0

def get_all_instances(dataset: dict):
    """
    Get's all SWEBench instance_ids from dataset.

    Inputs:
        dataset (dict) - SWEBench dataset loaded into memory

    Returns:
        List[str] - all SWEBench instance_ids
    """
    instances = []
    for row in dataset:
        instances.append(row['instance_id'])
    return instances


def clean_root(dataset: dict, dry_run: bool):
    """
    A routine to remove trajectories in the case that the swebench evaluator fails
    and get corrupted trajectories that get saved to the Recovery-Bench
    SWE Bench root. Toggle dry_run to see which files will get removed.

    Inputs:
        dataset (dict) - SWEBench dataset loaded into memory
        dry_run (bool) - flag to either print instance_ids that would be deleted or to delete them

    Returns:
        None
    """
    instances = set(get_all_instances(dataset))
    root = get_swe_root_dir()
    removed = []

    for d in root.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if any(inst_id in name for inst_id in instances):
            if dry_run:
                print(f"[DRY] would remove: {d}")
            else:
                shutil.rmtree(d, ignore_errors=True)
                removed.append(name)
                print(f"[DEL] {d}")
    print(removed)

def clean_preds_file():
    """
    Remove the preds.json file to prevent running evaluations on old predictions.
    """
    swe_root = get_swe_root_dir()
    preds_file = swe_root / "preds.json"
    
    if preds_file.exists():
        print(f"Removing old predictions file: {preds_file}")
        preds_file.unlink()
        print("Old preds.json removed\n")
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
        "--num-trajs",
        type = int,
        default = 1,
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
        "--max-workers",
        type = int,
        default = 1,
        help = "Number of worker threads for parallel processing",
    )

    parser.add_argument(
        "--run-id",
        type = str,
        required = True,
        help = "Unique run id to be used for trajectory generation and organization",
    )

    parser.add_argument(
        "--clean",
        action = "store_true",
        help = "Clean swe-root directory from corrupted trajectories"
    )

    args, unknown = parser.parse_known_args()    
    swebench_data = Path(args.dataset)
    dataset_name, dataset = get_dataset(swebench_data)
    if dataset == 1:
        return 1

    # Get test split from dataset
    test_data = dataset["test"]
    num_trajs = args.num_trajs
    model = args.model
    subset = args.subset
    workers = args.max_workers
    run_id = args.run_id

    clean_preds_file()
    if (get_result_file(get_results_dir(), model, run_id).exists()):
        print(f"Please input unique run_id ({run_id}) for model ({model})\n")
        return 1
    if args.clean:
        return clean_root(test_data, False)
    gen_dirty_states(num_trajs, model, subset, workers)
    return pred_and_org(test_data, model, workers, run_id, dataset_name)


if __name__ == "__main__":
    sys.exit(main())   