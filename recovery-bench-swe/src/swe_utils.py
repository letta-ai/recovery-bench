"""
A script containing core utility files, logic, for running Recovery Agent
on a datset. Predictions and evaluation results paths are relative to
recovery-bench-swe directory
"""

import json
import re
import json
import shutil
import os
import traceback
from typing import List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.models import get_model
from minisweagent.run.utils.save import save_traj
from minisweagent.agents.default import DefaultAgent
from .swe_replay_agent import RecoverySWEAgent 
from swebench.harness.run_evaluation import main as run_swe_bench

JSON = "json"
JSONL = "jsonl"
TXT = "txt"
DATASET = "princeton-nlp/SWE-bench_Verified"
PREDICTIONS_PATH = Path("predictions")
EVAL_RESULTS_PATH = Path("evaluation-results")
END_CMD = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached"

def predictions_dir(swe_root: Path) -> Path:
    """
    Outputs the predictions directory for agents to
    write diffs.

    Inputs:
        swe_root (Path) - path to recovery bench swe root directory

    Returns:
        Path - path to predictions directory for agents to write to
    """
    return swe_root / "predictions"

def agent_runs_dir(swe_root: Path, model: str, run_id: str) -> Path:
    """
    Outputs the agent-runs directory for the given model 
    and run_id.

    Inputs:
        swe_root (Path) - Recovery-Bench swe root
        model (str) - Litellm model utilized for current agent run
        run_id (str) - user inputted unique run identifier

    Outputs:
        Path - path to the agent-runs subdirectory for the given model and run_id
    """
    return swe_root / f"agent-runs/{safe_model_name(model)}__{run_id}"

def get_commands(traj: dict) -> list: 
    """
    Parses commands from minisweagent trajectory and returns a
    list of the commands weak agent ran on docker image for
    the corresponding SWEBench task. Used as a helper to store
    commands in separate commands.json file.

    Starts parsing from index 2 as index 0, 1 of the "messages"
    array correspond to the system and instance prompts respectively.
    Index 2 is where the commands begin.

    Inputs:
        traj (dict) - weak agent trajectory for command parsing

    Outputs:
        list - containing trajectories 
    """
    commands = []
    messages = traj["messages"]
    for i in range(2, len(messages)):
        curr_response = messages[i]
        #Ensure only agent commands get saved
        if curr_response["role"] != "assistant":
            continue
        action = re.findall(r"```bash\s*\n(.*?)\n```", curr_response["content"], re.DOTALL)
        if len(action) == 1:
            commands.append(action[0].strip())

    # Ensure that the last command isn't a submission command
    if (len(commands) > 1) and commands[-1] == END_CMD:
        commands.pop()
    return commands

def write_cmds(inst_run: Path, inst: str, cmds: list):
    """
    Writes commands.json file to inst_run. Utilized in generate_traces.py

    Inputs:
        inst_run (Path) - path to desired subdirectory to write commands.json to
        inst (str) - instance id of SWEBench task
        cmds (list) - array of commands agent applied during task trajectory

    Returns:
        0 (int) - indicator of successful write
        1 (int) - indicator of unsuccessful write
    """
    to_write = {"instance": inst, "commands": cmds}
    try:
        with open(str(inst_run / "commands.json"), "w") as f:
            json.dump(to_write, f, indent=2)
        return 0
    except Exception as e:
        print(f"Error: Failed to write commands to {inst_run}")
        return 1

def get_data(path_to_file: Path, file_type: str):
    """
    Helper routine to read file data based on the file type

    Inputs:
        path_to_file (Path) - absolute path to a file
        file_type (str) - file format

    Returns:
        data - bytes read from file
    """
    if not path_to_file.exists():
        print(f"ERROR: Path does not exist: {path_to_file}")
        return 1
    try:
        with open(str(path_to_file), 'r') as f:
            if file_type == "json":
                data = json.load(f)
            else:
                data = f.read()
        return data
    except Exception as e:
        print(f"ERROR: Failed to open or read from file path: {str(path_to_file)}")
        return 1

def get_cfg(traj: dict) -> dict:
    """
    Gets the weak miniswe agent's configuration for injecting
    into Recovery or Default Agent's configuration.

    Inputs:
        traj (dict) - weak miniswe agent trajectory

    Returns:
        dict - weak miniswe agent configuration
    """
    if not traj:
        print("ERROR: cfg not recieved")
        return 1
    return traj["info"]["config"]

def make_model_cfg(model: str, *, temperature: float = 0.0, extra: dict | None = None) -> dict:
    """
    Creates LiteLLM model configuration for recovery or 
    default agent.

    Inputs:
        model (str) - user inputted LiteLLM compatible model
        temperature (float) - desired temperature
        extra (dict) - extra configuration parametrs if desired

    Returns:
        dict - LiteLLM model configuration for recovery or default agent
    """
    if not model:
        print("ERROR: Please enter a valid model")
        return 1
    mk = {"drop_params": True, "temperature": temperature}
    #Overide or add extra kwargs to mk
    if extra:
        mk |= extra
    return {"model_name": model, "model_kwargs": mk, "litellm_model_registry": None}

def get_message_history(traj: dict, recovery_mode: str) -> List[dict]:
    """
    Get's desired message history for inputting into Recovery Agent's
    context window based on recovery_mode. 

    Inputs:
        traj (dict) - dirty state trajectory
        recovery_mode (str) - desired recovery mode

    Returns:
        List[dict] - dirty state message history
    """
    if not traj:
        print("ERROR: Proper data wasn't inputted to get_message_history")
        return 1
    if recovery_mode == "environment_only":
        return []
    elif recovery_mode == "action_summary":
        #TODO use LLM to summarize actions
        return "It's over 9000"
    elif recovery_mode == "full_history":
        #Skip over system template and instance template, agent will re add during run with injected system template
        return traj['messages'][2:] if traj['messages'] else 1
    else:
        print(f"ERROR: Please enter valid recovery_mode, user entered {recovery_mode}")
        return 1
    
def create_summary(msgs: List[dict], recovery_mode: str):
    """
    Creates summary based on recovery_mode. If user desires to
    evalaute based on action_summary, summarize messages. Otherwise
    don't summarize.

    Inputs:
        msgs (List[dict]) - dirty state message history
        recovery_mode (str) - user inputted recovery_mode to evalaute

    Returns:
        str - summary or empty str dependent on recovery_mode
    """
    if recovery_mode == "action_summary":
    #TODO, have to architect summary based on messages
        return ""
    return ""
    
def inject_recovery(cfg: dict, recovery_mode: str, summary: str) -> None:
    """
    Injects recovery mode context into Recovery Agent's system template.

    Inputs:
        cfg (dict) - agent configuration to inject recovery into system prompt
        recovery_mode (str) - user inputted recovery_mode to evalaute
        summary (str) - summary dependent on recovery_mode

    Returns:
        None
    """
    if recovery_mode == "action_summary" and summary:
        cfg["agent"]["system_template"] = (
            "You are taking over a partially attempted SWE-bench task.\n"
            f"Prior attempt summary:\n{summary}\n---\n"
        ) + cfg["agent"]["system_template"]
    else:
        cfg["agent"]["system_template"] = (
            "You are taking over a partially attempted SWE-bench task. "
            "A previous agent tried to solve this but failed. "
            "The codebase may contain their incomplete or incorrect changes. "
            "The working tree may contain incomplete or incorrect changes.\n\n"
            ) + cfg["agent"]["system_template"]
        print(cfg["agent"]["system_template"])


#For turning trajectory config into the default agent config, can define a recovery agent config potentially
def get_agent_config(cfg: dict) -> dict:
    """
    Creates agent configuration for Recovery Agent using weak
    agent configuration parameters.

    Inputs:
        cfg (dict) - weak agent configuration
    
    Returns:
        dict - configuration to modify for Recovery Agent
    """
    if not cfg:
        print("ERROR: Please give get_agent_config proper config dictionary")
        return 1
    tmp = cfg["agent"]
    agent_cfg = {"system_template": tmp["system_template"],
                 "instance_template": tmp["instance_template"],
                 "timeout_template": tmp["timeout_template"],
                 "format_error_template": tmp["format_error_template"],
                 "action_observation_template": tmp["action_observation_template"],
                 "step_limit": tmp["step_limit"],
                 "cost_limit": tmp["cost_limit"]
                 }
    cfg["agent"] = agent_cfg
    return agent_cfg

def write_predictions(
    swe_root: Path,
    instance_id: str, 
    model: str, 
    patch: str,
    run_id: str
) -> int:
    """
    Writes the predictions (patch) the recovery or default agent wrote
    into the SWEBench evaluator format utilizing the run_id for the
    output file.

    Inputs:
        swe_root (Path) - Recovery-Bench swe directory
        instance_id (str) - SWEBench instance/task id
        model (str) - model name powering recovery/default agent used to generate patch
        path (str) - patch recovery/default agent submitted to fix SWEBench repo issue
        run_id (str) - user inputted run_id for recovery agent run management

    Returns:
        0 (int) - indicating successful write to output file
        1 (int) - indicating unsuccessful write to output file
    """
    to_write = {
                "instance_id": instance_id,
                "model_name_or_path": model,
                "model_patch": patch
                }
    
    predictions = predictions_dir(swe_root)
    if not predictions.exists():
        predictions.mkdir(parents=True, exist_ok=True)

    path_to_predictions = str(predictions / f"{run_id}.{JSONL}")
    try:
        with open(path_to_predictions, "a") as f:
            f.write(json.dumps(to_write) + '\n')
        return 0
    except Exception as e:
        print("ERROR: Couldn't write predictions to json")
        return 1
    
def safe_model_name(model: str) -> str:
    """
    Transforms LiteLLM model str into a safe
    format for creating unique run/directory identifiers
    for entire model name (openai__gpt-5-mini)

    Inputs:
        model (str) - LiteLLM model used for run (openai/gpt-5-mini)

    Returns: 
        str - safe format model (openai__gpt-5-mini)
    """
    model = model.replace("/", "__")             
    return re.sub(r"[^A-Za-z0-9._-]", "-", model)

def reorganize_trajectories(runs_dir: Path):
    """
    Reorganize trajectory files into subdirectories.
    
    Structure:
    runs_dir/
       run_name/
          instance_id_1/
            trajectory.json
            commands.json
           instance_id_2/
            trajectory.json
            commands.json

    Inputs:
        runs_dir (Path) - runs directory to regorganize trajectories 

    Returns:
        None
    """
    traj_files = list(runs_dir.glob("*.traj.json"))
    for traj_file in traj_files:
        # just get the instance id out of the traj file and open
        instance_id = traj_file.stem.replace(".traj", "")
        with open(traj_file, 'r') as f:
            traj_data = json.load(f)

        # get commands and create subdirectory based on instance id
        commands = get_commands(traj_data)
        instance_dir = runs_dir / instance_id
        instance_dir.mkdir(exist_ok=True)
        
        # move trajectory to new subdirectory and write a commands.json file 
        new_traj_path = instance_dir / traj_file.name
        shutil.move(str(traj_file), str(new_traj_path))
        commands_path = instance_dir / "commands.json" #can write instance_id to commands.json file as well (future update)
        with open(commands_path, 'w') as f:
            json.dump(commands, f, indent=2)
        print(f"Processed {instance_id}: {len(commands)} commands extracted")

def run_default_agent_swe(
        cwd: Path,
        trajectory_folder: Path, 
        model: str, 
        run_id: str, 
        max_workers: int = 1
    ) -> int:
    """
    Core algorithm for running the default swe agent

    Inputs:
        cwd (Path) - Recovery-Bench swe root
        trajectory_folder (Path) - path to dirty state trajectories
        model (str) - desired LiteLLM compatible model to benchmark
        run_id (str) - unique run identifier from user input
        max_workers (int) - number of workers to parallelize task executions

    Returns:
        0 (int) - indicates successful evalaution
        1 (int) - indicates unsuccessful evaluation
    """
    
    def _run_instance_from_commands(
            p: Path,
            *,
            lock: Lock,
            cwd: Path,
            model: str,
            run_id: str,
        ):
        docker_env, agent = None, None
        exit_status, result, extra_info = None, None, None
        instance_id = None
        try:
            commands_path = p / "commands.json"
            if not commands_path.exists():
                return None
            commands = get_data(commands_path, JSON)
            if commands == [] or not isinstance(commands, dict):
                return None
            instance_id = commands.get("instance")
            if not instance_id:
                return None
            traj_path = p / f"{instance_id}.traj.{JSON}"
            if not traj_path.exists():
                return None
            traj = get_data(traj_path, JSON)
            if traj == 1:
                return None

            cfg = get_cfg(traj)
            if cfg == 1:
                print("ERROR: Couldn't parse config from trajectory")
                return None
            model_cfg = make_model_cfg(model)
            if model_cfg == 1:
                print("ERROR: Please enter a supported LiteLLM model name")
                return None
            cfg["model"] = model_cfg
            agent_model = get_model(model, model_cfg)

            env = cfg["environment"]
            docker_env = DockerEnvironment(**env)

            _agent_cfg = get_agent_config(cfg)
            agent = DefaultAgent(
                model=agent_model,
                env=docker_env,
                **_agent_cfg)

            p_task = p / f"task.{TXT}"
            task = get_data(p_task, TXT)
            exit_status, result = agent.run(task)

            print("\n\n\n\n\n")
            print("Default Agent Returned Patch:")
            print(result)

            with lock:
                written = write_predictions(cwd, instance_id, model, result, run_id)
            if written:
                print(f"ERROR: Couldn't write predictions for instance {instance_id}")
                print("\n\n\n")
                return None
            return instance_id
        except Exception as e:
            exit_status, result = type(e).__name__, str(e)
            extra_info = {"traceback": traceback.format_exc()}
            print(f"ERROR processing instance: {e}")
            return None
        finally:
            if instance_id is not None:
                traj_path = agent_runs_dir(cwd, model, run_id)
                extra_info = {"traceback": traceback.format_exc()}
                if not traj_path.exists():
                    traj_path.mkdir(parents=True, exist_ok=True)

                traj_file = traj_path / f"{instance_id}.traj.json"
                save_traj(agent=agent, path=traj_file, exit_status=exit_status, result=result, extra_info=extra_info)
            if docker_env is not None:
                try:
                    docker_env.cleanup()
                except Exception as e:
                    print(f"Warning: Cleanup failed: {e}")
    
    root = Path(trajectory_folder)
    if not root.exists():
        print(f"ERROR: Trajectory Path {trajectory_folder} does not exist")
        return 1
    
    pred_dir = predictions_dir(cwd)
    if not pred_dir.exists():
        pred_dir.mkdir(parents=True, exist_ok=True)
    
    predictions = pred_dir / f"{run_id}.{JSONL}"
    if predictions.exists():
        os.remove(predictions)

    max_workers = max(1, max_workers)
    prediction_lock = Lock()
    instances = []
    paths = [p for p in root.iterdir() if p.is_dir()]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _run_instance_from_commands,
                p,
                lock=prediction_lock,
                cwd=cwd,
                model=model,
                run_id=run_id,
            )
            for p in paths
        ]
        for future in as_completed(futures):
            inst = future.result()
            if inst:
                instances.append(inst)

    if not instances:
        print("WARNING: No instances completed successfully; skipping evaluation.")
        return 1
    
    #organize trajectories
    traj_path = agent_runs_dir(cwd, model, run_id)
    reorganize_trajectories(traj_path)

    predictions = str(predictions)
    return run_swe_bench(
        dataset_name = DATASET,
        split = "test",
        instance_ids=instances,
        predictions_path = predictions,
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

    
def run_replay_agent_swe(
    cwd: Path,
    trajectory_folder: Path, 
    model: str, 
    run_id: str, 
    recovery_mode: str = "full_history",
    max_workers: int = 1
) -> int:
    """
    Core algorithm for running the recovery/replay swe agent

    Inputs:
        cwd (Path) - Recovery-Bench swe root
        trajectory_folder (Path) - path to dirty state trajectories
        model (str) - desired LiteLLM compatible model to benchmark
        run_id (str) - unique run identifier from user input
        max_workers (int) - number of workers to parallelize task executions

    Returns:
        0 (int) - indicates successful evalaution
        1 (int) - indicates unsuccessful evaluation
    """
    
    def _run_instance_from_commands(
            p: Path,
            *,
            lock: Lock,
            cwd: Path,
            model: str,
            run_id: str,
            recovery_mode: str,
        ):
        docker_env, agent = None, None
        exit_status, result, extra_info = None, None, None
        instance_id = None
        try:
            commands_path = p / "commands.json"
            if not commands_path.exists():
                return None
            commands = get_data(commands_path, JSON)
            if commands == [] or not isinstance(commands, dict):
                return None
            instance_id = commands.get("instance")
            if not instance_id:
                return None
            traj_path = p / f"{instance_id}.traj.{JSON}"
            if not traj_path.exists():
                return None
            traj = get_data(traj_path, JSON)
            if traj == 1:
                return None

            cfg = get_cfg(traj)
            if cfg == 1:
                print("ERROR: Couldn't parse config from trajectory")
                return None
            model_cfg = make_model_cfg(model)
            if model_cfg == 1:
                print("ERROR: Please enter a supported LiteLLM model name")
                return None
            cfg["model"] = model_cfg
            agent_model = get_model(model, model_cfg)

            env = cfg["environment"]
            docker_env = DockerEnvironment(**env)

            msgs = get_message_history(traj, recovery_mode)
            summary = create_summary(msgs, recovery_mode)
            inject_recovery(cfg, recovery_mode, summary)

            _agent_cfg = get_agent_config(cfg)
            agent = RecoverySWEAgent(
                model=agent_model,
                env=docker_env,
                messages=msgs,
                **_agent_cfg)

            commands_lst = commands.get("commands", [])
            if not commands_lst:
                return None
            i = 0
            for command in commands_lst:
                print(f"Restoring State of {instance_id} with command: {command}, {i} \n\n")
                agent.env.execute(command)
                i += 1

            p_task = p / f"task.{TXT}"
            task = get_data(p_task, TXT)

            exit_status, result = agent.run(task)

            print("\n\n\n\n\n")
            print("Recovery Agent Returned Patch:")
            print(result)

            with lock:
                written = write_predictions(cwd, instance_id, model, result, run_id)
            if written:
                print(f"ERROR: Couldn't write predictions for instance {instance_id}")
                print("\n\n\n")
                return None
            return instance_id
        except Exception as e:
            exit_status, result = type(e).__name__, str(e)
            extra_info = {"traceback": traceback.format_exc()}
            print(f"ERROR processing instance: {e}")
            return None
        finally:
            if instance_id is not None:
                traj_path = agent_runs_dir(cwd, model, run_id)
                extra_info = {"traceback": traceback.format_exc()}
                if not traj_path.exists():
                    traj_path.mkdir(parents=True, exist_ok=True)

                traj_file = traj_path / f"{instance_id}.traj.json"
                save_traj(agent=agent, path=traj_file, exit_status=exit_status, result=result, extra_info=extra_info)
            if docker_env is not None:
                try:
                    docker_env.cleanup()
                except Exception as e:
                    print(f"Warning: Cleanup failed: {e}")
    
    root = Path(trajectory_folder)
    if not root.exists():
        print(f"ERROR: Trajectory Path {trajectory_folder} does not exist")
        return 1
    
    pred_dir = predictions_dir(cwd)
    if not pred_dir.exists():
        pred_dir.mkdir(parents=True, exist_ok=True)
    
    predictions = pred_dir / f"{run_id}.{JSONL}"
    if predictions.exists():
        os.remove(predictions)

    max_workers = max(1, max_workers)
    prediction_lock = Lock()
    instances = []
    paths = [p for p in root.iterdir() if p.is_dir()]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _run_instance_from_commands,
                p,
                lock=prediction_lock,
                cwd=cwd,
                model=model,
                run_id=run_id,
                recovery_mode=recovery_mode,
            )
            for p in paths
        ]
        for future in as_completed(futures):
            inst = future.result()
            if inst:
                instances.append(inst)

    if not instances:
        print("WARNING: No instances completed successfully; skipping evaluation.")
        return 1
    
    #organize trajectories
    traj_path = agent_runs_dir(cwd, model, run_id)
    reorganize_trajectories(traj_path)

    predictions = str(predictions)
    return run_swe_bench(
        dataset_name = DATASET,
        split = "test",
        instance_ids=instances,
        predictions_path = predictions,
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
    