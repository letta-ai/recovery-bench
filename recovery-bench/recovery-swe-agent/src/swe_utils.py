import json
import re
import json
import subprocess
import shutil
import os
import sys
from typing import List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from minisweagent.environments.docker import DockerEnvironment
from .swe_replay_agent import RecoverySWEAgent
from minisweagent.models import get_model
from swebench.harness.run_evaluation import main as run_swe_bench

#Using these global variables for now, integrate absolute paths later when deciding on entire recovery-bench structure
JSON = "json"
JSONL = "jsonl"
TXT = "txt"
DATASET = "princeton-nlp/SWE-bench_Verified"

#Predictions and Evaluation-Results Paths relative to recovery-swe-agent directory
PREDICTIONS_PATH = Path("predictions")
EVAL_RESULTS_PATH = Path("evaluation-results")
END_CMD = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached"

#Helper for parsing commands to store in separate commands.json file
def get_commands(data: dict) -> list: 
    commands = []
    messages = data["messages"]
    #Trajectories have system prompt and instance prompt in messages first, iterate over the rest to parse actions
    for i in range(2, len(messages)):
        curr_response = messages[i]
        action = re.findall(r"```bash\s*\n(.*?)\n```", curr_response["content"], re.DOTALL)
        if len(action) == 1:
            commands.append(action[0].strip())

    # Ensure that the last command isn't a submission command
    if (len(commands) > 1) and commands[-1] == END_CMD:
        commands.pop()
    return commands

def get_data(path_to_file: Path, file_type: str):
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
    if not traj:
        print("ERROR: cfg not recieved")
        return 1
    return traj["info"]["config"]

def make_model_cfg(model: str, *, temperature: float = 0.0, extra: dict | None = None) -> dict:
    #Can do more robust model checking for LiteLLM compatability
    if not model:
        print("ERROR: Please enter a valid model")
        return 1
    mk = {"drop_params": True, "temperature": temperature}
    #Overide or add extra kwargs to mk
    if extra:
        mk |= extra
    return {"model_name": model, "model_kwargs": mk, "litellm_model_registry": None}

def get_message_history(data: dict, recovery_mode: str) -> List[dict]:
    if not data:
        print("ERROR: Proper data wasn't inputted to get_message_history")
        return 1
    if recovery_mode == "environment_only":
        return []
    elif recovery_mode == "action_summary":
        #TODO
        #Use LLM to summarize actions
        return "It's over 9000"
    elif recovery_mode == "full_history":
        #Skip over system template and instance template, agent will re add during run with injected system template
        return data['messages'][2:] if data['messages'] else 1
    else:
        print(f"ERROR: Please enter valid recovery_mode, user entered {recovery_mode}")
        return 1
    
def create_summary(messages: str, recovery_mode: str):
    #TODO, have to architect the type of summary based on recovery_mode
    return ""
    
def inject_recovery(cfg: dict, recovery_mode: str, summary: str) -> None:
    if recovery_mode == "action_summary" and summary:
        cfg["agent"]["system_template"] = (
            "You are taking over a partially attempted SWE-bench task.\n"
            f"Prior attempt summary:\n{summary}\n---\n"
        ) + cfg["agent"]["system_template"]
    elif recovery_mode == "full_history":
        cfg["agent"]["system_template"] = "You are taking over a partially attempted SWE-bench task.\n" + cfg["agent"]["system_template"]
    else:
        # guard_rail = (
        #     "# TEST-GATE (hard requirement)\n"
        #     "Do NOT submit unless the test suite has just passed with exit code 0.\n"
        #     "You must first run the tests (e.g., `pytest -q`) and confirm success.\n"
        #     "If tests fail, continue iterating; do NOT submit.\n"
        #     "Do not submit in the same turn as you run the tests. Submit only in a subsequent turn after a passing run.\n"
        # )
        cfg["agent"]["system_template"] = (
        "You are taking over a partially attempted SWE-bench task. "
        "A previous agent tried to solve this but failed. "
        "The codebase may contain their incomplete or incorrect changes. "
        "The working tree may contain incomplete or incorrect changes.\n\n"
        ) + cfg["agent"]["system_template"]
        print(cfg["agent"]["system_template"])


#For turning trajectory config into the default agent config, can define a recovery agent config potentially
def get_agent_config(cfg: dict) -> dict:
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

#Puts recovery agent patch in swebench evaluator format
def write_predictions(
        cwd: Path,
        instance_id: str, 
        model: str, 
        patch: str,
        run_id: str
    ) -> int:
    to_write = {
                "instance_id": instance_id,
                "model_name_or_path": model,
                "model_patch": patch
                }
    path_to_predictions = str(cwd / PREDICTIONS_PATH / f"{run_id}.{JSONL}")
    try:
        with open(path_to_predictions, "a") as f:
            f.write(json.dumps(to_write) + '\n')
        return 0
    except Exception as e:
        print("ERROR: Couldn't write predictions to json")
        return 1
    
#For organizing swebench results
def safe_model_name(s: str) -> str:
    s = s.replace("/", "__")             
    return re.sub(r"[^A-Za-z0-9._-]", "-", s)

#Moves swe bench run to evaluation results
def move_eval_result(model: str, run_id: str) -> int:
    model_name = safe_model_name(model)
    run_path = Path(f"{model_name}.{run_id}.{JSON}")
    if not run_path.exists():
        print(f"ERROR: Path {run_path} to swe-bench run does not exist")
        return 1
    shutil.move(run_path, EVAL_RESULTS_PATH)
    return 0

#Get the result and return
def get_result(cwd: Path, model: str, run_id) -> int:
    model_name = safe_model_name(model)
    run_path = Path(cwd / EVAL_RESULTS_PATH / f"{model_name}.{run_id}.{JSON}")
    try:
        with open(str(run_path), "r") as f:
            results = json.load(f)
            total_inst, resolv_inst = results["total_instances"], results["resolved_instances"]
            print("\n\n")
            print("Recovery Agent Results: ", round(resolv_inst / total_inst, 2))
            return 0
    except Exception as e:
        print(f"ERROR: Couldn't find path or open result {run_path}")
        return 1

def restore_state(env : DockerEnvironment, patch : str) -> int: 
    """
    Restores the dirty state, returns 0 on successful restoration, 1 on fail.
    Assumes caller routine will clean env in case of failure.
    """
    if not env:
        print(f"ERROR: Can't restore environment with no valid environment input to restore_state\n")
        return 1
    if not patch or not patch.strip():
        print(f"ERROR: Can't restore environment if weak agent didn't provide patch")
        return 1
    
    #Write the patch to tmp file and check successfully wrote (redirect heredoc to write to /tmp/recovery.patch)
    write_patch = f"cat << 'PATCHEOF' > /tmp/recovery.patch\n{patch}\nPATCHEOF"
    result = env.execute(write_patch)
    if result.get("returncode") != 0:
        print("ERROR: Failed to create tmp file for patch in restore_state\n")
        print(result.get("output"))
        return 1
    
    #Apply patch to the docker image to restore state
    apply_patch = "cd /testbed && git apply /tmp/recovery.patch"
    result = env.execute(apply_patch)
    if result.get("returncode") != 0:
        print("ERROR: Failed to apply the patch in restore_state\n")
        print(result.get("output"))
        return 1
    
    #For testing, making sure patch was applied
    print("Getting patch statistics...\n\n")
    stats_cmd = "cd /testbed && git diff --stat"
    result = env.execute(stats_cmd)
    if result.get("returncode") != 0:
        print("ERROR: Failed to get diff stats")
        return 1
    stats = result.get("output", "").strip()
    if not stats:
        print("WARNING: No changes detected after patch application")
        return 1
    
    print(f"Patch statistics:\n{stats}")
    
    #Remove tmp file to give recovery agent exact failed state of weak agent
    remove_patch = "rm -f /tmp/recovery.patch"
    result = env.execute(remove_patch)
    if result.get("returncode") != 0:
        print("ERROR: Failed to remove /tmp/recovery.patch in restore_state\n")
        print(result.get("output"))
        return 1
    return 0

def get_diff(traj: dict):
    """
    Gets the submission patch from input trajectory, returns 1 if couldn't get
    the patch. Returns the patch otherwise.
    """
    patch = traj.get("info", {}).get("submission", {})
    if not patch or not patch.strip():
        return 1
    return patch


def run_swe_bench_instance(
        cwd: Path,
        instance_id: str, 
        run_id: str,
        max_workers: int=1,
    ):
    """
    A helper function to run a singular swe bench instance which saves the the result
    in a json file in CWD, useful for testing/debugging
    """
    #Make predictions path using run_id
    print("\n\n\n\n\n")
    print("Running SWEBench Evaluator")
    predictions_path = cwd / PREDICTIONS_PATH / f"{run_id}.{JSONL}"
    cmd = [
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "--predictions_path", str(predictions_path),
        "--instance_ids", instance_id,
        "--max_workers", str(max_workers),
        "--run_id", f"{run_id}",
        "--dataset", DATASET
    ]
    #SWE_Bench_Verified for now as that's where toy dirty state came from
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
        print(result.stdout)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with return code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")  # THIS is where the error message is!
        return 1
    except Exception as e:
        print(f"ERROR: Couldn't run swe bench on prediction for instance {instance_id}")
        print("\n\n\n")
        print()
        return 1
    
def run_replay_agent_swe(
        cwd: Path,
        trajectory_folder: Path, 
        model: str, 
        run_id: str, 
        recovery_mode: str = "full_history",
        max_workers: int = 1
    ) -> int:
    
    def _run_instance_from_commands(
            p: Path,
            *,
            lock: Lock,
            cwd: Path,
            model: str,
            run_id: str,
            recovery_mode: str,
        ):
        docker_env = None
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
            print(f"ERROR processing instance: {e}")
            return None
        finally:
            if docker_env is not None:
                try:
                    docker_env.cleanup()
                except Exception as e:
                    print(f"Warning: Cleanup failed: {e}")
    
    root = Path(trajectory_folder)
    if not root.exists():
        print(f"ERROR: Trajectory Path {trajectory_folder} does not exist")
        return 1
    
    predictions = cwd / PREDICTIONS_PATH / f"{run_id}.{JSONL}"
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

def restore_docker_testing(
        cwd: Path,
        trajectory_folder: Path, 
        model: str, 
        run_id: str, 
        recovery_mode: str = "full_history",
        max_workers: int = 1):
    def _run_instance_from_patch(
            p: Path,
            *,
            lock: Lock,
            cwd: Path,
            model: str,
            run_id: str,
            recovery_mode: str,
        ):
        docker_env = None
        try:
            traj_path = p / "last_swebench_single_run.traj.json"
            commands_path = p / "commands.json"
            if not traj_path.exists() or not commands_path.exists():
                return None
            traj = get_data(traj_path, JSON)
            commands = get_data(commands_path, JSON)
            if traj == 1 or commands == []:
                return None
            instance_id = commands.get("instance")
            if not instance_id or instance_id == "sympy__sympy-15599":
                return None

            print("KOBE\n" * 5 )
            print(f"Image being created: {traj['info']['config']['environment']['image']}\n")
            
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
                model = agent_model, 
                env = docker_env, 
                messages = msgs,
                **_agent_cfg)
            
            patch = get_diff(traj)
            if patch == 1:
                return None

            success = restore_state(agent.env, patch)
            if success != 0:
                print(f"Failed to recover state for path: {traj_path}\n\n")
                return None

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
            print(f"ERROR processing instance: {e}")
            return None
        finally:
            if docker_env is not None:
                try:
                    docker_env.cleanup()
                except Exception as e:
                    print("Failed to cleanup docker env")

    root = Path(trajectory_folder)
    if not root.exists():
        print(f"ERROR: Trajectory Path {trajectory_folder} does not exist")
        return 1

    predictions = cwd / PREDICTIONS_PATH / f"{run_id}.{JSONL}"
    if predictions.exists():
        os.remove(predictions)

    max_workers = max(1, max_workers)
    prediction_lock = Lock()
    instances = []
    paths = [p for p in root.iterdir() if p.is_dir()]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _run_instance_from_patch,
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

def write_cmds(inst_run: Path, inst: str, cmds: list):
    to_write = {"instance": inst, "commands": cmds}
    try:
        with open(str(inst_run / "commands.json"), "w") as f:
            json.dump(to_write, f, indent=2)
        return 0
    except Exception as e:
        print(f"Error: Failed to write commands to {inst_run}")
        return 1

def get_commands_and_write(trajectory):
    trajectory1 = trajectory / "last_swebench_single_run.traj.json"
    data = get_data(trajectory1, JSON)
    commands = get_commands(data)
    commands = {"instance": "sympy__sympy-15599", "commands": commands}
    with open(str(trajectory / "commands.json"), "w") as f:
        json.dump(commands, f, indent= 2)

if __name__ == "__main__":
    p = Path("swe-runs/sympy")
    get_commands_and_write(p)
    exit()



    print("RUNNING")
    trajectory_folder = Path("swe-runs")
    recovery_mode = "environment_only"
    model = "openai/gpt-5-nano"
    restore_docker_testing(trajectory_folder, recovery_mode, model)
