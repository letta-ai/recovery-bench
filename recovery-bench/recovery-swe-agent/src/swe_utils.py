from typing import List
from pathlib import Path
import json
import re
import json
from minisweagent.environments.docker import DockerEnvironment
from swe_replay_agent import RecoverySWEAgent
from minisweagent.models import get_model
import subprocess
import shutil
import os

#Using these global variables for now, integrate absolute paths later when deciding on entire recovery-bench structure
JSON = "json"
JSONL = "jsonl"
TXT = "txt"
PREDICTIONS_PATH = Path("predictions")
DATASET = "princeton-nlp/SWE-bench_Verified"

#For when full dataset is available, for now testing with individual instances
EVAL_RESULTS_PATH = "evaluation_results"

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
    return commands

def get_data(path_to_file: Path, file_type: str):
    if not path_to_file.exists():
        print(f"ERROR: Path does not exit: {path_to_file}")
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

#For turning trajectory config into the default agent config, can define a recovery agent config potentially
def get_agent_config(cfg: dict) -> dict:
    if not cfg:
        print("ERROR: Please give get_agent_config proper config dictionary")
        return 1
    temp = cfg["agent"]
    agent_cfg = {"system_template": temp["system_template"],
                 "instance_template": temp["instance_template"],
                 "timeout_template": temp["timeout_template"],
                 "format_error_template": temp["format_error_template"],
                 "action_observation_template": temp["action_observation_template"],
                 "step_limit": temp["step_limit"],
                 "cost_limit": temp["cost_limit"]
                 }
    cfg["agent"] = agent_cfg
    return agent_cfg

#Puts recovery agent patch in swebench evaluator format
### Right now written for one instance, refactor and scale to recovery-bench dataset 
def write_predictions(instance_id: str, 
                      model: str, 
                      patch: str,
                      run_id: str) -> int:
    to_write = {
                "instance_id": instance_id,
                "model_name_or_path": model,
                "model_patch": patch
                }
    path_to_predictions = str(PREDICTIONS_PATH / f"{run_id}.{JSONL}")
    try:
        with open(path_to_predictions, "w") as f:
            f.write(json.dumps(to_write))
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
def get_result(model: str, run_id) -> int:
    model_name = safe_model_name(model)
    run_path = Path(EVAL_RESULTS_PATH / f"{model_name}.{run_id}.{JSON}")
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

def run_swe_bench(instance_id: str, 
                  run_id: str,
                  max_workers: int=1,
                  ):
    #Make predictions path using run_id
    print("\n\n\n\n\n")
    print("Running SWEBench Evaluator")
    predictions_path = PREDICTIONS_PATH / f"{run_id}.{JSONL}"
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
        cwd = os.getcwd()
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
        print(result.stdout)
        return 0
    except Exception as e:
        print(f"ERROR: Couldn't run swe bench on prediction for instance {instance_id}")
        return 1


def run_replay_agent_swe(trajectory_folder: Path, 
                         model: str, 
                         run_id: str, 
                         recovery_mode: str = "full_history",
                         max_workers: int = 1) -> int:
    root = Path(trajectory_folder)
    if not root.exists():
        print(f"ERROR: Trajectory Path {trajectory_folder} does not exist")
        return 1
    for p in root.iterdir():
        traj_path, commands_path = p / "last_swebench_single_run.traj.json", p / "commands.json"
        #Check if trajectories and commands exist
        if not traj_path.exists() or not commands_path.exists():
            continue
        traj, commands = get_data(traj_path, JSON), get_data(commands_path, JSON)
        #Data didn't load properly, try for next trajectory or commands didn't load
        if traj == 1 or commands == []:
             continue
        
        #Create Config and set up Model Configuration
        cfg = get_cfg(traj)
        if cfg == 1:
            print("ERROR: Couldn't parse config from trajectory")
            return 1
        model_cfg = make_model_cfg(model)
        if model_cfg == 1:
            print("ERROR: Please enter a supported LiteLLM model name")
            return 1
        cfg["model"] = model_cfg
        agent_model = get_model(model, model_cfg)
        
        #Create Docker Environment
        env = cfg["environment"]
        docker_env = DockerEnvironment(**env)

        #Get message history based on recovery_mode
        msgs = get_message_history(traj, recovery_mode)
        summary = create_summary(msgs, recovery_mode)
        inject_recovery(cfg, recovery_mode, summary)

        #Create Agent
        _agent_cfg = get_agent_config(cfg)
        agent = RecoverySWEAgent(
            model = agent_model, 
            env = docker_env, 
            messages = msgs,
            **_agent_cfg)

        #Restore the dirty state
        commands_lst = commands["commands"]
        instance_id = commands["instance"]

        #Keep track of command numbers to verify if correct commands are being executed for state
        i = 0
        for command in commands_lst:
            print(f"Restoring State of {instance_id} with command: {command}, {i}")
            agent.env.execute(command)
            i += 1
        
        #Run Task 
        p_task = p / f'task.{TXT}'
        task = get_data(p_task, TXT)
        exit_status, result = agent.run(task)
        print("\n\n\n\n\n")
        print("Recovery Agent Returned Patch:")
        print(result)

        #Write predictions and check
        if (not write_predictions(instance_id, model, result, run_id)):
            continue

        #Run SWE Bench evaluator
        ### Until Dataset is guaranteed, just run on singular instance for testing
        return run_swe_bench(instance_id, run_id, max_workers)
    return run_swe_bench(instance_id, run_id, max_workers)